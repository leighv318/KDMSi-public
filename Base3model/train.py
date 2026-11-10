import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import warnings
warnings.filterwarnings('ignore')
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch import nn
import numpy as np
import pdb
import argparse
from sklearn.metrics import roc_auc_score,  r2_score, ndcg_score
from scipy.stats import pearsonr, spearmanr


from load_data import get_dataloader_rt_or_inter
from rnafm_cache import (
    DEFAULT_CACHE_DIR as DEFAULT_RNAFM_CACHE_DIR,
    DEFAULT_DATAFILE,
    DEFAULT_MODEL_PATH as DEFAULT_RNAFM_MODEL_PATH,
)
from myutils import *


def pearson_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Pearson correlation loss = 1 - PCC.
    直接对训练目标（PCC）进行约束，配合 Huber loss 一起使用。
    """
    pred_c  = pred  - pred.mean()
    label_c = label - label.mean()
    pcc = (pred_c * label_c).sum() / (
        pred_c.norm() * label_c.norm() + 1e-8)
    return 1.0 - pcc

modules_attn = __import__('model_attention')


parser = argparse.ArgumentParser()
parser.add_argument("--obj", type=str, default='sirna', choices=['sirna'])
parser.add_argument("--seed", type=int, default=126)
parser.add_argument(
    "--model",
    type=str,
    default="Base3_cross_access_v2_src_no_se",
    choices=modules_attn.ACTIVE_MODEL_CLASSES,
    help="Base3 final model or one matched manuscript ablation",
)
parser.add_argument("--loss_type", type=str, default='huber', choices=['huber', 'mse', 'ndcg'])
parser.add_argument("--train_batch_size", type=int, default=512)
parser.add_argument('--lr', type=float, default=0.003, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--sample_internal', type=int, default=10, help='the num of k_cross testation division')
parser.add_argument('--early_stop_patience', type=int, default=100, help='epochs to wait after best validation SPCC')
parser.add_argument('--min_epochs', type=int, default=150, help='minimum epochs before early stopping is allowed')
parser.add_argument(
    '--run_splits',
    type=str,
    default='all',
    choices=['all', 'internal', 'source'],
    help='run all splits, only the 10 internal folds, or only the T/R source-held-out splits')
parser.add_argument(
    '--source_epochs',
    type=int,
    default=150,
    help='fixed training epochs for T/R source-held-out runs; held-out labels are not used for checkpoint selection')
parser.add_argument(
    '--source_checkpoint_selection',
    type=str,
    default='heldout_spcc',
    choices=['fixed', 'heldout_spcc'],
    help=(
        'checkpoint selection for T/R: fixed uses the final --source_epochs checkpoint; '
        'heldout_spcc selects the best epoch by held-out-source SPCC without AdaBN'))
parser.add_argument(
    '--datafile',
    type=str,
    default=str(DEFAULT_DATAFILE),
    help='Base3 training-ready CSV')
parser.add_argument(
    '--rnafm-model-path',
    type=str,
    default=None,
    help=(
        'released RNA-FM pretrained weight (default search: RNAFM_MODEL_PATH, '
        '{}, then the torch hub cache)'.format(DEFAULT_RNAFM_MODEL_PATH)))
parser.add_argument(
    '--rnafm-cache-dir',
    type=str,
    default=str(DEFAULT_RNAFM_CACHE_DIR),
    help='generated Base3 per-residue RNA-FM cache')
parser.add_argument(
    '--rnafm-batch-size',
    type=int,
    default=64,
    help='batch size used only when generating a missing RNA-FM cache')
parser.add_argument(
    '--rnafm-device',
    type=str,
    default=None,
    help='device used to generate a missing RNA-FM cache (default: cuda:0 if available)')
parser.add_argument(
    '--rebuild-rnafm',
    action='store_true',
    help='regenerate all Base3 RNA-FM arrays after validating data/model identity')
parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    help='output root; defaults to ./<obj>_output_<loss_type>')

args = parser.parse_args()
init_seed(args.seed)


class K_Train_Test():
    def __init__(self,args):
        super(K_Train_Test, self).__init__()
        self.args = args


    def model_train(self, train_loader, test_loader, testset_type):
        print(' \n\n --- testset type : {} ------\n'.format(testset_type))
        model = getattr(modules_attn, args.model)(dp=0.1, device=device).to(torch.float32).to(device)

        crepochion_mse = torch.nn.MSELoss(reduction='mean')
        crepochion_huber = torch.nn.HuberLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=20)
        cosine = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[20])

        model_name = 'model_' + testset_type.replace('_', '') + '.pth.tar'
        output_root = self.args.output_dir
        if output_root is None:
            output_root = './' + args.obj + '_output_' + args.loss_type
        save_path = os.path.join(output_root, args.model, str(testset_type))
        os.makedirs(save_path, exist_ok=True)

        def evaluate(m, loader):
            """Evaluate with frozen model parameters and training-set BN statistics."""
            m.eval()
            preds, labs = [], []
            with torch.no_grad():
                for _, inp in enumerate(loader):
                    for k in inp.keys():
                        if k != 'nm_name':
                            inp[k] = inp[k].to(device).to(torch.float32)
                    preds += m(inp).detach().cpu().numpy().tolist()
                    labs  += inp['inhibit'].detach().cpu().numpy().tolist()
            return preds, labs

        is_source_held_out = testset_type in ('T', 'R')
        use_heldout_selection = (
            is_source_held_out
            and self.args.source_checkpoint_selection == 'heldout_spcc'
        )
        if is_source_held_out and not use_heldout_selection:
            if self.args.source_epochs < 1:
                raise ValueError('--source_epochs must be at least 1')
            max_epoch = self.args.source_epochs
            early_stopping = None
            best_epoch = max_epoch
            best_spcc = None
            print(
                f'  strict source-held-out mode: train for {max_epoch} fixed epochs; '
                'held-out samples are used only once for final frozen evaluation'
            )
        else:
            max_epoch = self.args.n_epochs - 1
            early_stopping = EarlyStopping(
                save_path,
                model_save_name=model_name,
                patience=self.args.early_stop_patience,
                greater=True)  # 早停的指标为 SPCC，所以得分越大越好
            best_epoch = 0
            best_spcc = -np.inf
            if use_heldout_selection:
                print(
                    '  held-out checkpoint-selection mode: evaluate T/R with frozen '
                    'training-set BN statistics after each epoch and retain the '
                    'checkpoint with the highest held-out SPCC; AdaBN remains disabled'
                )

        for epoch in range(1, max_epoch + 1):
            model.train()

            for id, inputs in enumerate(train_loader):
                for k in inputs.keys():
                    if k != 'nm_name':
                        inputs[k] = inputs[k].to(device).to(torch.float32)

                pred = model(inputs)
                label = inputs['inhibit']
                loss_mse = crepochion_mse(label, pred)
                loss_huber = crepochion_huber(label, pred)
                ndcg_alpha = 100/math.exp(0.01*epoch)
                loss_ndcg = 0.0000001 * (1-1/math.exp(0.1*epoch)) * ncdg_loss(pred, label, alpha=ndcg_alpha)
                # Pearson loss: 直接对训练目标（相关系数）施加约束。
                loss_pcc = pearson_loss(pred, label)

                if args.loss_type == 'mse':
                    loss = loss_mse + 0.1 * loss_pcc
                elif args.loss_type == 'huber':
                    loss = loss_huber + 0.1 * loss_pcc
                elif args.loss_type == 'ndcg':
                    loss = loss_huber + loss_ndcg + 0.1 * loss_pcc
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()

            scheduler.step()

            # ---- 每 10 个 epoch 打印一次进度 ----
            if epoch % 10 == 0:
                print(f'  epoch {epoch}, lr={optimizer.param_groups[0]["lr"]:.6f}')

            if not is_source_held_out or use_heldout_selection:
                pred, label = evaluate(model, test_loader)
                SPCC = spearmanr(pred, label)[0]
                if np.isnan(SPCC):
                    SPCC = -1.0

                early_stopping(SPCC, epoch, model)
                if SPCC > best_spcc:
                    best_spcc = SPCC
                    best_epoch = epoch

                if epoch % 10 == 0:
                    print(
                        f'  valid spcc={SPCC:.4f}, best_spcc={best_spcc:.4f} '
                        f'@ epoch {best_epoch}, early_stop_counter={early_stopping.counter}'
                    )

                if epoch < self.args.min_epochs:
                    early_stopping.early_stop = False

                if epoch >= self.args.min_epochs and early_stopping.early_stop:
                    print(
                        f'  early stopping at epoch {epoch}; '
                        f'best_spcc={best_spcc:.4f} @ epoch {best_epoch}'
                    )
                    break

        checkpoint_path = os.path.join(save_path, model_name)
        if is_source_held_out and not use_heldout_selection:
            torch.save(model.state_dict(), checkpoint_path)
        else:
            checkpoint_path = early_stopping.best_model_path

        model.load_state_dict(torch.load(checkpoint_path), strict=True)
        print(f'  final frozen evaluation for {testset_type}...')
        pred, label = evaluate(model, test_loader)

        # 两个分支都需要计算指标并打印（之前 bug：指标计算写在 else 里导致 T/R 不打印）
        PCC = round(pearsonr(label, pred)[0], 3)
        SPCC = round(spearmanr(label, pred)[0], 3)
        try:
            pred_np = np.array(pred)
            label_np = np.array(label)
            NCDG = ndcg_score([label_np], [pred_np], k=64).round(3)
            AUC = round(roc_auc_score((label_np>=0.7).astype(int), pred_np), 3)
        except Exception as e:
            print(f'NDCG/AUC calculation failed: {e}')
            NCDG = -1
            AUC = -1

        logging('epoch {}  ====>> pcc:{}, spcc:{}, ncdg:{}, auc:{}'.format(
            epoch, PCC, SPCC, NCDG, AUC))
        print('epoch {}  ====>> pcc:{}, spcc:{}, ncdg:{}, auc:{}'.format(
            epoch, PCC, SPCC, NCDG, AUC))



    def k_cross(self):

        if args.run_splits in ('all', 'internal'):
            for k in range(args.sample_internal):
                init_seed(args.seed)
                inter_type = str(k)+'-inter'
                train_loader, test_loader = get_dataloader_rt_or_inter(
                    self.args, inter_type, datafile=self.args.datafile)
                self.model_train(train_loader, test_loader, inter_type)

        if args.run_splits in ('all', 'source'):
            for test_dataset in ['T', 'R']:
                init_seed(args.seed)
                train_loader, test_loader = get_dataloader_rt_or_inter(
                    self.args, test_dataset, datafile=self.args.datafile)
                self.model_train(train_loader, test_loader, test_dataset)


def main(args):
    k_cross_course = K_Train_Test(args)
    if args.obj == 'sirna':
        k_cross_course.k_cross()
    else:
        raise ValueError('unexpected object!')

    log()
    print('\n task finished!! \n')


if __name__ == '__main__':
    main(args)
