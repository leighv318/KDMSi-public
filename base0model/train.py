import math
import traceback
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import warnings
warnings.filterwarnings('ignore')

import argparse
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, ndcg_score
from scipy.stats import pearsonr, spearmanr

from load_data import get_dataloader_kcross, get_dataloader_div_by_gene
from utils import *

convmcmsipred_models = __import__('model')


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=126)
parser.add_argument("--model", type=str, default='Base0_mrna_full_no_se')
parser.add_argument("--loss_type", type=str, default='ndcg', choices=['huber', 'mse', 'ndcg'])
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--sample_internal', type=int, default=10, help='the num of k_cross testation division')
parser.add_argument(
    '--data_path',
    type=str,
    default='data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv',
    help='training data csv/xlsx path'
)
parser.add_argument('--max_len', type=int, default=25, help='Fixed padded sequence length. Use 25 for pad25+flatten models.')
parser.add_argument('--log_interval', type=int, default=1, help='print validation metrics every N epochs')
parser.add_argument('--run_leave_gene', type=int, default=1, help='whether to run leave-one-gene-out after 10-fold CV')
parser.add_argument('--test_genes', type=str, default='', help='comma-separated leave-gene list. If empty, infer from data.')
parser.add_argument('--leave_gene_min_count', type=int, default=100, help='minimum rows per target_gene for inferred leave-gene tests')
parser.add_argument('--ndcg_weight', type=float, default=1e-7, help='weight for ndcg loss term when loss_type=ndcg')
parser.add_argument('--cell_embed_dim', type=int, default=8,
                    help='Dimension of cell-type source embedding (0 to disable)')
parser.add_argument('--pearson_weight', type=float, default=0.5,
                    help='Weight for differentiable Pearson loss (0 to disable)')
parser.add_argument('--as_ts', type=int, default=3,
                    help='Number of transformer steps for antisense strand (default: 3)')
parser.add_argument('--ss_ts', type=int, default=3,
                    help='Number of transformer steps for sense strand (default: 3)')
args = parser.parse_args()

init_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pearson_loss(pred, label):
    """Differentiable Pearson correlation loss = 1 - PCC."""
    pred_c  = pred  - pred.mean()
    label_c = label - label.mean()
    pcc = (pred_c * label_c).sum() / (pred_c.norm() * label_c.norm() + 1e-8)
    return 1.0 - pcc


def move_batch_to_device(batch, device):
    """Move a dict batch to device and convert all floating tensors to float32.

    Current load_data.py returns one dict per batch, with label stored in
    batch['inhibit']. This helper also tolerates old formats such as
    (inputs, labels).
    """
    if isinstance(batch, dict):
        inputs = batch
        label = batch.get('inhibit', None)
    elif isinstance(batch, (tuple, list)):
        inputs = batch[0]
        label = batch[1] if len(batch) > 1 else None
    else:
        raise TypeError(f'Unexpected batch type: {type(batch)}')

    for k in list(inputs.keys()):
        v = inputs[k]
        if torch.is_tensor(v):
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.float()
            inputs[k] = v

    if label is None:
        if 'inhibit' not in inputs:
            raise KeyError("Cannot find label. Expected batch['inhibit'].")
        label = inputs['inhibit']
    if torch.is_tensor(label):
        label = label.to(device).float()

    return inputs, label


def tensor_is_finite(name, x):
    if torch.is_tensor(x) and torch.is_floating_point(x):
        if not torch.isfinite(x).all():
            n_nan = torch.isnan(x).sum().item()
            n_inf = torch.isinf(x).sum().item()
            print(f'[BAD_INPUT] {name}: nan={n_nan}, inf={n_inf}, shape={tuple(x.shape)}', flush=True)
            return False
    return True


def safe_corr(label, pred):
    try:
        if len(label) < 2 or np.nanstd(label) == 0 or np.nanstd(pred) == 0:
            return -1, -1
        pcc = round(float(pearsonr(label, pred)[0]), 3)
        spcc = round(float(spearmanr(pred, label)[0]), 3)
        if np.isnan(pcc):
            pcc = -1
        if np.isnan(spcc):
            spcc = -1
        return pcc, spcc
    except Exception:
        return -1, -1


def compute_metrics(label, pred):
    label = np.asarray(label, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(label) & np.isfinite(pred)
    label = label[mask]
    pred = pred[mask]
    if len(label) == 0:
        return -1, -1, -1, -1

    PCC, SPCC = safe_corr(label, pred)
    try:
        NCDG = round(float(ndcg_score([label], [pred], k=min(64, len(label)))), 3)
    except Exception:
        NCDG = -1
    try:
        binary = (label >= 0.7).astype(int)
        if len(np.unique(binary)) < 2:
            AUC = -1
        else:
            AUC = round(float(roc_auc_score(binary, pred)), 3)
    except Exception:
        AUC = -1
    return PCC, SPCC, NCDG, AUC


class K_Train_Test:
    def __init__(self, args):
        self.args = args
        self.device = device

    def make_model(self):
        import inspect
        cls = getattr(convmcmsipred_models, self.args.model)
        sig = inspect.signature(cls.__init__)
        kwargs = dict(dp=0.1, device=self.device,
                      as_ts=self.args.as_ts, ss_ts=self.args.ss_ts)
        if 'cell_embed_dim' in sig.parameters:
            kwargs['cell_embed_dim'] = self.args.cell_embed_dim
        model = cls(**kwargs)
        return model.float().to(self.device)

    def evaluate_loader(self, model, loader, criterion_huber=None):
        model.eval()
        preds, labels = [], []
        huber_losses = []
        ndcg_losses = []

        with torch.no_grad():
            for batch in loader:
                inputs, label = move_batch_to_device(batch, self.device)
                pred = model(inputs)
                if not torch.isfinite(pred).all() or not torch.isfinite(label).all():
                    continue
                preds.append(pred.detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())
                if criterion_huber is not None:
                    huber_losses.append(float(criterion_huber(label, pred).detach().cpu()))
                try:
                    ndcg_losses.append(float(ncdg_loss(pred, torch.clamp(label, 0.0, 1.0), alpha=1.0).detach().cpu()))
                except Exception:
                    pass

        if len(preds) == 0:
            return -1, -1, -1, -1, float('nan'), float('nan')

        pred_np = np.concatenate(preds)
        label_np = np.concatenate(labels)
        PCC, SPCC, NCDG, AUC = compute_metrics(label_np, pred_np)
        huber_val = float(np.mean(huber_losses)) if huber_losses else float('nan')
        ndcg_val = float(np.mean(ndcg_losses)) if ndcg_losses else float('nan')
        return PCC, SPCC, NCDG, AUC, huber_val, ndcg_val

    def model_train(self, train_loader, test_loader, testset_type):
        print(f' \n\n --- testset type : {testset_type} ------\n', flush=True)
        model = self.make_model()
        print(f'[info] device={self.device}, model={self.args.model}, loss={self.args.loss_type}', flush=True)

        criterion_mse = torch.nn.MSELoss(reduction='mean')
        criterion_huber = torch.nn.HuberLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=4)

        save_path = os.path.join(f'./output_{self.args.loss_type}', self.args.model, str(testset_type))
        os.makedirs(save_path, exist_ok=True)

        best_metric = -1
        best_record = None
        epochs_no_improve = 0
        patience = 30

        for epoch in range(1, self.args.n_epochs + 1):
            model.train()
            train_losses = []
            huber_losses = []
            ndcg_losses = []

            for batch_id, batch in enumerate(train_loader):
                inputs, label = move_batch_to_device(batch, self.device)

                ok = True
                for k, v in inputs.items():
                    if not tensor_is_finite(k, v):
                        ok = False
                if not torch.isfinite(label).all():
                    print(f'[BAD_LABEL] epoch={epoch}, batch={batch_id}', flush=True)
                    ok = False
                if not ok:
                    raise RuntimeError('Non-finite input/label detected.')

                pred = model(inputs)
                if not torch.isfinite(pred).all():
                    print(f'[BAD_PRED] epoch={epoch}, batch={batch_id}, pred min/max={pred.nan_to_num().min().item()}/{pred.nan_to_num().max().item()}', flush=True)
                    raise RuntimeError('Non-finite prediction detected.')

                loss_mse = criterion_mse(label, pred)
                loss_huber = criterion_huber(label, pred)

                # NDCG expects non-negative relevance. Use clamped label only for
                # the NDCG auxiliary term; metrics still use original labels.
                ndcg_alpha = 100 / math.exp(0.01 * epoch)
                label_for_ndcg = torch.clamp(label, 0.0, 1.0)
                loss_ndcg = self.args.ndcg_weight * (1 - 1 / math.exp(0.1 * epoch)) * ncdg_loss(pred, label_for_ndcg, alpha=ndcg_alpha)

                loss_pearson = pearson_loss(pred, label) if self.args.pearson_weight > 0 else 0.0

                if self.args.loss_type == 'mse':
                    loss = loss_mse
                elif self.args.loss_type == 'huber':
                    loss = loss_huber
                elif self.args.loss_type == 'ndcg':
                    loss = loss_huber + loss_ndcg
                else:
                    raise ValueError(f'Unknown loss_type: {self.args.loss_type}')

                if self.args.pearson_weight > 0:
                    loss = loss + self.args.pearson_weight * loss_pearson

                if not torch.isfinite(loss):
                    print(f'[BAD_LOSS] epoch={epoch}, batch={batch_id}, loss={loss.item()}, huber={loss_huber.item()}, ndcg={loss_ndcg.item()}', flush=True)
                    print(f'pred min/max={pred.min().item()}/{pred.max().item()}, label min/max={label.min().item()}/{label.max().item()}', flush=True)
                    raise RuntimeError('Non-finite loss detected.')

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()

                train_losses.append(float(loss.detach().cpu()))
                huber_losses.append(float(loss_huber.detach().cpu()))
                ndcg_losses.append(float(loss_ndcg.detach().cpu()))

            scheduler.step()

            PCC, SPCC, NCDG, AUC, eval_huber, eval_ndcg = self.evaluate_loader(model, test_loader, criterion_huber)
            train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
            loss_huber_value = float(np.mean(huber_losses)) if huber_losses else float('nan')
            loss_ndcg_value = float(np.mean(ndcg_losses)) if ndcg_losses else float('nan')

            if epoch % self.args.log_interval == 0:
                msg = (f'epoch {epoch} ====>> train_loss:{train_loss:.6f}, '
                       f'pcc:{PCC}, spcc:{SPCC}, ncdg:{NCDG}, auc:{AUC}, '
                       f'huber:{eval_huber:.6f}, ndcg:{eval_ndcg:.6f}')
                logging(msg)
                print(msg, flush=True)

            monitor = SPCC if SPCC != -1 else PCC
            if monitor != -1 and monitor > best_metric:
                best_metric = monitor
                best_record = dict(epoch=epoch, PCC=PCC, SPCC=SPCC, NCDG=NCDG, AUC=AUC,
                                   huber=eval_huber, ndcg=eval_ndcg, train_loss=train_loss)
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(save_path, 'model_best.pth.tar'))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'[info] early stopping at epoch {epoch}', flush=True)
                break

        if best_record is None:
            best_record = dict(epoch=epoch, PCC=-1, SPCC=-1, NCDG=-1, AUC=-1,
                               huber=float('nan'), ndcg=float('nan'), train_loss=float('nan'))

        print(('BEST/FINAL {}  ====>> epoch:{}, pcc:{}, spcc:{}, ncdg:{}, auc:{}, '
               'huber:{}, ndcg:{}').format(
                   testset_type, best_record['epoch'], best_record['PCC'], best_record['SPCC'],
                   best_record['NCDG'], best_record['AUC'], best_record['huber'], best_record['ndcg']
               ), flush=True)

    def infer_test_genes(self):
        if self.args.test_genes.strip():
            return [g.strip() for g in self.args.test_genes.split(',') if g.strip()]
        try:
            if self.args.data_path.endswith('.csv'):
                df = pd.read_csv(self.args.data_path)
            else:
                df = pd.read_excel(self.args.data_path)
            if 'target_gene' not in df.columns:
                print('[warn] target_gene column not found; skip leave-gene tests.', flush=True)
                return []
            vc = df['target_gene'].dropna().astype(str)
            vc = vc[vc.str.strip() != ''].value_counts()
            genes = vc[vc >= self.args.leave_gene_min_count].index.tolist()
            print('[info] inferred leave-gene list:', genes, flush=True)
            print('[info] target_gene counts:\n{}'.format(vc.to_string()), flush=True)
            return genes
        except Exception as e:
            print(f'[warn] failed to infer test_genes: {e}', flush=True)
            return []

    def k_cross(self):
        for i in range(10):
            init_seed(self.args.seed)
            testset_type = str(i) + '-inter'
            train_loader, test_loader = get_dataloader_kcross(self.args, testset_type)
            self.model_train(train_loader, test_loader, testset_type)

        if self.args.run_leave_gene:
            for gene in self.infer_test_genes():
                try:
                    init_seed(self.args.seed)
                    train_loader, test_loader = get_dataloader_div_by_gene(self.args, gene)
                    self.model_train(train_loader, test_loader, gene)
                except Exception as e:
                    print(f'Skip gene {gene}: {e}', flush=True)
                    traceback.print_exc()

        log()
        print('finish !!', flush=True)


def main(args):
    k_cross_course = K_Train_Test(args)
    k_cross_course.k_cross()


if __name__ == '__main__':
    main(args)
