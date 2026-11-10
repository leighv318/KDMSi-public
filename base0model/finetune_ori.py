"""
finetune_ori.py — Few-shot leave-gene finetuning
================================================

从 leave-gene-out 预训练好的 checkpoint 出发，用目标基因内随机抽取的 N 条样本进行
finetune，剩余样本作为 validation set 评估效果。

策略
----
Strategy A  对 Base0_mrna_full_no_se 只更新 head (bn + DNN)，backbone 冻结且设为 eval()。
            骨干网络没有 BatchNorm（Conv 无 BN，Transformer 用 LayerNorm），
            所以 eval() 主要冻结 dropout，不影响 BN running stats。

PSSM 处理
---------
PSSM 用「其他所有基因」的序列计算，与 leave-gene-out 预训练时保持一致。
浓度归一化统计量同理。

用法示例
--------
python finetune_ori.py \\
    --gene all \\
    --model Base0_mrna_full_no_se \\
    --pretrained_root ../liubin_pipeline/potency_prediction/pretrain_models/Base0_mrna_full_no_se \\
    --mrna_rnafm_path ./data/mrna_rnafm_22622_maxlen65.npy \\
    --n_finetune 300 \\
    --seeds 42,123,456
"""

import math
import os
import sys
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, ndcg_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PRETRAINED_ROOT = (
    PROJECT_ROOT
    / 'liubin_pipeline'
    / 'potency_prediction'
    / 'pretrain_models'
    / 'Base0_mrna_full_no_se'
)
DEFAULT_DATA_PATH = SCRIPT_DIR / 'data' / 'CMsiRNAdb_Dsicm10_FINAL_trainready.csv'
DEFAULT_MRNA_RNAFM_PATH = SCRIPT_DIR / 'data' / 'mrna_rnafm_22622_maxlen65.npy'

# ── 引入项目内模块 ────────────────────────────────────────────────────────────
from load_data import (
    _load_data,
    create_pssm,
    Generate_dataset,
    effective_cm_len,
    MRNA_THERMO_COLS,
    CELL_TYPE2IDX,
)
import load_data as LD
from utils import init_seed, ncdg_loss

convmcmsipred_models = __import__('model')


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='Few-shot gene-specific finetuning')

    # 数据 & 模型
    parser.add_argument('--gene', type=str, default='all',
                        help='Target gene name, or "all" to finetune every available leave-gene checkpoint')
    parser.add_argument('--pretrained_path', type=str, default='',
                        help='Optional checkpoint path for a single gene. With --gene all, checkpoints '
                             'are discovered under pretrained_root.')
    parser.add_argument('--pretrained_root', type=str,
                        default=str(DEFAULT_PRETRAINED_ROOT),
                        help=('Root containing <GENE>/model_best.pth.tar leave-gene checkpoints '
                              '(default: the local checkpoint directory described in README.md)'))
    parser.add_argument('--model', type=str, default='Base0_mrna_full_no_se',
                        help='Model class name (must match pretrained checkpoint)')
    parser.add_argument('--data_path', type=str,
                        default=str(DEFAULT_DATA_PATH),
                        help='Path to the full training CSV')
    parser.add_argument('--max_len', type=int, default=25,
                        help='Padded sequence length (must match pretrain)')
    parser.add_argument('--mrna_rnafm_path', type=str,
                        default=os.environ.get(
                            'MRNA_RNAFM_PATH', str(DEFAULT_MRNA_RNAFM_PATH)),
                        help='Per-row local-mRNA RNA-FM cache aligned to data_path')

    # Finetune 设置
    parser.add_argument('--n_finetune', type=int, default=300,
                        help='Number of samples to use for finetuning (rest → validation). '
                             'Automatically capped at max_finetune_pct of gene total.')
    parser.add_argument('--max_finetune_pct', type=float, default=0.5,
                        help='Max fraction of a gene\'s data used for finetuning (default 0.5). '
                             'Actual n_finetune = min(n_finetune, int(N * max_finetune_pct)). '
                             'Prevents small genes (MAPT/APP/PLN) from having no validation set.')
    parser.add_argument('--strategy', type=str, default='A', choices=['A'],
                        help='Finetuning strategy: A=head-only (backbone frozen)')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated random seeds, e.g. 42,123,456')

    # 优化器 & 训练
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for finetuning (300 samples → ~9 batches at bs=32)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Maximum number of finetuning epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (monitored: SPCC on validation)')
    parser.add_argument('--ndcg_weight', type=float, default=1e-7,
                        help='Weight for NDCG auxiliary loss term')

    # 模型架构参数（必须与预训练 checkpoint 一致）
    parser.add_argument('--cell_embed_dim', type=int, default=8,
                        help='Cell-type embedding dim (must match pretrained model)')
    parser.add_argument('--as_ts', type=int, default=3,
                        help='Transformer steps for antisense strand (must match pretrained model)')
    parser.add_argument('--ss_ts', type=int, default=3,
                        help='Transformer steps for sense strand (must match pretrained model)')

    # 输出
    parser.add_argument('--output_dir', type=str, default='output_finetune_leave_gene',
                        help='Root directory for saving finetuned checkpoints and logs')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='Print validation metrics every N epochs')

    args = parser.parse_args()

    # 设置默认学习率
    if args.lr is None:
        args.lr = 1e-4

    args.seeds = [int(s.strip()) for s in args.seeds.split(',')]
    return args


# ═══════════════════════════════════════════════════════════════════════════════
# 数据工具
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_concentration_nm(data):
    """从 DataFrame 中提取浓度列（nM），缺失值填 10。"""
    cand_cols = ['concentration/nM', 'concentration_nM', 'concentration_nm', 'Concentration_nM']
    found = None
    for c in cand_cols:
        if c in data.columns:
            found = c
            break
    if found is None:
        return pd.Series([10.0] * len(data), index=data.index, dtype=float)
    conc = pd.to_numeric(data[found], errors='coerce').fillna(10.0)
    return conc.astype(float).clip(lower=1e-9)


def _getdict(data):
    """把 DataFrame 转成 Generate_dataset 需要的 dict 格式。"""
    d = {
        'ori_as':  np.array(data['ori_as'].astype(str).str.strip()),
        'ori_ss':  np.array(data['ori_ss'].astype(str).str.strip()),
        'cm_as':   np.array(data['cm_as'].astype(str).str.strip()),
        'cm_ss':   np.array(data['cm_ss'].astype(str).str.strip()),
        'inhibition': np.array(
            data['inhibit/0-1'] if 'inhibit/0-1' in data.columns else data['norm_inhibit'],
            dtype=np.float32
        ),
        'concentration_nM': np.array(_extract_concentration_nm(data), dtype=np.float32),
    }
    if 'rnafm_ind' in data.columns:
        d['rnafm_ind'] = np.array(data['rnafm_ind'], dtype=int)
    if all(c in data.columns for c in MRNA_THERMO_COLS):
        d['mrna_thermo'] = np.array(
            data[MRNA_THERMO_COLS].apply(pd.to_numeric, errors='coerce').values,
            dtype=np.float32
        )
    if 'cell_type_idx' in data.columns:
        d['cell_type_idx'] = np.array(data['cell_type_idx'], dtype=np.int64)
    elif 'Cell_Type' in data.columns:
        d['cell_type_idx'] = np.array(
            data['Cell_Type'].map(CELL_TYPE2IDX).fillna(0), dtype=np.int64
        )
    return d


def configure_mrna_inputs(all_data, model_name, mrna_rnafm_path):
    """Load and strictly validate mRNA inputs required by Base0_mrna models."""
    if not model_name.startswith('Base0_mrna'):
        return

    if LD.model_requires_mrna_thermo(model_name):
        missing = [c for c in MRNA_THERMO_COLS if c not in all_data.columns]
        if missing:
            raise KeyError(
                f'{model_name} requires mRNA thermodynamic columns; missing={missing}'
            )

        thermo = all_data[MRNA_THERMO_COLS].apply(pd.to_numeric, errors='coerce')
        bad_rows = ~np.isfinite(thermo.to_numpy(dtype=float)).all(axis=1)
        if bad_rows.any():
            examples = np.where(bad_rows)[0][:10].tolist()
            raise ValueError(
                f'Non-finite mRNA thermodynamic features in {int(bad_rows.sum())} rows; '
                f'example row indices={examples}'
            )

    if LD.model_requires_mrna_rnafm(model_name):
        cache_path = Path(mrna_rnafm_path).expanduser()
        if not cache_path.exists():
            raise FileNotFoundError(f'mRNA RNA-FM cache not found: {cache_path}')
        cache = np.load(cache_path, mmap_mode='r')
        expected_shape = (len(all_data), 65, 640)
        if cache.shape != expected_shape:
            raise ValueError(
                f'mRNA RNA-FM shape mismatch: got {cache.shape}, expected '
                f'{expected_shape}. The cache must match data_path row-for-row.'
            )

        LD.MRNA_RNAFM = cache
        LD.MRNA_MAX_LEN = cache.shape[1]
        print(f'[finetune] validated mRNA RNA-FM: {cache.shape} from {cache_path}',
              flush=True)


def build_loaders(all_data, gene, n_finetune, seed, max_len, batch_size, max_finetune_pct=0.5):
    """
    构建 finetune_loader 和 val_loader。

    PSSM 和浓度归一化统计量均来自「其他所有基因」的数据，与 leave-gene-out
    预训练保持一致。

    实际 finetune 数量 = min(n_finetune, int(N * max_finetune_pct))，
    保证小基因（MAPT/APP/PLN 等）至少有一半数据留做 validation。
    """
    mask_gene = np.array(all_data['target_gene'].astype(str)) == gene
    gene_data  = all_data[mask_gene].reset_index(drop=False)   # keep original index
    other_data = all_data[~mask_gene]

    n_gene = len(gene_data)

    # 自动调整 n_finetune：不超过基因总量的 max_finetune_pct
    n_finetune = min(n_finetune, int(n_gene * max_finetune_pct))
    n_finetune = max(n_finetune, 1)  # 至少 1 条

    if n_gene < n_finetune + 1:
        raise ValueError(
            f'Gene {gene} has only {n_gene} samples, cannot split into finetune + validation.'
        )

    print(f'[finetune] gene={gene}: total={n_gene}, '
          f'n_finetune={n_finetune} ({n_finetune/n_gene:.0%}), '
          f'val={n_gene - n_finetune}', flush=True)

    # ── 按 seed 随机划分 ──────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_gene)
    ft_local_idx  = perm[:n_finetune]
    val_local_idx = perm[n_finetune:]

    ft_data  = gene_data.iloc[ft_local_idx]
    val_data = gene_data.iloc[val_local_idx]

    print(f'[finetune] gene={gene}, seed={seed}: '
          f'finetune={len(ft_data)}, val={len(val_data)}', flush=True)

    # ── PSSM & 归一化统计量：来自其他基因 ────────────────────────────────────
    other_dict = _getdict(other_data)
    pssm_as = create_pssm(other_dict['ori_as'])
    pssm_ss = create_pssm(other_dict['ori_ss'])

    other_log_conc = np.log10(np.clip(other_dict['concentration_nM'].astype(float), 1e-9, None))
    conc_log_mean  = float(np.mean(other_log_conc))
    conc_log_std   = float(np.std(other_log_conc)) if float(np.std(other_log_conc)) > 1e-8 else 1.0

    # ── 构建 Dataset ──────────────────────────────────────────────────────────
    ft_dict  = _getdict(ft_data)
    val_dict = _getdict(val_data)

    ft_set  = Generate_dataset(ft_dict,  pssm_as, pssm_ss, max_len, conc_log_mean, conc_log_std)
    val_set = Generate_dataset(val_dict, pssm_as, pssm_ss, max_len, conc_log_mean, conc_log_std)

    # drop_last=True 保证 BatchNorm 不崩（batch size ≥ 2）
    ft_loader  = DataLoader(ft_set,  batch_size=batch_size, shuffle=True,
                            drop_last=(len(ft_set) > batch_size))
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False, drop_last=False)

    return ft_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# 模型工具
# ═══════════════════════════════════════════════════════════════════════════════

def load_pretrained(model_cls_name, pretrained_path, device,
                    cell_embed_dim=8, as_ts=3, ss_ts=3):
    """加载预训练 checkpoint，返回 model。"""
    model_cls = getattr(convmcmsipred_models, model_cls_name)
    model = model_cls(dp=0.1, device=device,
                      cell_embed_dim=cell_embed_dim,
                      as_ts=as_ts, ss_ts=ss_ts).float().to(device)

    state = torch.load(pretrained_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    # 兼容 DataParallel wrapper（key 前缀 'module.'）
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f'[warn] missing keys: {missing}', flush=True)
    if unexpected:
        print(f'[warn] unexpected keys: {unexpected}', flush=True)
    print(f'[finetune] loaded pretrained from {pretrained_path}', flush=True)
    return model


def apply_strategy(model, lr):
    """Freeze the backbone and train BN + the complete DNN prediction head."""
    for p in model.parameters():
        p.requires_grad = False
    for m in [model.bn, model.DNN]:
        for p in m.parameters():
            p.requires_grad = True
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    unexpected = [
        name for name in trainable_names
        if not (name.startswith('bn.') or name.startswith('DNN.'))
    ]
    if unexpected:
        raise RuntimeError(f'Unexpected trainable parameters: {unexpected}')
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f'[finetune] trainable params: {n_trainable:,} / {n_total:,}', flush=True)
    print(f'[finetune] trainable modules: bn + full DNN ({len(trainable_names)} tensors)',
          flush=True)
    return torch.optim.Adam(trainable, lr=lr, weight_decay=5e-4)


def set_train_mode(model):
    """backbone eval（冻结 dropout），head train。"""
    model.eval()
    model.bn.train()
    model.DNN.train()


# ═══════════════════════════════════════════════════════════════════════════════
# 评估 & 损失工具
# ═══════════════════════════════════════════════════════════════════════════════

def move_to_device(batch, device):
    """将 batch dict 移至设备，浮点 tensor 转 float32。"""
    if isinstance(batch, dict):
        inputs = batch
    else:
        inputs = batch[0]
    for k in list(inputs.keys()):
        v = inputs[k]
        if torch.is_tensor(v):
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.float()
            inputs[k] = v
    label = inputs['inhibit'].to(device).float()
    return inputs, label


def compute_metrics(labels, preds):
    """返回 PCC, SPCC, NCDG, AUC。"""
    labels = np.asarray(labels, dtype=float).reshape(-1)
    preds  = np.asarray(preds,  dtype=float).reshape(-1)
    mask   = np.isfinite(labels) & np.isfinite(preds)
    labels, preds = labels[mask], preds[mask]
    if len(labels) < 2:
        return -1, -1, -1, -1

    try:
        pcc  = round(float(pearsonr(labels, preds)[0]), 4)
    except Exception:
        pcc = -1
    try:
        spcc = round(float(spearmanr(preds, labels)[0]), 4)
    except Exception:
        spcc = -1
    try:
        ncdg = round(float(ndcg_score([labels], [preds], k=min(64, len(labels)))), 4)
    except Exception:
        ncdg = -1
    try:
        binary = (labels >= 0.7).astype(int)
        auc = round(float(roc_auc_score(binary, preds)), 4) if len(np.unique(binary)) > 1 else -1
    except Exception:
        auc = -1
    return pcc, spcc, ncdg, auc


def evaluate(model, loader, device, criterion_huber):
    model.eval()
    all_preds, all_labels, huber_vals = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, label = move_to_device(batch, device)
            pred = model(inputs)
            if not (torch.isfinite(pred).all() and torch.isfinite(label).all()):
                continue
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            huber_vals.append(float(criterion_huber(label, pred).cpu()))

    if not all_preds:
        return -1, -1, -1, -1, float('nan')

    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    pcc, spcc, ncdg, auc = compute_metrics(labels_np, preds_np)
    return pcc, spcc, ncdg, auc, float(np.mean(huber_vals))


# ═══════════════════════════════════════════════════════════════════════════════
# 单次 finetune（单个 seed）
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_one_seed(args, all_data, gene, seed, device):
    """
    对单个 seed 执行一次 finetune，返回 best_record dict。
    """
    init_seed(seed)

    # ── 数据 ──────────────────────────────────────────────────────────────────
    ft_loader, val_loader = build_loaders(
        all_data, gene, args.n_finetune, seed, args.max_len, args.batch_size,
        max_finetune_pct=args.max_finetune_pct
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = load_pretrained(args.model, args.pretrained_path, device,
                            cell_embed_dim=args.cell_embed_dim,
                            as_ts=args.as_ts, ss_ts=args.ss_ts)
    optimizer = apply_strategy(model, args.lr)

    # CosineAnnealingWarmRestarts：T_0=30 与预训练保持一致（虽然 epoch 数通常更少）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    criterion_huber = nn.HuberLoss(reduction='mean')
    criterion_mse   = nn.MSELoss(reduction='mean')

    # ── 保存路径 ──────────────────────────────────────────────────────────────
    save_dir = Path(args.output_dir) / args.model / gene / args.strategy / f'seed{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'model_best.pth.tar'

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_metric  = -float('inf')
    best_record  = None
    no_improve   = 0

    # Zero-shot baseline（直接用预训练模型评估）
    zs_pcc, zs_spcc, zs_ncdg, zs_auc, zs_huber = evaluate(model, val_loader, device, criterion_huber)
    print(f'[zero-shot] gene={gene}, seed={seed}: '
          f'pcc={zs_pcc}, spcc={zs_spcc}, ncdg={zs_ncdg}, auc={zs_auc}, huber={zs_huber:.5f}',
          flush=True)

    for epoch in range(1, args.n_epochs + 1):
        set_train_mode(model)
        train_losses = []

        for batch in ft_loader:
            inputs, label = move_to_device(batch, device)

            pred = model(inputs)
            if not torch.isfinite(pred).all():
                print(f'[warn] epoch {epoch}: non-finite pred, skipping batch', flush=True)
                continue

            loss_huber = criterion_huber(label, pred)

            # NDCG 辅助损失（与 train.py 保持一致）
            ndcg_alpha      = 100 / math.exp(0.01 * epoch)
            label_for_ndcg  = torch.clamp(label, 0.0, 1.0)
            loss_ndcg       = args.ndcg_weight * (1 - 1 / math.exp(0.1 * epoch)) * \
                              ncdg_loss(pred, label_for_ndcg, alpha=ndcg_alpha)

            loss = loss_huber + loss_ndcg
            if not torch.isfinite(loss):
                print(f'[warn] epoch {epoch}: non-finite loss, skipping batch', flush=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        scheduler.step()

        pcc, spcc, ncdg, auc, huber = evaluate(model, val_loader, device, criterion_huber)
        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f'epoch {epoch:4d} >> train_loss={train_loss:.6f}, '
                  f'pcc={pcc}, spcc={spcc}, ncdg={ncdg}, auc={auc}, huber={huber:.5f}',
                  flush=True)

        # ── Early stopping（monitor: SPCC，fallback PCC）──────────────────────
        monitor = spcc if spcc != -1 else pcc
        if monitor != -1 and monitor > best_metric:
            best_metric = monitor
            best_record = dict(
                epoch=epoch, pcc=pcc, spcc=spcc, ncdg=ncdg, auc=auc,
                huber=huber, train_loss=train_loss, seed=seed,
                zero_shot=dict(pcc=zs_pcc, spcc=zs_spcc, ncdg=zs_ncdg, auc=zs_auc)
            )
            no_improve = 0
            torch.save(model.state_dict(), str(ckpt_path))
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f'[early stop] epoch {epoch}', flush=True)
            break

    if best_record is None:
        best_record = dict(epoch=0, pcc=-1, spcc=-1, ncdg=-1, auc=-1,
                           huber=float('nan'), train_loss=float('nan'), seed=seed,
                           zero_shot=dict(pcc=zs_pcc, spcc=zs_spcc, ncdg=zs_ncdg, auc=zs_auc))

    print(f'\nBEST  gene={gene}, strategy={args.strategy}, seed={seed}: '
          f'epoch={best_record["epoch"]}, '
          f'pcc={best_record["pcc"]}, spcc={best_record["spcc"]}, '
          f'ncdg={best_record["ncdg"]}, auc={best_record["auc"]}, '
          f'huber={best_record["huber"]:.5f}',
          flush=True)
    print(f'  zero-shot: pcc={zs_pcc}, spcc={zs_spcc}, ncdg={zs_ncdg}, auc={zs_auc}',
          flush=True)

    # 保存单 seed 结果 JSON
    with open(save_dir / 'result.json', 'w') as f:
        json.dump(best_record, f, indent=2)

    return best_record


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总多 seed 结果
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(records, gene, strategy, n_finetune):
    """打印 mean ± std 汇总，返回 summary dict。"""
    metrics = ['pcc', 'spcc', 'ncdg', 'auc']
    zs_metrics = ['pcc', 'spcc', 'ncdg', 'auc']

    summary = {'gene': gene, 'strategy': strategy, 'n_finetune': n_finetune,
               'n_seeds': len(records)}

    print(f'\n{"="*60}', flush=True)
    print(f'SUMMARY  gene={gene}, strategy={strategy}, n_finetune={n_finetune}, '
          f'n_seeds={len(records)}', flush=True)
    print(f'{"="*60}', flush=True)

    for m in metrics:
        vals = [r[m] for r in records if r[m] != -1]
        if vals:
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            summary[f'finetune_{m}_mean'] = mean_v
            summary[f'finetune_{m}_std']  = std_v
            print(f'  finetune {m:6s}: {mean_v:.4f} ± {std_v:.4f}', flush=True)

    # Zero-shot 参考（所有 seed 应相同，取第一个）
    zs = records[0].get('zero_shot', {})
    print('  --- zero-shot baseline ---', flush=True)
    for m in zs_metrics:
        v = zs.get(m, -1)
        summary[f'zeroshot_{m}'] = v
        print(f'  zero-shot {m:6s}: {v:.4f}', flush=True)

    print(f'{"="*60}\n', flush=True)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_gene_runs(args, all_data):
    """Return [(gene, checkpoint)] and genes skipped for missing checkpoints."""
    available_genes = sorted(
        set(all_data['target_gene'].dropna().astype(str)) - {'', 'nan', 'None'}
    )
    root = Path(args.pretrained_root).expanduser()

    if args.gene.lower() != 'all':
        gene = args.gene
        if gene not in available_genes:
            raise ValueError(
                f'Gene {gene} not found. Available genes: {available_genes}'
            )
        checkpoint = (
            Path(args.pretrained_path).expanduser()
            if args.pretrained_path
            else root / gene / 'model_best.pth.tar'
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f'Leave-gene checkpoint not found: {checkpoint}')
        return [(gene, checkpoint)], []

    runs = []
    skipped = []
    for gene in available_genes:
        checkpoint = root / gene / 'model_best.pth.tar'
        if checkpoint.exists():
            runs.append((gene, checkpoint))
        else:
            skipped.append(gene)
    if not runs:
        raise FileNotFoundError(
            f'No leave-gene checkpoints found under {root}/<GENE>/model_best.pth.tar'
        )
    return runs, skipped


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'\n{"="*60}', flush=True)
    print('Few-shot leave-gene finetuning', flush=True)
    print(f'  gene={args.gene}, model={args.model}', flush=True)
    print(f'  strategy={args.strategy}, n_finetune={args.n_finetune}', flush=True)
    print(f'  lr={args.lr}, bs={args.batch_size}, epochs={args.n_epochs}, '
          f'patience={args.patience}', flush=True)
    print(f'  seeds={args.seeds}', flush=True)
    print(f'  pretrained_root={args.pretrained_root}', flush=True)
    print(f'  mrna_rnafm_path={args.mrna_rnafm_path}', flush=True)
    print(f'  device={device}', flush=True)
    print(f'{"="*60}\n', flush=True)

    # Load the full dataset and aligned mRNA cache once for every gene/seed.
    all_data = _load_data(args.data_path)
    configure_mrna_inputs(all_data, args.model, args.mrna_rnafm_path)
    gene_counts = all_data['target_gene'].value_counts()
    gene_runs, skipped_genes = resolve_gene_runs(args, all_data)
    print(f'[info] genes to finetune ({len(gene_runs)}): '
          f'{[gene for gene, _ in gene_runs]}', flush=True)
    if skipped_genes:
        print(f'[warn] skipped genes without leave-gene checkpoint: {skipped_genes}',
              flush=True)

    combined = {
        'model': args.model,
        'strategy': args.strategy,
        'requested_n_finetune': args.n_finetune,
        'seeds': args.seeds,
        'skipped_missing_checkpoint': skipped_genes,
        'genes': {},
    }

    for run_index, (gene, checkpoint) in enumerate(gene_runs, start=1):
        gene_args = copy.copy(args)
        gene_args.gene = gene
        gene_args.pretrained_path = str(checkpoint)
        print(f'\n{"#"*70}', flush=True)
        print(f'GENE {run_index}/{len(gene_runs)}: {gene} '
              f'(n={int(gene_counts[gene])})', flush=True)
        print(f'checkpoint={checkpoint}', flush=True)
        print(f'{"#"*70}', flush=True)

        records = []
        for seed in gene_args.seeds:
            print(f'\n{"-"*50}', flush=True)
            print(f'GENE {gene} | SEED {seed}', flush=True)
            print(f'{"-"*50}', flush=True)
            records.append(
                finetune_one_seed(gene_args, all_data, gene, seed, device)
            )

        summary = summarize(
            records, gene, gene_args.strategy, gene_args.n_finetune
        )
        combined['genes'][gene] = summary
        out_dir = (
            Path(gene_args.output_dir)
            / gene_args.model
            / gene
            / gene_args.strategy
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f'summary_nft{gene_args.n_finetune}.json'
        with open(summary_path, 'w') as handle:
            json.dump(summary, handle, indent=2)
        print(f'Summary saved to {summary_path}', flush=True)

    combined_dir = Path(args.output_dir) / args.model
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / f'summary_all_genes_nft{args.n_finetune}.json'
    with open(combined_path, 'w') as handle:
        json.dump(combined, handle, indent=2)
    print(f'\nAll-gene summary saved to {combined_path}', flush=True)


if __name__ == '__main__':
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# 批量运行示例（Shell）
# ═══════════════════════════════════════════════════════════════════════════════
#
# SEEDS="42,123,456"
#
# python finetune_ori.py \
#   --gene all \
#   --model Base0_mrna_full_no_se \
#   --pretrained_root ../liubin_pipeline/potency_prediction/pretrain_models/Base0_mrna_full_no_se \
#   --mrna_rnafm_path data/mrna_rnafm_22622_maxlen65.npy \
#   --n_finetune 300 \
#   --seeds $SEEDS \
#   --n_epochs 100 \
#   --patience 15
