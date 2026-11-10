"""
compute_mrna_thermo.py
为带 local_mrna 上下文的数据计算两个 gene-agnostic 的 mRNA 侧热力学特征:

  1. target accessibility  - 靶位点区域的平均单链(可及)概率, 用 RNAplfold
  2. local folding dG       - 60nt 局部窗口自身折叠的最小自由能, 用 RNAfold

这两个量都只关于 mRNA 那一侧, 不涉及 siRNA 的化学修饰, 所以没有
"修饰体系参数缺失" 的问题, 是干净的物理量。

依赖: ViennaRNA (pip install ViennaRNA)
输入: CMsi_with_local_mrna_final_matched_only.csv (含 local_mrna 及位置列)
输出: 在原表基础上追加以下列
  - acc_target_mean   靶位点全长平均可及性 (0~1, 越大越开放)
  - acc_seed_mean     seed 对应区域(guide 2-8 位)的平均可及性
  - acc_target_min    靶位点内最不可及位置的可及性 (瓶颈)
  - local_fold_dg     局部窗口折叠 MFE (kcal/mol, 越负越稳定/越难打开)
  - local_fold_dg_norm  MFE / 窗口长度 (归一化, 消除长度差异)
"""

import argparse
import numpy as np
import pandas as pd
import RNA


def compute_accessibility(local_mrna, target_rel_start, target_len, seed_len=7):
    """
    用 RNAplfold 计算靶位点区域的单链概率(可及性).

    RNAplfold 给出每个位置作为长度为 u 的单链片段一部分的概率。
    我们用 u=1 (单核苷酸单链概率) 作为可及性度量, 对靶位点区域取平均。

    local_mrna: 局部 mRNA 序列 (5'->3')
    target_rel_start: 靶位点在 local_mrna 中的起始 index (0-based)
    target_len: 靶位点长度
    返回: (acc_target_mean, acc_seed_mean, acc_target_min)
    """
    seq = str(local_mrna).upper().replace('T', 'U')
    L = len(seq)
    if L < 5 or target_rel_start < 0 or target_rel_start >= L:
        return np.nan, np.nan, np.nan

    # RNAplfold 参数: W=窗口大小, L=最大配对跨度, u=单链片段长度
    W = min(80, L)          # 局部配对窗口
    span = min(40, L)       # 最大碱基对跨度
    u = 1                   # 单核苷酸可及性

    md = RNA.md()
    fc = RNA.fold_compound(seq, md, RNA.OPTION_WINDOW)

    # 收集每个位置的单链概率
    unpaired = np.full(L, np.nan)

    def callback(v, v_size, i, maxsize, what, data):
        # what & RNA.PROBS_WINDOW_UP 表示这是 up (unpaired) 概率
        if (what & RNA.PROBS_WINDOW_UP) and v is not None:
            # v[u] = 位置 i 处长度为 u 的片段为单链的概率
            if len(v) > u and v[u] is not None:
                data[i - 1] = v[u]  # i 是 1-based

    fc.probs_window(u, RNA.PROBS_WINDOW_UP, callback, unpaired)

    # 靶位点区域
    t0 = target_rel_start
    t1 = min(target_rel_start + target_len, L)
    target_acc = unpaired[t0:t1]
    target_acc = target_acc[~np.isnan(target_acc)]

    if len(target_acc) == 0:
        return np.nan, np.nan, np.nan

    acc_target_mean = float(np.mean(target_acc))
    acc_target_min = float(np.min(target_acc))

    # seed 区: guide 的 2-8 位对应 mRNA 靶位点的 3' 端
    # (guide 5'->3' 与 mRNA 反平行, guide seed 对应 target 的 3' 侧)
    seed_acc = unpaired[max(t1 - seed_len, t0):t1]
    seed_acc = seed_acc[~np.isnan(seed_acc)]
    acc_seed_mean = float(np.mean(seed_acc)) if len(seed_acc) > 0 else acc_target_mean

    return acc_target_mean, acc_seed_mean, acc_target_min


def compute_local_fold_dg(local_mrna):
    """用 RNAfold 计算局部窗口折叠的 MFE (kcal/mol)"""
    seq = str(local_mrna).upper().replace('T', 'U')
    if len(seq) < 5:
        return np.nan, np.nan
    structure, mfe = RNA.fold(seq)
    return float(mfe), float(mfe) / len(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/mnt/user-data/uploads/CMsi_with_local_mrna_final_matched_only.csv')
    parser.add_argument('--output', type=str,
                        default='/mnt/user-data/outputs/CMsi_with_mrna_thermo.csv')
    parser.add_argument('--guide_col', type=str, default='ori_as')
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    print(f'读取 {len(df)} 行')

    acc_target_mean = []
    acc_seed_mean = []
    acc_target_min = []
    local_fold_dg = []
    local_fold_dg_norm = []

    n_ok = 0
    n_skip = 0
    for idx, row in df.iterrows():
        lm = row.get('local_mrna', None)
        if pd.isna(lm) or str(lm).strip() == '':
            acc_target_mean.append(np.nan); acc_seed_mean.append(np.nan)
            acc_target_min.append(np.nan); local_fold_dg.append(np.nan)
            local_fold_dg_norm.append(np.nan)
            n_skip += 1
            continue

        lm = str(lm)
        # 靶位点在 local_mrna 中的相对起点
        try:
            t_start = int(row['target_start_0based'] - row['local_mrna_start_0based'])
        except Exception:
            # 退而求其次: 用 target_core 在 local_mrna 中定位
            tc = str(row.get('target_core', '')).upper().replace('T', 'U')
            t_start = lm.upper().replace('T', 'U').find(tc) if tc and tc != 'NAN' else -1

        tc = str(row.get('target_core', ''))
        t_len = len(tc) if tc and tc != 'nan' else len(str(row[args.guide_col]))

        # accessibility
        a_mean, a_seed, a_min = compute_accessibility(lm, t_start, t_len)
        acc_target_mean.append(a_mean)
        acc_seed_mean.append(a_seed)
        acc_target_min.append(a_min)

        # local folding dG
        dg, dg_norm = compute_local_fold_dg(lm)
        local_fold_dg.append(dg)
        local_fold_dg_norm.append(dg_norm)

        n_ok += 1
        if (idx + 1) % 1000 == 0:
            print(f'  已处理 {idx+1}/{len(df)}')

    df['acc_target_mean'] = acc_target_mean
    df['acc_seed_mean'] = acc_seed_mean
    df['acc_target_min'] = acc_target_min
    df['local_fold_dg'] = local_fold_dg
    df['local_fold_dg_norm'] = local_fold_dg_norm

    print(f'\n成功计算: {n_ok}, 跳过(无mRNA): {n_skip}')
    print('\n=== 特征统计 ===')
    for col in ['acc_target_mean', 'acc_seed_mean', 'acc_target_min',
                'local_fold_dg', 'local_fold_dg_norm']:
        s = df[col].dropna()
        print(f'  {col}: mean={s.mean():.3f}, std={s.std():.3f}, '
              f'min={s.min():.3f}, max={s.max():.3f}')

    df.to_csv(args.output, index=False)
    print(f'\n已保存到: {args.output}')


if __name__ == '__main__':
    main()
