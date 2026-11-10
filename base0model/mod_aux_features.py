"""
mod_aux_features.py
化学修饰辅助特征模块 — 最终版

整合三轮讨论共识：
  1. PS 是 linkage，不是 residue → left_PS / right_PS
  2. 功能区位置编码 → seed / central / supplementary / 3' region
  3. 构象先验（不依赖实验条件）→ pucker_preference / lock_degree
  4. 化学类别（对低频修饰仍有帮助）→ 少量 boolean
  5. 可选：2'-F 弱 ΔG 先验（Pallan 2011 Table 2）
  6. 可选：2'-OMe/RNA NN ΔG 近似（Kierzek 2006 Table 3, 仅全局）
  7. 保留原始 embedding + energy_index，新特征并联不替换
"""

import numpy as np
import math

# ═══════════════════════════════════════════════════════════
# A. 构象先验 — 从化学结构直接推断，不依赖实验条件
# ═══════════════════════════════════════════════════════════
# pucker_preference: C3'-endo倾向 → +1, C2'-endo → -1, 无偏好/柔性 → 0
# lock_degree: 完全锁定 → +1, 部分约束 → 0.5, 无约束 → 0, 解锁/柔性 → -1

CONFORMATIONAL_PRIOR = {
    # 未修饰碱基
    'A': (0.5, 0.0),  'G': (0.5, 0.0),  'C': (0.5, 0.0),  'U': (0.5, 0.0),
    't': (-0.5, 0.0),  # DNA中的脱氧T偏好C2'-endo
    # 常见修饰
    'F': (1.0, 0.3),   # 2'-F: 强C3'-endo偏好, 不锁定 (Pallan 2011: 构象与RNA几乎相同)
    'Y': (0.8, 0.2),   # 2'-OMe: C3'-endo偏好, 不锁定
    'L': (1.0, 1.0),   # LNA: 完全锁定在C3'-endo (2'-O,4'-C-methylene桥)
    'Z': (-0.8, 0.0),  # 2'-Deoxy: 偏好C2'-endo (DNA-like)
    'X': (0.0, -1.0),  # UNA: 开环, 完全柔性/无约束
    'N': (1.0, 1.0),   # α-L-LNA: 锁定, 但构型翻转 (α-L-ribo)
    'S': (0.8, 0.5),   # HNA: 六元环, 偏好A-form, 部分约束
    'I': (0.8, 0.5),   # ANA: 六元环, 偏好A-form, 比HNA多一个OH
    'M': (1.0, 1.0),   # ENA: ethylene桥锁定
    'V': (1.0, 1.0),   # Oxetane-locked: 四元环锁定 (但实际去稳定化)
    'H': (-0.3, 0.0),  # 4'-C-Hydroxymethyl-DNA: DNA-like
    # LNA 衍生物 — 核心都是锁定构象
    'B': (1.0, 1.0),   # Adamantyl-amino-LNA
    'W': (1.0, 1.0),   # Pyrenyl-amino-LNA
    'P': (1.0, 1.0),   # Carbocyclic-EBL
    'Q': (1.0, 1.0),   # Carbocyclic-LNA
    # 2'-氨基烷基系列 — 方向不确定
    'D': (0.3, 0.0),   # 2'-Aminoethoxymethyl
    'E': (0.3, 0.0),   # 2'-Aminoethyl
    'K': (0.3, 0.0),   # 2'-Aminopropoxymethyl
    'J': (0.3, 0.0),   # 2'-Aminopropyl
    'R': (0.3, 0.0),   # 2'-Guanidinoethyl
    'O': (0.3, 0.0),   # 2'-Cyanoethyl
    # 其他
    'n': (0.0, 0.0),   # Peptide: 样本太少
    'q': (0.0, 0.0),   # PS: 骨架修饰, 不影响糖环构象, 单独编码
}


# ═══════════════════════════════════════════════════════════
# B. 化学类别 — 对低频修饰有帮助（MACCS对<50条样本学不稳）
# ═══════════════════════════════════════════════════════════
# 只保留少量和构象先验不重复的维度

def get_chemical_class(symbol):
    """返回 [is_2prime_f, is_2prime_ome, is_locked_family, is_unlocked, is_six_membered_ring, is_charged]"""
    return {
        'A': [0,0,0,0,0,0], 'G': [0,0,0,0,0,0], 'C': [0,0,0,0,0,0], 'U': [0,0,0,0,0,0], 't': [0,0,0,0,0,0],
        'F': [1,0,0,0,0,0],
        'Y': [0,1,0,0,0,0],
        'L': [0,0,1,0,0,0], 'N': [0,0,1,0,0,0], 'M': [0,0,1,0,0,0],
        'B': [0,0,1,0,0,0], 'W': [0,0,1,0,0,0], 'P': [0,0,1,0,0,0], 'Q': [0,0,1,0,0,0], 'V': [0,0,1,0,0,0],
        'X': [0,0,0,1,0,0],
        'S': [0,0,0,0,1,0], 'I': [0,0,0,0,1,0],
        'Z': [0,0,0,0,0,0], 'H': [0,0,0,0,0,0], 'O': [0,0,0,0,0,0],
        'D': [0,0,0,0,0,0], 'E': [0,0,0,0,0,1], 'K': [0,0,0,0,0,0],
        'J': [0,0,0,0,0,1], 'R': [0,0,0,0,0,1],  # 氨基/胍基带正电
        'n': [0,0,0,0,0,0], 'q': [0,0,0,0,0,0],
    }.get(symbol, [0,0,0,0,0,0])


# ═══════════════════════════════════════════════════════════
# C. PS linkage 编码 — q 不是 residue，是相邻核苷酸之间的连接
# ═══════════════════════════════════════════════════════════

def parse_ps_linkage(cm_seq, max_len=23):
    """
    解析 cm_seq 中的 PS (q) 位置, 返回每个核苷酸的 left_PS 和 right_PS.
    
    原始编码中 q 不占核苷酸位置 (res_ind 不前进), 它标记的是当前位置
    的磷酸骨架连接为 PS. 具体来说:
    
      qA...  → A 的左侧 (5'端) 是 PS
      ...Aq... → A 的右侧 (3'端) 是 PS, 下一个核苷酸左侧也是 PS
    
    例: qACCCU → position 0 (A) left_PS=1
    例: AqUGqC → pos 0 (A) right_PS=1, pos 1 (U) left_PS=1,
                  pos 2 (G) right_PS=1, pos 3 (C) left_PS=1
    """
    left_ps = np.zeros(max_len)
    right_ps = np.zeros(max_len)
    
    # 遍历序列, 跟踪当前核苷酸位置和是否刚遇到 q
    res_ind = 0
    ps_pending = False  # 上一个字符是 q, 下一个核苷酸左侧有 PS
    
    for ch in cm_seq:
        if ch == 'q':
            # q 标记: 前一个核苷酸右侧有 PS, 下一个核苷酸左侧有 PS
            if res_ind > 0:
                right_ps[res_ind - 1] = 1.0
            ps_pending = True
        else:
            if res_ind < max_len:
                if ps_pending:
                    left_ps[res_ind] = 1.0
                    ps_pending = False
                res_ind += 1
    
    return left_ps, right_ps


# ═══════════════════════════════════════════════════════════
# D. 功能区位置编码 — siRNA 机制先验
# ═══════════════════════════════════════════════════════════

def get_position_features(seq_len=21, max_len=23):
    """
    返回 (max_len, 5) 的位置特征矩阵:
      [pos_norm, is_pos_2_8, is_pos_9_11, is_pos_12_16, is_pos_17_plus]
    
    用中性命名, AS和SS都适用.
    对 antisense (guide) strand 来说:
      2-8   = seed region
      9-11  = central / Ago2 cleavage region
      12-16 = supplementary region
      17+   = 3' region
    """
    result = np.zeros((max_len, 5))
    for i in range(min(seq_len, max_len)):
        pos_1based = i + 1
        result[i, 0] = i / (seq_len - 1) if seq_len > 1 else 0  # pos_norm
        result[i, 1] = 1.0 if 2 <= pos_1based <= 8 else 0.0     # is_pos_2_8
        result[i, 2] = 1.0 if 9 <= pos_1based <= 11 else 0.0    # is_pos_9_11
        result[i, 3] = 1.0 if 12 <= pos_1based <= 16 else 0.0   # is_pos_12_16
        result[i, 4] = 1.0 if 17 <= pos_1based else 0.0          # is_pos_17_plus
    return result


# ═══════════════════════════════════════════════════════════
# E. 可选: 2'-F 弱 ΔG 先验 (Pallan 2011 NAR Table 2)
# ═══════════════════════════════════════════════════════════
# 来源: Pallan et al. 2011 Table 2
#   八聚体 CGAAUUCG: ΔΔG°37 = -4.1 kcal/mol / 8 mod = -0.51/mod
#   十二聚体 UAUAUAUAUAUA: ΔΔG°37 = -5.3 kcal/mol / 12 mod = -0.44/mod
#   体系: 2'-F-RNA self-complementary duplex, 300mM NaCl, pH 7.4
#   限制: 仅测了两个序列, 且是自互补duplex, 不是siRNA:mRNA

F_DELTA_G37_PER_MOD = -0.47   # 两个序列的平均值
F_DELTA_G37_CONFIDENCE = 0.75  # 中-高置信 (同文献两序列, 但非siRNA体系)


# ═══════════════════════════════════════════════════════════
# F. 可选: 2'-OMe/RNA NN 参数 (Kierzek 2006 NAR Table 3)
# ═══════════════════════════════════════════════════════════
# 来源: Kierzek et al. 2006, Table 3
# 体系: 2'-O-methyl RNA strand / RNA complement, 100mM NaCl, pH 7.0
# 限制: 适用于全2'-OMe链配RNA, 不适用于混合修饰链

# ΔG°37 (kcal/mol), m=2'-OMe, r=RNA
NN_PARAMS_OME_RNA = {
    # (5'XY on OMe strand, read 5'→3') : ΔG°37
    ('AA', 'UU'): -0.55, ('AU', 'UA'): -0.84, ('UU', 'AA'): -0.94, ('UA', 'AU'): -1.20,
    ('AC', 'UG'): -1.60, ('AG', 'UC'): -1.81, ('CA', 'GU'): -1.89, ('UC', 'AG'): -1.90,
    ('UG', 'AC'): -1.94, ('GA', 'CU'): -2.06, ('CU', 'GA'): -2.17, ('GU', 'CA'): -2.17,
    ('CG', 'GC'): -2.35, ('CC', 'GG'): -2.78, ('GG', 'CC'): -2.82, ('GC', 'CG'): -3.02,
    'initiation': 3.32,
    'terminal_AU': 0.30,
}

# RNA/RNA NN 参数 baseline (Xia 1998, 1M NaCl)
NN_PARAMS_RNA_RNA = {
    ('AA', 'UU'): -0.93, ('AU', 'UA'): -1.10, ('UU', 'AA'): -0.93, ('UA', 'AU'): -1.33,
    ('AC', 'UG'): -2.24, ('AG', 'UC'): -2.08, ('CA', 'GU'): -2.11, ('UC', 'AG'): -2.35,
    ('UG', 'AC'): -2.11, ('GA', 'CU'): -2.35, ('CU', 'GA'): -2.08, ('GU', 'CA'): -2.24,
    ('CG', 'GC'): -2.36, ('CC', 'GG'): -3.26, ('GG', 'CC'): -3.26, ('GC', 'CG'): -3.42,
    'initiation': 4.09,
    'terminal_AU': 0.45,
}
# 注意: 两套参数盐浓度不同 (100mM vs 1M NaCl), ΔΔG 比绝对值更可靠


def compute_nn_deltaG(strand1_5to3, strand2_5to3, nn_params):
    """
    用 nearest-neighbor 模型计算反平行双链 ΔG°37.
    
    strand1_5to3: 5'→3' 方向的链1 (只含 AGCU/t)
    strand2_5to3: 5'→3' 方向的链2 (反平行互补链)
    nn_params: NN_PARAMS_OME_RNA 或 NN_PARAMS_RNA_RNA
    
    NN 参数格式: (5'XY on strand1, 3'WZ on strand2)
    反平行配对: strand1[i] 配 strand2[n-1-i]
    
    返回 ΔG°37 (kcal/mol), 或 None 如果有未知 dinucleotide
    """
    base_map = {'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U', 't': 'U'}
    
    s1 = ''.join(base_map.get(c, '?') for c in strand1_5to3)
    s2 = ''.join(base_map.get(c, '?') for c in strand2_5to3)
    
    if '?' in s1 or '?' in s2:
        return None
    
    # 取配对区域 (两条链可能不等长, 取较短的)
    n = min(len(s1), len(s2))
    s1 = s1[:n]
    s2 = s2[:n]
    
    # strand2 反向 → 得到 3'→5' 方向 (与 strand1 平行)
    s2_rev = s2[::-1]
    
    dG = nn_params['initiation']
    
    for i in range(n - 1):
        stack_s1 = s1[i:i+2]          # strand1 上的 5'→3' dinucleotide
        stack_s2 = s2_rev[i:i+2]      # strand2 上的 3'→5' dinucleotide
        key = (stack_s1, stack_s2)
        if key not in nn_params:
            return None
        dG += nn_params[key]
    
    # terminal AU penalty: 检查双链两端的碱基对
    if s1[0] in 'AU' or s2_rev[0] in 'AU':
        dG += nn_params['terminal_AU']
    if s1[-1] in 'AU' or s2_rev[-1] in 'AU':
        dG += nn_params['terminal_AU']
    
    return dG


def compute_global_nn_features(ori_as, ori_ss):
    """
    计算全局 NN 热力学特征 (做法A: global duplex-level).
    
    返回 dict:
      rna_rna_dG:        RNA/RNA duplex 的 ΔG°37 (Xia 1998 参数)
      ome_rna_dG:        2'-OMe/RNA duplex 的 ΔG°37 近似 (Kierzek 2006 参数)
      ome_delta_dG:      ΔΔG = ome_rna_dG - rna_rna_dG
    
    注意:
      - ome_rna_dG 是"如果整条链全是2'-OMe"的近似, 实际混合修饰链不完全适用
      - 两套参数盐浓度不同, ΔΔG 比绝对值更有参考价值
      - 如果计算失败返回 0 (序列含非标准碱基等)
    """
    # 取两条链的可比较区域 (通常 19nt 核心配对区)
    min_len = min(len(ori_as), len(ori_ss))
    core_len = min(min_len, 19)
    
    as_seq = ori_as[:core_len]
    ss_seq = ori_ss[:core_len]
    
    rna_dG = compute_nn_deltaG(as_seq, ss_seq, NN_PARAMS_RNA_RNA)
    ome_dG = compute_nn_deltaG(as_seq, ss_seq, NN_PARAMS_OME_RNA)
    
    if rna_dG is None or ome_dG is None:
        return {'rna_rna_dG': 0.0, 'ome_rna_dG': 0.0, 'ome_delta_dG': 0.0}
    
    return {
        'rna_rna_dG': rna_dG,
        'ome_rna_dG': ome_dG,
        'ome_delta_dG': ome_dG - rna_dG,
    }


# ═══════════════════════════════════════════════════════════
# G. 主函数: 生成 per-position 辅助特征
# ═══════════════════════════════════════════════════════════

MOD_SYMBOLS = set('BDEFHIJKLMNOPQRSVWXYZnq')
AUX_DIM = 15  # 总维度

def compute_per_position_aux(cm_seq, max_len=23, seq_len=21):
    """
    对一条修饰序列生成 per-position 辅助特征矩阵.
    
    输入:
      cm_seq: 修饰后序列字符串 (含修饰符号和q)
      max_len: 填充到的最大长度 (默认23)
      seq_len: 原始序列长度 (用于功能区计算, 默认21)
    
    输出:
      np.array, shape (max_len, AUX_DIM=15)
      
    维度说明:
      [0]  pucker_preference     构象: C3'-endo偏好
      [1]  lock_degree           构象: 锁定程度
      [2]  is_2prime_f           化学类别
      [3]  is_2prime_ome         化学类别
      [4]  is_locked_family      化学类别
      [5]  is_unlocked           化学类别
      [6]  is_six_membered_ring  化学类别
      [7]  is_charged            化学类别
      [8]  left_PS               PS linkage (左侧)
      [9]  right_PS              PS linkage (右侧)
      [10] pos_norm              位置: 归一化位置
      [11] is_pos_2_8            位置: 2-8区 (AS=seed)
      [12] is_pos_9_11           位置: 9-11区 (AS=central)
      [13] is_pos_12_16          位置: 12-16区 (AS=supplementary)
      [14] is_pos_17_plus        位置: 17+区 (AS=3'端)
    """
    result = np.zeros((max_len, AUX_DIM))
    
    # --- 解析序列, 提取每个位置的修饰符号 ---
    nucleotide_symbols = []
    seq_ind = 0
    while seq_ind < len(cm_seq):
        ch = cm_seq[seq_ind]
        if ch == 'q':
            seq_ind += 1  # q 不占核苷酸位置
        else:
            nucleotide_symbols.append(ch)
            seq_ind += 1
    
    # --- 构象先验 + 化学类别 (dim 0-7) ---
    for i in range(min(len(nucleotide_symbols), max_len)):
        sym = nucleotide_symbols[i]
        pucker, lock = CONFORMATIONAL_PRIOR.get(sym, (0.0, 0.0))
        result[i, 0] = pucker
        result[i, 1] = lock
        chem = get_chemical_class(sym)
        result[i, 2:8] = chem
    
    # --- PS linkage (dim 8-9) ---
    left_ps, right_ps = parse_ps_linkage(cm_seq, max_len)
    result[:, 8] = left_ps
    result[:, 9] = right_ps
    
    # --- 位置功能区 (dim 10-14) ---
    pos_features = get_position_features(seq_len, max_len)
    result[:, 10:15] = pos_features
    
    return result


# ═══════════════════════════════════════════════════════════
# H. 可选: 2'-F 弱先验 per-position
# ═══════════════════════════════════════════════════════════

def compute_f_prior_per_position(cm_seq, max_len=23):
    """
    对含 2'-F 的位置返回弱 ΔG 先验.
    
    返回 (max_len, 2): [deltaG_prior, deltaG_confidence]
    - 2'-F 位置: [-0.47, 0.75]
    - 其他位置: [0.0, 0.0]
    
    来源: Pallan 2011 Table 2, 平均 -0.47 kcal/mol/mod
    体系: 2'-F-RNA self-complementary duplex (非siRNA)
    """
    result = np.zeros((max_len, 2))
    res_ind = 0
    for ch in cm_seq:
        if ch == 'q':
            continue
        if res_ind >= max_len:
            break
        if ch == 'F':
            result[res_ind, 0] = F_DELTA_G37_PER_MOD
            result[res_ind, 1] = F_DELTA_G37_CONFIDENCE
        res_ind += 1
    return result


# ═══════════════════════════════════════════════════════════
# I. 汇总: 在 load_data.py 中调用的接口
# ═══════════════════════════════════════════════════════════

def compute_all_aux_features(cm_seq, ori_as=None, ori_ss=None, max_len=23, seq_len=21):
    """
    一次性计算所有辅助特征. 在 load_data.py 的 load_sirna() 中调用.
    
    返回 dict:
      'per_position_aux':     (23, 15)  必加特征
      'f_prior':              (23, 2)   可选: 2'-F ΔG 先验
      'global_nn':            (3,)      可选: 全局 NN ΔG 特征
    """
    per_pos = compute_per_position_aux(cm_seq, max_len, seq_len)
    f_prior = compute_f_prior_per_position(cm_seq, max_len)
    
    global_nn = np.zeros(3)
    if ori_as is not None and ori_ss is not None:
        nn_feats = compute_global_nn_features(ori_as, ori_ss)
        global_nn = np.array([
            nn_feats['rna_rna_dG'],
            nn_feats['ome_rna_dG'],
            nn_feats['ome_delta_dG'],
        ])
    
    return {
        'per_position_aux': per_pos,
        'f_prior': f_prior,
        'global_nn': global_nn,
    }
