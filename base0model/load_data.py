import os
import pdb
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse
from itertools import product
import math
import sklearn
from pathlib import Path

try:
    from .mod_aux_features import compute_all_aux_features
except ImportError:  # Preserve direct `python train.py` execution.
    from mod_aux_features import compute_all_aux_features

BASE0_DIR = Path(__file__).resolve().parent

# ===== mRNA RNA-FM per-residue embedding (按行号 rnafm_ind 取用) =====
# 路径可用环境变量 MRNA_RNAFM_PATH 覆盖
MRNA_RNAFM_PATH = os.environ.get(
    'MRNA_RNAFM_PATH', str(BASE0_DIR / 'data' / 'mrna_rnafm_22622_maxlen65.npy')
)
MRNA_THERMO_COLS = ['acc_target_mean', 'acc_seed_mean', 'acc_target_min',
                    'local_fold_dg', 'local_fold_dg_norm']
try:
    MRNA_RNAFM = np.load(MRNA_RNAFM_PATH, mmap_mode='r')  # [N, 65, 640], mmap省内存
    MRNA_MAX_LEN = MRNA_RNAFM.shape[1]
    MRNA_RNAFM_LOAD_ERROR = None
    print(f'[info] loaded mRNA RNA-FM: {MRNA_RNAFM.shape} from {MRNA_RNAFM_PATH}')
except Exception as e:
    MRNA_RNAFM = None
    MRNA_MAX_LEN = 65
    MRNA_RNAFM_LOAD_ERROR = e
    print(
        f'[warn] mRNA RNA-FM not loaded ({e}). Models that consume RNA-FM '
        'will stop with an actionable error instead of training on zero embeddings.'
    )


def model_requires_mrna_rnafm(model_name):
    """Return whether a Base0 model consumes the per-residue mRNA embedding."""
    name = str(model_name or '')
    return name.startswith('Base0_mrna') and name != 'Base0_mrna_thermo_only'


def model_requires_mrna_thermo(model_name):
    """Return whether a Base0 model consumes the five mRNA thermodynamic scalars."""
    name = str(model_name or '')
    return name.startswith('Base0_mrna') and name != 'Base0_mrna_rnafm_only'


def validate_mrna_rnafm_for_model(model_name, expected_rows):
    """Fail before splitting if an RNA-FM-dependent training run lacks its cache."""
    if not model_requires_mrna_rnafm(model_name):
        return
    if MRNA_RNAFM is None:
        detail = f' Original load error: {MRNA_RNAFM_LOAD_ERROR}' if MRNA_RNAFM_LOAD_ERROR else ''
        raise FileNotFoundError(
            f'{model_name} requires the aligned mRNA RNA-FM cache at '
            f'{MRNA_RNAFM_PATH}.{detail}\n'
            'Generate it from the released RNA-FM weight with '
            'base0model/mrna_features.py (see the KDMSi-Mod section in README.md).'
        )
    expected_shape = (int(expected_rows), 65, 640)
    if tuple(MRNA_RNAFM.shape) != expected_shape:
        raise ValueError(
            f'{model_name} requires an RNA-FM cache aligned row-for-row to the '
            f'training CSV: got {tuple(MRNA_RNAFM.shape)}, expected {expected_shape} '
            f'at {MRNA_RNAFM_PATH}'
        )

#### 修饰的smiles表征、修饰的名称到字符之间的相互转换词典 #######------------------------
symbol_2_smiles = {'B': 'OCC12CN(C(CO1)C2O)C(=O)CC12CC3CC(CC(C3)C1)C2','D': 'NCCOCOC1COC(CO)C1O',
                'E': 'NCCOC1COC(CO)C1O', 'F': 'OCC1OCC(F)C1O', 'H': 'OCC1(CO)OCCC1',
                'I': 'C([C@@H]([C@H]([C@H]([C@H](CO)O)O)O)O)O',
                # https://www.chemspider.com/Chemical-Structure.1266058.html  从这里获取的smiles，sirnamod网站上没有
                'J': 'NCCCOC1C(O)C(CO)OC1N1C=NC2=C1C=C(F)C=C2F',
                'K': 'NCCCOCOC1COC(CO)C1O',
                'L': 'OCC1OC2OC2C1O',
                # 用的是Oxetane Locked nucleic acid的smiles
                'M': 'OCC12CCNC(CO1)C2O', 'N': 'OCC1OCCC(O)C1O',
                'O': 'OCC1OCC(OCC#N)C1O','P': 'OCC12CCCC(CO1)C2O',
                'Q': 'OCC12CCC(CO1)C2O','R': 'NC(N)=NCCC1COC(CO)C1O',
                'S': 'OCC1OCCCC1O','V': 'OCC1OC2OC2C1O','W': 'OCC12CN(CC3=CC=C4C=CC5=CC=CC6=CC=C3C4=C56)C(CO1)C2O',
                'X': 'OCCOC(CO)CO','Y': 'COC1CSC(CO)C1O','Z': 'OCC1OCCC1O',
                # 'd': '2-O-Benzyl','e': '2-O-Guanidinopropyl','f': 'Boranophosphate',
                # 'h': 'Dihydrouridine', 'm': 'Hypoxanthine',
                'n': 'CNC(=O)CN(C)C(C)=O', 'q': 'OCC1OCC(O)C1OP(O)(S)=O',#'r': 'OC1CCC2(O)CCOC12',
                }
name_2_symbol = {'2-N-Adamant-1-yl-Methylcarbonyl-2-Amino-Locked': 'B', '2-Aminoethoxymethyl': 'D',
                '2-Aminoethyl': 'E','2-Fluoro': 'F','4-C-HydroxymethylDeoxyribo': 'H','Altritol': 'I',
                '2-Aminopropyl': 'J','2-Aminopropoxymethyl': 'K','Locked': 'L',
                '2-Deoxy-2-N,4-C-Ethylene-Locked': 'M','Alfa-L-Locked': 'N','2-Cyanoethyl': 'O',
                '2,4-Carbocyclic-Ethylene-bridged-Locked': 'P','2,4-Carbocyclic-Locked-Locked': 'Q',
                '2-Guanidinoethyl': 'R','Hexitol': 'S','Oxetane-Locked': 'V','2-N-Pyren-1-ylMethyl-2-Amino-Locked': 'W',
                'UnLocked': 'X','2-O-Methyl': 'Y','2-Deoxy': 'Z',
                # '2-O-Benzyl': 'd', '2-O-Guanidinopropyl': 'e', # 'Dihydrouridine': 'h'.  'Hypoxanthine': 'm',
                'Peptides': 'n','Phosphorothioate': 'q',
                # 'Tricyclodeoxyribo': 'r', 'Boranophosphate': 'f',
                }
symbol_2_name = {'B': '2-N-Adamant-1-yl-Methylcarbonyl-2-Amino-Locked','D': '2-Aminoethoxymethyl',
                'E': '2-Aminoethyl', 'F': '2-Fluoro', 'H': '4-C-HydroxymethylDeoxyribo',
                'I': 'Altritol', 'J': '2-Aminopropyl', 'K': '2-Aminopropoxymethyl',
                'L': 'Locked','M': '2-Deoxy-2-N,4-C-Ethylene-Locked', 'N': 'Alfa-L-Locked',
                'O': '2-Cyanoethyl','P': '2,4-Carbocyclic-Ethylene-bridged-Locked',
                'Q': '2,4-Carbocyclic-Locked-Locked','R': '2-Guanidinoethyl',
                'S': 'Hexitol','V': 'Oxetane-Locked','W': '2-N-Pyren-1-ylMethyl-2-Amino-Locked',
                'X': 'UnLocked','Y': '2-O-Methyl','Z': '2-Deoxy',
                # 'f': 'Boranophosphate', 'd': '2-O-Benzyl','e': '2-O-Guanidinopropyl',
                # 'h': 'Dihydrouridine', 'm': 'Hypoxanthine',
                'n': 'Peptides', 'q': 'Phosphorothioate',
                # 'r': 'Tricyclodeoxyribo',
                'A': None,  'G': None,  'C': None,  'U': None, 't': None,
                }
# 位置有sugar、base、phosphate，分别记为1，2，3
name_2_pos = {'2-N-Adamant-1-yl-Methylcarbonyl-2-Amino-Locked': 1, '2-Aminoethoxymethyl': 1,
                '2-Aminoethyl': 1,'2-Fluoro': 1,'4-C-HydroxymethylDeoxyribo': 1,'Altritol': 1,
                '2-Aminopropyl': 1,'2-Aminopropoxymethyl': 1,'Locked': 1,
                '2-Deoxy-2-N,4-C-Ethylene-Locked': 1,'Alfa-L-Locked': 1,'2-Cyanoethyl': 1,
                '2,4-Carbocyclic-Ethylene-bridged-Locked': 1,'2,4-Carbocyclic-Locked-Locked': 1,
                '2-Guanidinoethyl': 1,'Hexitol': 1,'Oxetane-Locked': 1,'2-N-Pyren-1-ylMethyl-2-Amino-Locked': 1,
                'UnLocked': 1,'2-O-Methyl': 1,'2-Deoxy': 1,
                # '2-O-Benzyl': 'd', '2-O-Guanidinopropyl': 'e', # 'Dihydrouridine': 'h'.  'Hypoxanthine': 'm',
                'Peptides': 3,'Phosphorothioate': 3,
                # 'Tricyclodeoxyribo': 'r', 'Boranophosphate': 'f',
                }
dict_onehot = {
        'A': 0,  'G': 1,  'C': 2,  'U': 3, 't': 4,
        'B': 5,'D': 6, 'E': 7, 'F': 8, 'H': 9, 'I': 10, 'J': 11, 'K': 12,
        'L': 13,'M': 14, 'N': 15, 'O': 16,'P': 17, 'Q': 18,'R': 19,
        'S': 20,'V': 21,'W': 22, 'X': 23,'Y': 24,'Z': 25, 'n': 26, 'q': 27}

rdkit_fingerprints = np.load(
    str(BASE0_DIR / 'data' / 'fingerprints.npy'), allow_pickle=True
).item()
gvae_represents = np.load(
    str(BASE0_DIR / 'data' / 'gvae.npy'), allow_pickle=True, encoding='latin1'
).item()  # python2中默认是ASCII编码，而python3中是utf8


# ---------------- cell-type (source) encoding ----------------
# Index 0 covers missing labels and Not reported. The historical literal
# Unknown cell line keeps its own index (18).
_CELL_TYPE_LIST = [
    'Be(2)C cell line', 'COS7', 'HEK293A', 'Hela', 'Hep3B', 'HepG2', 'Hepa1-6', 'Huh7',
    'Human iPSC-derived cortical neurons', 'Non-human hepatocytes',
    'Primary Cynomolgus Monkey Hepatocytes', 'Primary Macaque Hepatocytes',
    'Primary human hepatocytes', 'Primary mouse hepatocytes',
    'RT-4', 'SH-SY5Y cell line', 'T98G cell', 'Unknown cell line',
]
CELL_TYPE2IDX = {ct: i + 1 for i, ct in enumerate(_CELL_TYPE_LIST)}
NUM_CELL_TYPES = len(_CELL_TYPE_LIST) + 1  # +1 for index-0 unknown

# ---------------- variable-length / auto-padding helpers ----------------
DEFAULT_MIN_LEN = 25


def effective_cm_len(cm_seq):
    """Return nucleotide-position length of a modified sequence.

    In this codebase, symbol 'q' means phosphorothioate/backbone modification
    attached to the current nucleotide position, so it should NOT advance the
    nucleotide-position counter. Other symbols occupy one nucleotide position.
    """
    cm_seq = str(cm_seq).strip()
    n = 0
    for ch in cm_seq:
        if ch != 'q':
            n += 1
    return n


def infer_max_len_from_dataframe(df, min_len=DEFAULT_MIN_LEN):
    """Infer the padded sequence length from the current dataset."""
    max_len = int(min_len)
    for col in ['ori_as', 'ori_ss']:
        if col in df.columns:
            max_len = max(max_len, int(df[col].astype(str).str.strip().str.len().max()))
    for col in ['cm_as', 'cm_ss']:
        if col in df.columns:
            max_len = max(max_len, int(df[col].astype(str).map(effective_cm_len).max()))
    return max_len


def normalize_seq_value(x):
    return str(x).strip()


def validate_known_symbols(seq, allowed, seq_name='sequence'):
    bad = sorted(set(str(seq)) - set(allowed))
    if bad:
        raise ValueError(f"Unknown symbol(s) in {seq_name}: {bad}; sequence={seq}")

#### 子函数调用 #######------------------------

def onehot_encoding_(cm_seq, max_len=DEFAULT_MIN_LEN):
    # max_len * 28; q modifies current position and does not consume a new base position
    dim = 28
    dearth_ = np.array([0.05] * dim, dtype=np.float32)
    result = np.zeros((max_len, dim), dtype=np.float32)

    cm_seq = normalize_seq_value(cm_seq)
    validate_known_symbols(cm_seq, dict_onehot.keys(), 'cm_seq')

    seq_ind, res_ind = 0, 0
    while seq_ind < len(cm_seq) and res_ind < max_len:
        cur_symbel = cm_seq[seq_ind]
        result[res_ind, dict_onehot[cur_symbel]] = 1
        seq_ind += 1
        if cur_symbel != 'q':
            res_ind += 1

    for i in range(res_ind, max_len):
        result[i] = dearth_

    return result

def get_gc_sterch(ori_seq):  # 1,
    seq = ori_seq.upper()
    max_len, tem_len = 0, 0
    for i in range(len(seq)):
        if seq[i] == 'G' or seq[i] == 'C':
            tem_len += 1
            max_len = max(max_len, tem_len)
        else:
            tem_len = 0

    result = round((max_len / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_gc_percentage(ori_seq):  # 1,
    seq = ori_seq.upper()
    result = round(((seq.count('C') + seq.count('G')) / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_single_comp_percent(cm_seq):  # 28
    nt_percent = []
    for base_i in symbol_2_name.keys():
        nt_percent.append(round((cm_seq.count(base_i) / len(cm_seq)), 3))
    return np.array(nt_percent)[:, np.newaxis]

def get_di_comp_percent(ori_seq):  # 25,
    seq = ori_seq.upper() #初始序列是不会有a/c/g的脱氧核苷酸的，修饰后的才可能会有。而修饰前也是有t的脱氧核苷酸的，所以bases中有五种
    bases = ['A', 'G', 'C', 'U', 't']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / (len(seq) - 1)), 3))
    return np.array(di_nt_percent)[:, np.newaxis]

def get_tri_comp_percent(ori_seq):  # 125,
    seq = ori_seq.upper()
    bases = ['A', 'G', 'C', 'U', 't']
    pmt = list(product(bases, repeat=3))
    tri_nt_percent = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0] + pmt_i[1] + pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt) / (len(seq) - 2)), 3))
    return np.array(tri_nt_percent)[:, np.newaxis]


def sequence_encoding_(cm_seq, ori_seq, max_len=DEFAULT_MIN_LEN):
    # max_len * (5 + 101 + 56): original-base onehot + modification RDKit/GVAE features
    dim = 5 + 101 + 56
    dearth_ = np.array([0.05] * dim, dtype=np.float32)
    result = np.zeros((max_len, dim), dtype=np.float32)
    base_onehot = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 't': 4, 'T': 4}

    cm_seq = normalize_seq_value(cm_seq)
    ori_seq = normalize_seq_value(ori_seq)
    validate_known_symbols(ori_seq, base_onehot.keys(), 'ori_seq')
    validate_known_symbols(cm_seq, dict_onehot.keys(), 'cm_seq')

    # Fill original-sequence onehot. Longer sequences are supported up to max_len.
    for i, nt in enumerate(ori_seq[:max_len]):
        result[i, base_onehot[nt]] = 1

    # Fill chemical-modification descriptors. q stays on the current nucleotide position.
    seq_ind, res_ind = 0, 0
    while seq_ind < len(cm_seq) and res_ind < max_len:
        cur_symbel = cm_seq[seq_ind]
        if cur_symbel not in base_onehot:
            rdkit = rdkit_fingerprints[cur_symbel].astype(float)
            gvae = gvae_represents[cur_symbel].astype(float)
        else:
            rdkit, gvae = np.zeros((101)), np.zeros((56))

        result[res_ind, 5:5+101] += rdkit
        result[res_ind, 106:] += gvae
        seq_ind += 1
        if cur_symbel != 'q':
            res_ind += 1

    # Padding positions get a small non-zero sentinel value, preserving original behavior.
    filled_len = min(max(len(ori_seq), effective_cm_len(cm_seq)), max_len)
    for i in range(filled_len, max_len):
        result[i] = dearth_

    return result

def score_seq_by_pssm(pssm, ori_seq):  # 1,
    nt_order = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 't':4, 'N': 0}
    seq = ori_seq[:19].ljust(19, 'N')  # 补齐到19nt
    ind_all = list(range(0, len(seq)))
    scores = [pssm[nt_order.get(nt, 0), i] for nt, i in zip(seq, ind_all)]
    log_score = sum([-math.log2(max(i, 1e-10)) for i in scores])
    return np.array([log_score])[:, np.newaxis]

def create_pssm(train_seq):
    min_len = 19
    # 截断到19nt，不足19的用'N'补齐
    train_seq = [list(seq[0:min_len].ljust(min_len, 'N')) for seq in train_seq]
    train_seq = np.array(train_seq)

    nr, nc = np.shape(train_seq)
    pseudocount = nr ** 0.5  # Introduce a pseudocount (sqrt(N)) to make sure that we do not end up with a score of 0
    bases = ['A', 'G', 'C', 'U', 't']
    pssm = []
    for c in range(0, nc): #去掉两个游离的,前面去掉过了
        col_c = train_seq[:, c].tolist()
        f_A = round(((col_c.count('A') + pseudocount) / (nr + pseudocount)), 3)
        f_G = round(((col_c.count('G') + pseudocount) / (nr + pseudocount)), 3)
        f_C = round(((col_c.count('C') + pseudocount) / (nr + pseudocount)), 3)
        f_U = round(((col_c.count('U') + pseudocount) / (nr + pseudocount)), 3)
        f_T = round(((col_c.count('t') + pseudocount) / (nr + pseudocount)), 3)
        pssm.append([f_A, f_G, f_C, f_U, f_T])
    pssm = np.array(pssm)
    pssm = pssm.transpose()  # Make each column correspond to the nucleotide position and each row have the frequencies
    # for different nucleotides at that position.
    return pssm

def Energy_index(cm_seq, max_len=DEFAULT_MIN_LEN):
    dearth_ = 0  # padding value (scalar; np.array([0]) errors on numpy>=2.0)
    # 0 is padding index; real symbols use dict_onehot[symbol] + 1
    result = np.zeros((max_len), dtype=np.float32)
    cm_seq = normalize_seq_value(cm_seq)
    validate_known_symbols(cm_seq, dict_onehot.keys(), 'cm_seq')

    seq_ind, res_ind = 0, 0
    while seq_ind < len(cm_seq) and res_ind < max_len:
        cur_symbel = cm_seq[seq_ind]
        result[res_ind] = dict_onehot[cur_symbel] + 1
        seq_ind += 1
        if cur_symbel != 'q':
            res_ind += 1

    for i in range(res_ind, max_len):
        result[i] = dearth_

    return result


####  加载解析数据  ----------------------------------------------------------------




class Generate_dataset(Dataset):
    def __init__(self, dataset, pssm_as, pssm_ss, max_len=DEFAULT_MIN_LEN, conc_log_mean=1.0, conc_log_std=1.0):
        super().__init__()
        self.max_len = int(max_len)
        self.conc_log_mean = float(conc_log_mean)
        self.conc_log_std = float(conc_log_std) if float(conc_log_std) > 1e-8 else 1.0
        self.dict, self.num = self.load_sirna(dataset, pssm_as, pssm_ss)
    def __getitem__(self, index):
        return self.dict[index]
    def __len__(self):
        return self.num
    
    def load_sirna(self, dataset, pssm_as, pssm_ss):
        ori_as, ori_ss, cm_as, cm_ss, inhibition, concentration_nM = \
            dataset['ori_as'], dataset['ori_ss'], dataset['cm_as'], dataset['cm_ss'], dataset['inhibition'], dataset['concentration_nM']
        rnafm_ind = dataset.get('rnafm_ind', None)
        mrna_thermo_arr = dataset.get('mrna_thermo', None)
        cell_type_idx_arr = dataset.get('cell_type_idx', None)
        result_dict = []

        for i in range(len(ori_as)):
            dict_tem = {}

            dict_tem['as_enenrgy_index'] = Energy_index(cm_as[i], self.max_len)
            dict_tem['ss_enenrgy_index'] = Energy_index(cm_ss[i], self.max_len)

            dict_tem['cm_as_embedding'] = sequence_encoding_(cm_as[i], ori_as[i], self.max_len)
            dict_tem['cm_ss_embedding'] = sequence_encoding_(cm_ss[i], ori_ss[i], self.max_len)

            # ---- 修饰辅助特征 (15维 per position) ----
            aux_as = compute_all_aux_features(
                cm_as[i], ori_as=ori_as[i], ori_ss=ori_ss[i],
                max_len=self.max_len, seq_len=len(ori_as[i])
            )
            aux_ss = compute_all_aux_features(
                cm_ss[i], ori_as=ori_ss[i], ori_ss=ori_as[i],
                max_len=self.max_len, seq_len=len(ori_ss[i])
            )
            dict_tem['as_mod_aux'] = aux_as['per_position_aux'].astype(np.float32)  # (23, 15)
            dict_tem['ss_mod_aux'] = aux_ss['per_position_aux'].astype(np.float32)  # (23, 15)

            dict_tem['cm_as_onehot'] = onehot_encoding_(cm_as[i], self.max_len)
            dict_tem['cm_ss_onehot'] = onehot_encoding_(cm_ss[i], self.max_len)

            dict_tem['as_tri_nt_percent'] = get_tri_comp_percent(ori_as[i])  # 125,
            dict_tem['as_di_nt_percent'] = get_di_comp_percent(ori_as[i])  # 25,
            dict_tem['as_single_nt_percent'] = get_single_comp_percent(cm_as[i])  # 28
            dict_tem['as_gc_content'] = get_gc_percentage(ori_as[i])  # 1,
            dict_tem['as_gc_sterch'] = get_gc_sterch(ori_as[i])  # 1,
            dict_tem['as_pssm_score'] = score_seq_by_pssm(pssm_as, ori_as[i][:19])  # 1,

            dict_tem['ss_tri_nt_percent'] = get_tri_comp_percent(ori_ss[i])  # 125,
            dict_tem['ss_di_nt_percent'] = get_di_comp_percent(ori_ss[i])  # 25,
            dict_tem['ss_single_nt_percent'] = get_single_comp_percent(cm_ss[i])  # 31,
            dict_tem['ss_gc_content'] = get_gc_percentage(ori_ss[i])  # 1,
            dict_tem['ss_gc_sterch'] = get_gc_sterch(ori_ss[i])  # 1,
            dict_tem['ss_pssm_score'] = score_seq_by_pssm(pssm_ss, ori_ss[i][:19])  # 1,

            conc_nm = float(concentration_nM[i])
            log_conc = float(np.log10(max(conc_nm, 1e-9)))
            log_conc_z = (log_conc - self.conc_log_mean) / self.conc_log_std
            dict_tem['concentration_nM'] = np.array([conc_nm], dtype=np.float32)
            dict_tem['concentration_feature'] = np.array([log_conc, log_conc_z], dtype=np.float32)

            dict_tem['inhibit'] = inhibition[i]
            dict_tem['attsipred_index'] = np.zeros((1, 16), dtype=np.float32)  # dummy, Base0_abla_attsipred不使用

            # Cell-type index for source embedding
            if cell_type_idx_arr is not None:
                dict_tem['cell_type_idx'] = np.array([cell_type_idx_arr[i]], dtype=np.int64)
            else:
                dict_tem['cell_type_idx'] = np.array([0], dtype=np.int64)

            # ---- mRNA 特征 ----
            # per-residue RNA-FM: 用固定行号 rnafm_ind 取, 不受过滤/切分影响
            if rnafm_ind is not None and MRNA_RNAFM is not None:
                ridx = int(rnafm_ind[i])
                dict_tem['mrna_rnafm'] = np.asarray(MRNA_RNAFM[ridx], dtype=np.float32)  # (65,640)
            else:
                dict_tem['mrna_rnafm'] = np.zeros((MRNA_MAX_LEN, 640), dtype=np.float32)
            # 5个热力学标量 (NaN填0; BN会自动归一化各维度尺度)
            if mrna_thermo_arr is not None:
                dict_tem['mrna_thermo'] = np.nan_to_num(
                    np.asarray(mrna_thermo_arr[i], dtype=np.float32), nan=0.0)
            else:
                dict_tem['mrna_thermo'] = np.zeros((5,), dtype=np.float32)

            result_dict.append(dict_tem)

        return result_dict, len(ori_as)


def get_dataloader(args, all_data, train_indx, test_indx):
    validate_mrna_rnafm_for_model(getattr(args, 'model', ''), len(all_data))

    def _extract_concentration_nm(data):
        # Prefer an already-clean numeric nM column. If a legacy dataset has no
        # concentration column, assume the historical Dsicm10/Base0 condition: 10 nM.
        cand_cols = ['concentration/nM', 'concentration_nM', 'concentration_nm', 'Concentration_nM']
        found = None
        for c in cand_cols:
            if c in data.columns:
                found = c
                break
        if found is None:
            conc = pd.Series([10.0] * len(data), index=data.index, dtype=float)
        else:
            conc = pd.to_numeric(data[found], errors='coerce')
            # For merged legacy rows with missing concentration, use 10 nM rather than crash.
            conc = conc.fillna(10.0)
        conc = conc.astype(float).clip(lower=1e-9)
        return np.array(conc, dtype=np.float32)

    def getdict(data):
        d = {'ori_as': np.array(data['ori_as'].astype(str).str.strip()),
                'ori_ss': np.array(data['ori_ss'].astype(str).str.strip()),
                'cm_as': np.array(data['cm_as'].astype(str).str.strip()),
                'cm_ss': np.array(data['cm_ss'].astype(str).str.strip()),
                'inhibition': np.array(data['inhibit/0-1'] if 'inhibit/0-1' in data.columns else data['norm_inhibit'], dtype=np.float32),
                'concentration_nM': _extract_concentration_nm(data),
                }
        # mRNA: 固定行号 + 热力学标量
        if 'rnafm_ind' in data.columns:
            d['rnafm_ind'] = np.array(data['rnafm_ind'], dtype=int)
        if all(c in data.columns for c in MRNA_THERMO_COLS):
            d['mrna_thermo'] = np.array(
                data[MRNA_THERMO_COLS].apply(pd.to_numeric, errors='coerce').values,
                dtype=np.float32)  # (n, 5)
        if 'cell_type_idx' in data.columns:
            d['cell_type_idx'] = np.array(data['cell_type_idx'], dtype=np.int64)
        return d

    test_data = pd.DataFrame([all_data.iloc[i, :] for i in test_indx])
    train_data = pd.DataFrame([all_data.iloc[i, :] for i in train_indx])

    result_train = getdict(train_data)
    result_test = getdict(test_data)

    # Auto padding length. If args.max_len > 0 is provided, use it; otherwise infer from this train+test split.
    if hasattr(args, 'max_len') and int(args.max_len) > 0:
        max_len = int(args.max_len)
    else:
        max_len = max(
            DEFAULT_MIN_LEN,
            max(len(x) for x in result_train['ori_as']), max(len(x) for x in result_train['ori_ss']),
            max(len(x) for x in result_test['ori_as']), max(len(x) for x in result_test['ori_ss']),
            max(effective_cm_len(x) for x in result_train['cm_as']), max(effective_cm_len(x) for x in result_train['cm_ss']),
            max(effective_cm_len(x) for x in result_test['cm_as']), max(effective_cm_len(x) for x in result_test['cm_ss']),
        )
    print(f'[info] auto padding max_len = {max_len}')

    if max_len != 25:
        print(f'[warning] max_len is {max_len}; for pad25+flatten experiments use --max_len 25.')

    pssm_as = create_pssm(result_train['ori_as'])
    pssm_ss = create_pssm(result_train['ori_ss'])

    train_log_conc = np.log10(np.clip(result_train['concentration_nM'].astype(float), 1e-9, None))
    conc_log_mean = float(np.mean(train_log_conc))
    conc_log_std = float(np.std(train_log_conc)) if float(np.std(train_log_conc)) > 1e-8 else 1.0

    trainset = Generate_dataset(result_train, pssm_as, pssm_ss, max_len=max_len,
                                conc_log_mean=conc_log_mean, conc_log_std=conc_log_std)
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    testset = Generate_dataset(result_test, pssm_as, pssm_ss, max_len=max_len,
                               conc_log_mean=conc_log_mean, conc_log_std=conc_log_std)
    test_loader = DataLoader(testset, batch_size=testset.__len__(), shuffle=False, drop_last=False)

    return train_loader, test_loader


def _load_data(data_path):
    """读取数据文件，支持 csv 和 xlsx；只做 strip，不截断。"""
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path, header=0)
    else:
        data = pd.read_excel(data_path, header=0)
    data = data.reset_index(drop=True)
    # ===== 关键: 固定行号作为 mRNA embedding 的对齐 key =====
    # 必须在任何过滤/切分之前设置, 且之后绝不再 reset_index。
    # 这样无论 10-fold 还是 leave-gene 怎么切, 每行都带着它原始的 embedding 行号。
    data['rnafm_ind'] = np.arange(len(data), dtype=int)

    # Cell-type index for source embedding
    if 'Cell_Type' in data.columns:
        normalized_cell_types = data['Cell_Type'].where(
            data['Cell_Type'].isna(), data['Cell_Type'].astype(str).str.strip()
        )
        allowed_cell_types = set(CELL_TYPE2IDX) | {'Not reported'}
        unexpected_cell_types = set(normalized_cell_types.dropna()) - allowed_cell_types
        if unexpected_cell_types:
            raise ValueError(
                f'Unexpected Cell_Type labels: {sorted(unexpected_cell_types)}'
            )
        data['cell_type_idx'] = normalized_cell_types.map(CELL_TYPE2IDX).fillna(0).astype(int)
    else:
        data['cell_type_idx'] = 0

    if MRNA_RNAFM is not None and len(data) != MRNA_RNAFM.shape[0]:
        print(f'[warn] data rows ({len(data)}) != mRNA embedding rows ({MRNA_RNAFM.shape[0]}); '
              f'确认训练用的CSV和生成embedding用的是同一个文件同样顺序!')
    for col in ['ori_as', 'ori_ss', 'cm_as', 'cm_ss']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    # Use the cleaned label column when available. CMsiRNAdb rows often have
    # empty norm_inhibit, while inhibit/0-1 is complete in the cleaned dataset.
    if 'inhibit/0-1' in data.columns:
        data['inhibit/0-1'] = pd.to_numeric(data['inhibit/0-1'], errors='coerce')
        if 'norm_inhibit' not in data.columns:
            data['norm_inhibit'] = data['inhibit/0-1']
        else:
            data['norm_inhibit'] = pd.to_numeric(data['norm_inhibit'], errors='coerce')
            data['norm_inhibit'] = data['norm_inhibit'].fillna(data['inhibit/0-1'])

    print(f'[info] loaded {data_path}, inferred global max_len = {infer_max_len_from_dataframe(data)}')
    return data


def get_dataloader_kcross(args, test_type, test_sample_interval=10, data_path='data/merged_10nM.csv'):
    # 10-fold cross validation
    data_path = getattr(args, 'data_path', data_path)
    all_data = _load_data(data_path)
    rows = all_data.shape[0]
    start_ind = int(test_type.split('-')[0])
    test_indx = np.arange(start_ind, rows, test_sample_interval)
    train_indx = np.array(list(set(list(range(0, rows))) - set(list(test_indx))))
    return get_dataloader(args, all_data, train_indx, test_indx)


def get_dataloader_div_by_gene(args, test_gene, data_path='data/merged_10nM.csv'):
    """按靶基因做 leave-one-gene-out 测试"""
    data_path = getattr(args, 'data_path', data_path)
    all_data = _load_data(data_path)
    rows = all_data.shape[0]
    all_genes = np.array(all_data['target_gene'])
    test_indx = np.where(all_genes == test_gene)[0]
    train_indx = np.array(list(set(range(rows)) - set(test_indx.tolist())))
    return get_dataloader(args, all_data, train_indx, test_indx)


def get_dataloader_div_by_ori_AS(args, test_ori_as, data_path='data/merged_10nM.csv'):
    """按 ori_as 序列做 leave-one-out 测试 (兼容旧接口)"""
    data_path = getattr(args, 'data_path', data_path)
    all_data = _load_data(data_path)
    rows = all_data.shape[0]
    all_ori_as = np.array(all_data['ori_as'])
    test_indx = np.where(all_ori_as == test_ori_as)[0]
    if len(test_indx) == 0:
        raise ValueError('{} not found in data'.format(test_ori_as))
    train_indx = np.array(list(set(range(rows)) - set(test_indx.tolist())))
    return get_dataloader(args, all_data, train_indx, test_indx)
