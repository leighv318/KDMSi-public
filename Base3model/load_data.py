import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import product
import math
import RNA
import os
from pathlib import Path

try:
    from .rnafm_cache import (
        DEFAULT_CACHE_DIR as DEFAULT_RNAFM_CACHE_DIR,
        DEFAULT_DATAFILE,
        load_cached_embedding,
        prepare_rnafm_cache,
    )
except ImportError:  # direct execution from Base3model/train.py
    from rnafm_cache import (
        DEFAULT_CACHE_DIR as DEFAULT_RNAFM_CACHE_DIR,
        DEFAULT_DATAFILE,
        load_cached_embedding,
        prepare_rnafm_cache,
    )

########################## 各种特征 #############################

def get_gc_sterch(seq):  # 1,
    max_len, tem_len = 0, 0
    for i in range(len(seq)):
        if seq[i] == 'G' or seq[i] == 'C':
            tem_len += 1
            max_len = max(max_len, tem_len)
        else:
            tem_len = 0

    result = round((max_len / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_gc_percentage(seq):  # 1,
    result = round(((seq.count('C') + seq.count('G')) / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_single_comp_percent(seq):  # 4,
    nt_percent = []
    for base_i in list(['A', 'G', 'C', 'U']):
        nt_percent.append(round((seq.count(base_i) / len(seq)), 3))
    return np.array(nt_percent)[:, np.newaxis]

def get_di_comp_percent(seq):  # 16,
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / (len(seq) - 1)), 3))
    return np.array(di_nt_percent)[:, np.newaxis]

def get_tri_comp_percent(seq):  # 64,
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=3))
    tri_nt_percent = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0] + pmt_i[1] + pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt) / (len(seq) - 2)), 3))
    return np.array(tri_nt_percent)[:, np.newaxis]

def single_encoding_4dim(sequence):  # 21*4
    single_encode_dict = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    result = []
    for i in range(len(sequence)):
        if sequence[i] in single_encode_dict.keys():
            tem = single_encode_dict.get(sequence[i])
        else:
            tem = single_encode_dict.get('A')
        result.append(tem)
    return np.stack(result)

def secondary_struct(seq):  # 2+1, 前者是配/不配对的百分比。后者是预测的最小自由能
    def _frequency(if_paired):
        paired_result, unpaired_result = [], []
        for i in range(len(if_paired)):
            if if_paired[i] == '.':
                paired_result.append(0)
                unpaired_result.append(1)
            else:
                paired_result.append(1)
                unpaired_result.append(0)
        return np.array(paired_result + unpaired_result)[:, np.newaxis]

    def _percentage(if_paired):
        paired_percent = (if_paired.count('(') + if_paired.count(')')) / len(if_paired)
        unpaired_percent = (if_paired.count('.')) / len(if_paired)
        return np.array([[paired_percent], [unpaired_percent]])

    paired_seq, min_free_energy = RNA.fold(seq)
    return _percentage(paired_seq), np.array([min_free_energy])[:, np.newaxis]


def score_seq_by_pssm(pssm, seq):  # 1,
    nt_order = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    ind_all = list(range(0, len(seq)))
    scores = [pssm[nt_order[nt], i] for nt, i in zip(seq, ind_all)]  # 对seq的每个位置的情况遍历，共19次得分
    log_score = sum([-math.log2(i) for i in scores])
    return np.array([log_score])[:, np.newaxis]

def gibbs_energy(seq):  # 20 —— 18个相邻碱基对，1个siRNA两端热力学差，1个总的能量
    energy_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'table': np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}

    result = []
    for i in range(len(seq)-1):
        index_1 = energy_dict.get(seq[i])
        index_2 = energy_dict.get(seq[i + 1])
        result.append(energy_dict['table'][index_1, index_2])

    result.append(np.array(result).sum().round(3))
    result.append((result[0] - result[-2]).round(3))  # 总的，和两端差

    result = np.array(result)[:, np.newaxis]
    return result  # / abs(result).max()

def gibbs_energy_embedding_single(seq):
    # 19 —— 19个单碱基对的embedding
    energy_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    result = np.zeros((19))
    for i in range(len(seq)):
        result[i] = energy_index[seq[i]]

    return result  # / abs(result).max()

def gibbs_energy_embedding_adjoint(seq):
    # 18 —— 18个相邻碱基对的embedding
    energy_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    result = np.zeros((18))
    for i in range(len(seq)-1):
        result[i] = energy_index[seq[i]] * 4 + energy_index[seq[i+1]]

    return result  # / abs(result).max()


########################## siRNA-mRNA 交互热力学特征 #############################

def clean_rna(seq):
    """Convert sequence to uppercase RNA alphabet; keep dot padding for window indexing."""
    if not isinstance(seq, str):
        seq = str(seq)
    return seq.upper().replace('T', 'U')


def revcomp_rna(seq):
    """Reverse-complement RNA sequence. Dot padding is kept as dot."""
    seq = clean_rna(seq)
    comp = str.maketrans('AUGC.N', 'UACG.N')
    return seq.translate(comp)[::-1]


def get_target_19_from_window(mrna_seq, center=29):
    """
    Extract the 19nt target region from a 59nt mRNA window.
    Uses fixed center index=29 as anchor; does NOT strip dots before finding center.
    """
    seq = clean_rna(mrna_seq)
    if len(seq) >= 59:
        target = seq[center - 9:center + 10]
    else:
        c = len(seq) // 2
        target = seq[max(0, c - 9):c + 10]
    return target.replace('.', '')


def _compute_duplex_energy(as_seq, mrna_seq):
    guide = clean_rna(as_seq)[:19]
    target_19 = get_target_19_from_window(mrna_seq)
    if len(guide) < 5 or len(target_19) < 5:
        return np.array([[0.0], [0.0]])
    try:
        result = RNA.duplexfold(guide, target_19)
        dg = float(result.energy)
    except Exception:
        try:
            _, dg = RNA.cofold(guide + '&' + target_19)
            dg = float(dg)
        except Exception:
            dg = 0.0
    n_bp = min(len(guide), len(target_19))
    dg_norm = dg / n_bp if n_bp > 0 else 0.0
    return np.array([[dg], [dg_norm]])


def _compute_seed_energy(as_seq, mrna_seq):
    """
    guide[j] pairs with target_19[18-j] (antiparallel).
    So: guide[1:8] (seed) pairs with target_19[11:18] in mRNA original orientation.
         guide[8:19] (non-seed) pairs with target_19[0:11].
    duplexfold expects two separate strands, NOT reverse-complemented.
    """
    guide = clean_rna(as_seq)[:19]
    target_19 = get_target_19_from_window(mrna_seq)
    if len(guide) < 19 or len(target_19) < 19:
        return np.array([[0.0], [0.0]])
    # seed: guide pos 2-8 (index 1:8) ↔ mRNA target_19[11:18]
    seed_guide = guide[1:8]
    seed_target = target_19[11:18]
    # non-seed: guide pos 9-19 (index 8:19) ↔ mRNA target_19[0:11]
    nonseed_guide = guide[8:19]
    nonseed_target = target_19[0:11]
    try:
        seed_dg = float(RNA.duplexfold(seed_guide, seed_target).energy)
    except Exception:
        try:
            _, seed_dg = RNA.cofold(seed_guide + '&' + seed_target)
            seed_dg = float(seed_dg)
        except Exception:
            seed_dg = 0.0
    if len(nonseed_guide) >= 5 and len(nonseed_target) >= 5:
        try:
            nonseed_dg = float(RNA.duplexfold(nonseed_guide, nonseed_target).energy)
        except Exception:
            try:
                _, nonseed_dg = RNA.cofold(nonseed_guide + '&' + nonseed_target)
                nonseed_dg = float(nonseed_dg)
            except Exception:
                nonseed_dg = 0.0
    else:
        nonseed_dg = 0.0
    return np.array([[seed_dg], [nonseed_dg]])


def _compute_target_accessibility(mrna_seq):
    clean = clean_rna(mrna_seq).replace('.', '')
    if len(clean) < 5:
        return np.array([[0.0], [0.0], [1.0]])
    try:
        _, full_mfe = RNA.fold(clean)
        full_mfe = float(full_mfe)
    except Exception:
        full_mfe = 0.0
    target_19 = get_target_19_from_window(mrna_seq)
    if len(target_19) < 5:
        return np.array([[full_mfe], [0.0], [1.0]])
    try:
        local_struct, local_mfe = RNA.fold(target_19)
        local_mfe = float(local_mfe)
        unpaired_ratio = local_struct.count('.') / len(local_struct) if len(local_struct) > 0 else 1.0
    except Exception:
        local_mfe = 0.0
        unpaired_ratio = 1.0
    return np.array([[full_mfe], [local_mfe], [unpaired_ratio]])


def precompute_thermo_features(datafile, cache_dir):
    """
    对整个 CSV 预计算交互热力学特征，保存到 cache_dir。
    只在第一次调用时计算，之后直接跳过。
    返回 3 个数组: duplex(N,2,1), seed(N,2,1), access(N,3,1)
    """
    os.makedirs(cache_dir, exist_ok=True)
    f_duplex = os.path.join(cache_dir, 'duplex_energy.npy')
    f_seed = os.path.join(cache_dir, 'seed_energy.npy')
    f_access = os.path.join(cache_dir, 'target_accessibility.npy')

    if os.path.exists(f_duplex) and os.path.exists(f_seed) and os.path.exists(f_access):
        # 缓存已存在，直接加载
        return np.load(f_duplex), np.load(f_seed), np.load(f_access)

    # 第一次：计算并保存
    import time
    print(f'[thermo] Precomputing interaction features for {datafile} ...')
    data = pd.read_csv(datafile, header=0)
    all_as = np.array(data['Antisense'])
    all_mrna = np.array(data['mrna'])
    n = len(all_as)

    duplex_arr = np.zeros((n, 2, 1))
    seed_arr = np.zeros((n, 2, 1))
    access_arr = np.zeros((n, 3, 1))

    t0 = time.time()
    for i in range(n):
        duplex_arr[i] = _compute_duplex_energy(all_as[i], all_mrna[i])
        seed_arr[i] = _compute_seed_energy(all_as[i], all_mrna[i])
        access_arr[i] = _compute_target_accessibility(all_mrna[i])
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f'  [{i+1}/{n}]  elapsed {elapsed:.1f}s  ETA {eta:.1f}s')

    np.save(f_duplex, duplex_arr)
    np.save(f_seed, seed_arr)
    np.save(f_access, access_arr)
    print(f'[thermo] Done! {n} samples in {time.time()-t0:.1f}s. Cached to {cache_dir}/')

    # sanity check
    for name, arr in [('duplex_energy', duplex_arr), ('seed_energy', seed_arr), ('target_accessibility', access_arr)]:
        flat = arr.squeeze()
        print(f'  {name}: shape={arr.shape}, min={np.nanmin(flat):.3f}, max={np.nanmax(flat):.3f}, mean={np.nanmean(flat):.3f}')

    return duplex_arr, seed_arr, access_arr


########################## PSSM #############################

def create_pssm(train_seq):
    train_seq = [list(seq.upper()) for seq in train_seq]
    train_seq = np.array(train_seq)

    nr, nc = np.shape(train_seq)
    pseudocount = nr ** 0.5
    bases = ['A', 'G', 'C', 'U']
    pssm = []
    for c in range(0, nc):
        col_c = train_seq[:, c].tolist()
        # Four bases all receive a pseudocount, so the denominator must add 4*pseudocount.
        denom = nr + 4 * pseudocount
        f_A = round(((col_c.count('A') + pseudocount) / denom), 3)
        f_G = round(((col_c.count('G') + pseudocount) / denom), 3)
        f_C = round(((col_c.count('C') + pseudocount) / denom), 3)
        f_U = round(((col_c.count('U') + pseudocount) / denom), 3)
        pssm.append([f_A, f_G, f_C, f_U])
    pssm = np.array(pssm)
    pssm = pssm.transpose()
    return pssm

def prepare_RNAFM(datafile=None, args=None):
    """Create/reuse Base3 embeddings from the separately released RNA-FM weight.

    This keeps the historical public function name while removing the dependency
    on an untracked ``RNAFM_ENCODING/redevelop`` source tree.
    """
    datafile = Path(datafile or DEFAULT_DATAFILE)
    cache_dir = getattr(args, 'rnafm_cache_dir', DEFAULT_RNAFM_CACHE_DIR)
    model_path = getattr(args, 'rnafm_model_path', None)
    batch_size = getattr(args, 'rnafm_batch_size', 64)
    rebuild = getattr(args, 'rebuild_rnafm', False)
    rnafm_device = getattr(args, 'rnafm_device', None)
    return prepare_rnafm_cache(
        datafile=datafile,
        cache_dir=cache_dir,
        model_path=model_path,
        batch_size=batch_size,
        rebuild=rebuild,
        device=rnafm_device,
    )

def process_mrna_RNAFM(ori_mrna, rnafm_mrna):
    assert rnafm_mrna.shape[0] == 59
    return rnafm_mrna

#### 子函数结束----------------------------------------------------------------


def load_sirna(args, dataset, pssm, thermo_cache=None):
    """
    pssm 是整个训练得到的一个4*19矩阵，测试时也用同一个矩阵
    thermo_cache: (duplex_all, seed_all, access_all, orig_indices) 预计算的热力学缓存
    :return: list. each element is a dict. keys={}
    """
    result_dict = []
    ori_as, ori_ss, mrna, mrna_len, nm_names, pos_inds, s_biopredsi, dsir, iscore, inhibition, rnafm_indexes = \
        dataset['seq'], dataset['ss_seq'], dataset['mrna'], dataset['mrna_len'], dataset['nm_names'], dataset['pos_inds'], \
        dataset['s-biopredsi'], dataset['dsir'], dataset['i-score'], dataset['inhibition'], dataset['RNAFM_ind']
    target_flag = dataset['target_flag']
    orig_indices = dataset['orig_idx']
    source_ids = dataset['source_id']   # 新增：H=0, T=1, R=2
    rnafm_cache_dir = Path(
        getattr(args, 'rnafm_cache_dir', DEFAULT_RNAFM_CACHE_DIR)
    ).expanduser().resolve()

    ori_as = np.array([ori_as[i].upper() for i in range(ori_as.shape[0])])

    # 解包预计算缓存
    duplex_all, seed_all, access_all = None, None, None
    if thermo_cache is not None:
        duplex_all, seed_all, access_all = thermo_cache

    for i in range(ori_as.shape[0]):
        dict_tem = {}

        dict_tem['onehot_encode'] = single_encoding_4dim(ori_as[i])
        dict_tem['onehot_encode_mrna'] = single_encoding_4dim(mrna[i])

        dict_tem['rnafm_encode'] = load_cached_embedding(
            rnafm_cache_dir, rnafm_indexes[i], 'as')
        dict_tem['rnafm_encode_mrna'] = load_cached_embedding(
            rnafm_cache_dir, rnafm_indexes[i], 'mrna')
        dict_tem['rnafm_encode_mrna'] = process_mrna_RNAFM(mrna[i], dict_tem['rnafm_encode_mrna'])

        dict_tem['sirna_gibbs_energy'] = gibbs_energy(ori_as[i][:19])
        dict_tem['sirna_gibbs_energy_embed_single'] = gibbs_energy_embedding_single(ori_as[i][:19])
        dict_tem['sirna_gibbs_energy_embed_adjoint'] = gibbs_energy_embedding_adjoint(ori_as[i][:19])
        dict_tem['pssm_score'] = score_seq_by_pssm(pssm, ori_as[i])
        dict_tem['sirna_second_percent'], dict_tem['sirna_second_energy'] = secondary_struct(ori_as[i])
        dict_tem['tri_nt_percent'] = get_tri_comp_percent(ori_as[i])
        dict_tem['di_nt_percent'] = get_di_comp_percent(ori_as[i])
        dict_tem['single_nt_percent'] = get_single_comp_percent(ori_as[i])
        dict_tem['gc_content'] = get_gc_percentage(ori_as[i])
        dict_tem['gc_sterch'] = get_gc_sterch(ori_as[i])

        # ---- 交互热力学特征：从预计算缓存中按原始行号索引 ----
        row_idx = orig_indices[i]
        if duplex_all is not None:
            dict_tem['duplex_energy'] = duplex_all[row_idx]
            dict_tem['seed_energy'] = seed_all[row_idx]
            dict_tem['target_accessibility'] = access_all[row_idx]
        else:
            # fallback: 填零（用 Base3 原模型时不会用到这些 key）
            dict_tem['duplex_energy'] = np.zeros((2, 1))
            dict_tem['seed_energy'] = np.zeros((2, 1))
            dict_tem['target_accessibility'] = np.zeros((3, 1))

        dict_tem['inhibit'] = inhibition[i]
        dict_tem['target_flag'] = target_flag[i]
        dict_tem['source_id'] = np.array([source_ids[i]])[:, np.newaxis]  # (1,1), float32

        result_dict.append(dict_tem)

    return result_dict

class Generate_dataset(Dataset):
    def __init__(self, args, dataset, pssm, thermo_cache=None):
        super().__init__()
        self.dict = load_sirna(args, dataset, pssm, thermo_cache=thermo_cache)
    def __getitem__(self, index):
        return self.dict[index]
    def __len__(self):
        return len(self.dict)


def get_dataloader_for_all_condition(args, all_data, train_indx, test_indx,
                                     thermo_cache=None,
                                     whole_mrna_path='./data/whole_mrna_seq.fasta',
                                     test_source_id_override=None):
    _SOURCE_MAP = {'H': 0, 'T': 1, 'R': 2}

    def getdict(data, indices):
        return {'seq': np.array(data['Antisense']),
                'ss_seq': np.array(data['Sense']),
                'mrna': np.array(data['mrna']),
                'mrna_len': np.array(data['mrna_length']),
                'nm_names': np.array(data['nm_names']),
                'pos_inds': np.array(data['pos_index']),
                's-biopredsi': np.array(data['s-Biopredsi']),
                'dsir': np.array(data['DSIR'])/100.,
                'i-score': np.array(data['i-score'])/100.,
                'inhibition': np.array(data['inhibition']),
                'RNAFM_ind': np.array(data['RNAFM_ind']),
                'target_flag':np.array(data['source_ndcg']),
                'orig_idx': indices,  # 原始 CSV 行号，用于索引预计算特征
                # source_paper → 整数 id: H=0, T=1, R=2
                'source_id': np.array([_SOURCE_MAP.get(s, 0) for s in data['source_paper']], dtype=np.float32),
                }

    test_data = pd.DataFrame([all_data.iloc[i, :] for i in test_indx])
    train_data = pd.DataFrame([all_data.iloc[i, :] for i in train_indx])

    result_train, result_test = getdict(train_data, train_indx), getdict(test_data, test_indx)

    # T/R fold：测试集 source_id 置 0（避免随机噪声）
    # 同时把训练集 source_id 也全部置 0：
    #   - T fold 训练集=H+R，R的embedding虽然学了但对T泛化没帮助，且会让模型更source-specific
    #   - R fold 训练集=H+T，同理
    #   - 置 0 后模型在 T/R fold 训练时完全不依赖 source 信息，学出更 source-agnostic 的特征
    if test_source_id_override is not None:
        # id=3 是 padding_idx（零向量），比 H=0 更中立：
        # 不携带任何来源的偏差信息，让模型完全依赖序列特征做预测
        result_test['source_id']  = np.full_like(result_test['source_id'],  3.0)
        result_train['source_id'] = np.full_like(result_train['source_id'], 3.0)

    # 修正：测试集也使用训练集的 PSSM
    pssm_as_train = create_pssm(result_train['seq'])

    trainset = Generate_dataset(args, result_train, pssm_as_train, thermo_cache=thermo_cache)
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    testset = Generate_dataset(args, result_test, pssm_as_train, thermo_cache=thermo_cache)
    test_loader = DataLoader(testset, batch_size=testset.__len__(), shuffle=False, drop_last=False)

    return train_loader, test_loader

def get_dataloader_rt_or_inter(args, test_type, datafile=None):
    datafile = str(Path(datafile or DEFAULT_DATAFILE).expanduser().resolve())
    # Generate once on the first split; subsequent folds reuse the validated cache.
    prepare_RNAFM(datafile=datafile, args=args)
    all_data = pd.read_csv(datafile, sep=',', header=0)
    all_sources = np.array(all_data['source_paper'])
    rows = all_data.shape[0]
    train_indx, test_indx = None, None

    if test_type.endswith('inter'):
        start_ind = int(test_type.split('-')[0])
        test_indx = np.arange(start_ind, rows, args.sample_internal)
    elif test_type == 'R' or test_type == 'T':
        test_indx = np.array(np.where(all_sources == test_type)[0])
    else:
        raise ValueError('unsupported test type : {}'.format(test_type))

    train_indx = np.array(list(set(list(range(0, rows))) - set(list(test_indx))))

    # 预计算交互热力学特征（第一次调用时计算并缓存，之后直接加载）
    cache_dir = str(Path(datafile).parent / 'thermo_cache_sirna')
    duplex_all, seed_all, access_all = precompute_thermo_features(datafile, cache_dir)
    thermo_cache = (duplex_all, seed_all, access_all)

    # 当整个来源被作为测试集时（T fold 或 R fold），该来源的 source embedding
    # 在训练阶段从未被更新，是随机初始化的噪声向量。
    # 修复：把测试集的 source_id 替换成训练数据中最大来源 H(=0)，
    # 让模型用熟悉的校正向量而非随机噪声。
    # inter fold 不受影响（H/T/R 均匀分布于训练集，embedding 均已学好）。
    source_id_override = test_type if test_type in ('T', 'R') else None

    train_loader, test_loader = get_dataloader_for_all_condition(
        args, all_data, train_indx, test_indx, thermo_cache=thermo_cache,
        test_source_id_override=source_id_override)
    return train_loader, test_loader
