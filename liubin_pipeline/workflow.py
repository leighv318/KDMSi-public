#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
2024.03.06
@author: created by Bin Liu | sjtu
"""


"""
command example:
    python workflow.py --filter sirna --g Homo --m 1,2,3 --obj NR,NM,XR,XM
    
    python workflow.py --filter sirnacm --g TTR --m 1,2,3 \
        --sirnacm_path output/sirnacm/TTR2.csv \
        --base0_target_fasta output/mrna/TTR.fasta

    python workflow.py --filter sirnacm --g TTR --m 5 \
        --base0_target_fasta output/mrna/TTR.fasta \
        --design_base3_top_n 100 --design_patterns_per_sirna 20 \
        --design_ss_length 21 --design_as_overhang target-derived

command params:
    g:
    target gene names
    m:
    to predict potency or offtarget effect, default potency prediction
    off-target filter just support human genes for now
    1 -> potency prediction
    2 -> mirna-like-offtarget
    3 -> homology-offtarget using <substr>
    4 -> homology-offtarget using improved Smith-Waterman (this method will cost much time)
    5 -> design modified siRNAs from FASTA using Base3 selection and Base0 reranking
    6 -> design modified siRNAs from user-provided naked siRNAs using Base0 reranking
    k:
    use offtarget filter for top-k sirnas; -1 means all candidates
    obj:
    types of mrnas (collected from Refseq database) to use in homology-offtarget prediction
"""
import os
os.environ['CUDA_VISIBLE_dEVICES']='0,1,2,3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import warnings
warnings.filterwarnings('ignore')
import torch
import pandas as pd
from utils.sliding_mrna import read_by_windows
from utils.logger import logging
from potency_prediction import pred_sirna
from off_target import *
import argparse

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)

def default_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = default_torch_device()

parser = argparse.ArgumentParser()
parser.add_argument("--g", type=str, default=None)
parser.add_argument("--m", type=str, default='1')
parser.add_argument('--k', type=int, default=-1, help="if k==-1, that means predicting on all samples. if k is a positive number, that means only predict on top-k samples")
parser.add_argument('--obj', type=str, default="NR,NM,XR,XM")
parser.add_argument('--logging_path', type=str, default="logging.txt")
parser.add_argument('--filter', type=str, default="sirna", choices=['sirna','sirnacm'])
parser.add_argument('--sirnacm_path', type=str, default=None)
parser.add_argument('--base3_model', type=str, default='Base3_cross_access_v2_src_no_se',
                    help='Base3 model class from Base3model/model_attention.py')
parser.add_argument('--base3_ckpt_dir', type=str,
                    default=os.path.join(PIPELINE_DIR, 'potency_prediction', 'pretrain_models',
                                         'Base3_cross_access_v2_src_no_se'),
                    help='Directory containing Base3 fold checkpoint subdirectories')
parser.add_argument('--base3_folds', type=str, default='0,1,2,3,4,5,6,7,8,9',
                    help='Comma-separated folds to average, e.g. 0,1,2 or 0-inter,1-inter')
parser.add_argument('--base3_train_data', type=str,
                    default=os.path.join(PROJECT_ROOT, 'Base3model', 'data',
                                         'normalized_sirna_with_mrna.csv'),
                    help='Training CSV used to rebuild fold-specific PSSM')
parser.add_argument('--base3_batch_size', type=int, default=128,
                    help='Batch size for Base3 model inference')
parser.add_argument('--base3_rnafm_batch_size', type=int, default=64,
                    help='Batch size for RNA-FM embedding during Base3 inference')
parser.add_argument('--base0_model', type=str, default='Base0_mrna_full_no_se',
                    help='Final Base0 model class from base0model/model.py')
parser.add_argument('--base0_ckpt_dir', type=str,
                    default=os.path.join(PIPELINE_DIR, 'potency_prediction', 'pretrain_models',
                                         'Base0_mrna_full_no_se'),
                    help='Directory containing Base0 fold checkpoint subdirectories')
parser.add_argument('--base0_folds', type=str, default='0,1,2,3,4,5,6,7,8,9',
                    help='Comma-separated Base0 folds to average')
parser.add_argument('--base0_train_data', type=str,
                    default=os.path.join(PROJECT_ROOT, 'base0model', 'data',
                                         'CMsiRNAdb_Dsicm10_FINAL_trainready.csv'),
                    help='Final 22,622-row Base0 training CSV for fold feature statistics')
parser.add_argument('--base0_target_fasta', type=str, default=None,
                    help='Target transcript FASTA; optional when input CSV contains local_mrna')
parser.add_argument('--base0_rnafm_cache', type=str, default='',
                    help='Optional RNA-FM cache path; default is derived from the input CSV')
parser.add_argument('--base0_rebuild_rnafm', action='store_true',
                    help='Rebuild the aligned Base0 mRNA RNA-FM cache')
parser.add_argument('--base0_batch_size', type=int, default=128)
parser.add_argument('--base0_rnafm_batch_size', type=int, default=64)
parser.add_argument('--base0_max_len', type=int, default=25)
parser.add_argument('--base0_mrna_max_len', type=int, default=65)
parser.add_argument('--base0_window_up', type=int, default=20)
parser.add_argument('--base0_window_down', type=int, default=20)
parser.add_argument('--base0_concentration_nm', type=float, default=10.0,
                    help='Scoring concentration; defaults to the 10 nM reference condition')
parser.add_argument('--base0_cell_type', type=str, default='',
                    help='Scoring cell type; omitted/empty means unknown cell type (index 0)')
parser.add_argument('--base0_as_ts', type=int, default=3)
parser.add_argument('--base0_ss_ts', type=int, default=3)
parser.add_argument('--base0_cell_embed_dim', type=int, default=8)
parser.add_argument('--design_base3_top_n', type=int, default=100,
                    help='Number of Base3-ranked 19-nt cores retained for modification design')
parser.add_argument('--design_patterns_per_sirna', type=int, default=20,
                    help='Unique modification patterns sampled for each retained Base3 core')
parser.add_argument('--design_final_top_n', type=int, default=100,
                    help='Number of Base0-ranked modified candidates to save; -1 saves all')
parser.add_argument('--design_ss_length', type=int, default=21,
                    help='Final sense-strand length; antisense length is sense length + 2')
parser.add_argument('--design_as_overhang', type=str, default='target-derived',
                    choices=['target-derived', 'UU', 'tt'],
                    help='Two-nucleotide antisense 3-prime overhang')
parser.add_argument('--design_seed', type=int, default=126,
                    help='Random seed for empirical modification-pattern sampling')
args = parser.parse_args()
if args.g is not None:
    args.g = args.g.split(',')
args.obj = args.obj.split(',')
args.m = args.m.split(',')
args.device = device


def _resolve_sirnacm_path(path_value):
    candidates = [path_value]
    if not os.path.isabs(path_value):
        candidates.extend([
            os.path.join(PIPELINE_DIR, path_value),
            os.path.join(PIPELINE_DIR, 'output', 'sirnacm', path_value),
        ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError('modified-siRNA CSV not found: {}'.format(path_value))


def _ensure_target_fasta(gene):
    if args.base0_target_fasta is not None:
        target_fasta = args.base0_target_fasta
        if not os.path.isabs(target_fasta):
            pipeline_relative = os.path.join(PIPELINE_DIR, target_fasta)
            if os.path.exists(pipeline_relative):
                target_fasta = pipeline_relative
        if not os.path.exists(target_fasta):
            raise FileNotFoundError('target FASTA not found: {}'.format(target_fasta))
        args.base0_target_fasta = os.path.abspath(target_fasta)
        return args.base0_target_fasta

    mrna_dir = os.path.join(PIPELINE_DIR, 'output', 'mrna')
    os.makedirs(mrna_dir, exist_ok=True)
    target_fasta = os.path.join(mrna_dir, '{}.fasta'.format(gene))
    if not os.path.exists(target_fasta):
        from utils.pygb_index import Genome
        seqs = Genome("GRCh38", data="rna").get_sequences_by_gene(gene)
        if not seqs:
            raise ValueError('failed finding target transcript for {}'.format(gene))
        with open(target_fasta, 'w') as handle:
            for name, sequence in seqs.items():
                handle.write('>{}\n{}\n'.format(name, sequence))
        print('[workflow] saved {} target transcript(s) to {}'.format(
            len(seqs), target_fasta
        ))
    args.base0_target_fasta = target_fasta
    return target_fasta


def _prepare_sirnacm_context():
    design_requested = '5' in args.m
    user_design_requested = '6' in args.m
    if design_requested and user_design_requested:
        raise ValueError('Do not combine method 5 and method 6 in one run')
    if design_requested:
        if '1' in args.m:
            raise ValueError('Method 5 already runs Base3 and Base0; do not combine --m 5 with method 1')
        if args.m[0] != '5':
            raise ValueError('When combined with off-target methods, method 5 must be first in --m')
        if args.g is None or len(args.g) != 1:
            raise ValueError('Method 5 requires exactly one target via --g GENE')
        _ensure_target_fasta(args.g[0])
        if args.sirnacm_path is None:
            return None

    if user_design_requested:
        if '1' in args.m:
            raise ValueError('Method 6 already runs Base0; do not combine --m 6 with method 1')
        if args.m[0] != '6':
            raise ValueError('When combined with off-target methods, method 6 must be first in --m')
        if args.sirnacm_path is None:
            raise ValueError('Method 6 requires --sirnacm_path pointing to a naked-siRNA CSV')
        if args.g is None or len(args.g) != 1:
            raise ValueError('Method 6 requires exactly one target via --g GENE')
        candidate_path = _resolve_sirnacm_path(args.sirnacm_path)
        args.sirnacm_path = candidate_path
        candidate_data = pd.read_csv(candidate_path)
        if 'local_mrna' not in candidate_data.columns:
            _ensure_target_fasta(args.g[0])
        return candidate_data

    if args.sirnacm_path is None:
        raise ValueError('--sirnacm_path is required when --filter sirnacm')
    candidate_path = _resolve_sirnacm_path(args.sirnacm_path)
    args.sirnacm_path = candidate_path
    candidate_data = pd.read_csv(candidate_path)

    if args.g is None and 'target_gene' in candidate_data.columns:
        genes = candidate_data['target_gene'].dropna().astype(str).str.strip().unique()
        genes = [gene for gene in genes if gene]
        if len(genes) == 1:
            args.g = [genes[0]]

    off_target_requested = any(method in {'2', '3', '4'} for method in args.m)
    needs_fasta_for_base0 = '1' in args.m and 'local_mrna' not in candidate_data.columns
    if off_target_requested or needs_fasta_for_base0:
        if args.g is None or len(args.g) != 1:
            raise ValueError(
                'Provide exactly one target with --g GENE (or one unique target_gene '
                'column) so the workflow can build the target FASTA and exclude true '
                'on-target transcripts.'
            )
        _ensure_target_fasta(args.g[0])

    return candidate_data

def __sirna_main():
    """
    workflow:
        1. for given gene names, get mature mrna sequences by pyGB package, and save to '/output/mrna'
           (or put mrna.fasta to this folder directly)
        2. for each mrna sequence, get all possible 19-nt SS via sliding windows, and save to '/output/sirna'
           (also get 21-nt AS, 59-nt local mrna, position of target site, etc.)
        3. use our pretrained model to predict potency for every sirna, sort in descending order by potency,
           and save to '/output/sirna'
        4. optional off-target filter for top-k sirnas, and save summarized result to '/output/sirna', detailed 
           info to '/output/mirna-like_offtarget', or '/output/homo_sw_offtarget', or '/output/homo_substr_offtarget'
    """
    os.makedirs('./output/mrna', exist_ok=True)
    if args.g is not None:
        from utils.pygb_index import Genome
        genome = Genome("GRCh38", data="rna")
        for gene in args.g:
            seqs = genome.get_sequences_by_gene(gene)
            if seqs == {}:
                print('failed finding {}'.format(gene))
                logging(args, 'failed finding gene {}'.format(gene))
                continue
            cur_mrna_path = os.path.join('./output/mrna', '%s.fasta' % (gene))
            if not os.path.exists(cur_mrna_path):
                with open(cur_mrna_path, 'a') as f:
                    for n in seqs.keys():
                        f.write(">" + n + "\n")
                        f.write(seqs[n] + "\n")

    read_by_windows()

    for method in args.m:
        if method == '1':
            pred_sirna(args)
        elif method == '2':
            UTR_SEQ = pd.read_csv(os.path.join(PIPELINE_DIR, 'off_target', 'UTR_SEQ.csv'), header=0)
            mirna_like_offtarget(args, UTR_SEQ, None)
        elif method == '3':
            NM_SEQ = pd.read_csv(os.path.join(PIPELINE_DIR, 'off_target', 'NM_SEQ.csv'), header=0)
            homo_substr_offtarget(args, NM_SEQ)
        elif method == '4':
            NM_SEQ = pd.read_csv(os.path.join(PIPELINE_DIR, 'off_target', 'NM_SEQ.csv'), header=0)
            homo_sw_offtarget(args, NM_SEQ)
        else:
            raise ValueError('not supported method {} yet'.format(method))


def __sirnaCM_main():
    _prepare_sirnacm_context()
    for method in args.m:
        if method == '1':
            from potency_prediction.predictor_base0 import pred_sirnacm
            pred_sirnacm(args)
        elif method == '2':
            UTR_SEQ = pd.read_csv(os.path.join(PIPELINE_DIR, 'off_target', 'UTR_SEQ.csv'), header=0)
            mirna_like_offtarget(args, UTR_SEQ, None)
        elif method == '3':
            NM_SEQ = pd.read_csv(os.path.join(PIPELINE_DIR, 'off_target', 'NM_SEQ.csv'), header=0)
            homo_substr_offtarget(args, NM_SEQ)
        elif method == '4':
            NM_SEQ = pd.read_csv(os.path.join(PIPELINE_DIR, 'off_target', 'NM_SEQ.csv'), header=0)
            homo_sw_offtarget(args, NM_SEQ)
        elif method == '5':
            from potency_prediction.design_modified import design_modified_sirnas
            args.sirnacm_path = str(design_modified_sirnas(args))
        elif method == '6':
            from potency_prediction.design_modified import design_modified_sirnas_from_naked
            args.sirnacm_path = str(design_modified_sirnas_from_naked(args))
        else:
            raise ValueError('not supported method {} yet'.format(method))

if __name__ == '__main__':
    if args.filter == 'sirna':
        __sirna_main()
    elif args.filter == 'sirnacm':
        __sirnaCM_main()
