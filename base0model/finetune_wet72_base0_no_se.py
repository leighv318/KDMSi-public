#!/usr/bin/env python3
"""Finetune BN and the full DNN of Base0_mrna_full_no_se.

For each of the ten pretrained folds, the sequence/modification and mRNA
feature extractors remain frozen in eval mode. BatchNorm and every layer in
the downstream DNN are updated on a fixed wet-lab feedback set. The script
reports zero-shot and finetuned 10-fold ensemble metrics on the same remaining
holdout records. It supports both the original AGT input preparation and
already prepared workflow inputs with local-mRNA context.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LIUBIN_PIPELINE_DIR = PROJECT_DIR / "liubin_pipeline"
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(LIUBIN_PIPELINE_DIR))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

import load_data as LD
import model as base0_models
from compute_mrna_thermo import compute_accessibility, compute_local_fold_dg
from mrna_features import candidate_queries, find_all, prepare_clean_sequence


THERMO_COLS = list(LD.MRNA_THERMO_COLS)
MODEL_INPUT_KEY = [
    "ori_as",
    "ori_ss",
    "cm_as",
    "cm_ss",
    "cell_type_idx",
    "concentration/nM",
]
SPLIT_GROUP_KEY = ["ori_as", "ori_ss", "cm_as", "cm_ss"]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def resolve_glob(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else SCRIPT_DIR / path)


def portable_path(value) -> str:
    """Render output provenance without exposing a workstation/server path."""
    path = Path(value).expanduser()
    try:
        return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def portable_argument_path(value: str) -> str:
    path = Path(value).expanduser()
    return path.name if path.is_absolute() else str(path)


def read_fasta_records(path: Path):
    records = []
    name = None
    lines = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequence = re.sub(r"[^ACGTUN]", "", "".join(lines).upper()).replace("U", "T")
                    if sequence:
                        records.append({"name": name, "sequence": sequence})
                name = line[1:].split()[0]
                lines = []
            else:
                lines.append(line)
    if name is not None:
        sequence = re.sub(r"[^ACGTUN]", "", "".join(lines).upper()).replace("U", "T")
        if sequence:
            records.append({"name": name, "sequence": sequence})
    if not records:
        raise ValueError(f"No nucleotide sequence found in FASTA: {path}")
    return records


def choose_match(matches, preferred_accession):
    if not matches:
        return None
    preferred_accession = str(preferred_accession or "").strip()
    if preferred_accession:
        preferred = [
            match for match in matches
            if preferred_accession in str(match["record_name"])
        ]
        if preferred:
            return preferred[0]
    unique_record_matches = {}
    for match in matches:
        unique_record_matches.setdefault(match["record_name"], match)
    return next(iter(unique_record_matches.values()))


def add_target_context(df, transcript_records, up, down, min_query_len, preferred_accession=""):
    out = df.copy()
    records = []
    failures = []
    ambiguous_resolved = 0
    for row_index, row in out.iterrows():
        selected = None
        for query_type, query in candidate_queries(
            row.get("ori_ss"), row.get("ori_as"), min_len=min_query_len
        ):
            matches = []
            for record in transcript_records:
                positions = find_all(record["sequence"], query)
                for position in positions:
                    matches.append(
                        {
                            "record_name": record["name"],
                            "transcript": record["sequence"],
                            "position": position,
                        }
                    )
            if matches:
                selected = (query_type, query, matches)
                break

        if selected is None:
            failures.append((row.get("raw_ID", row_index), "no_match"))
            records.append(None)
            continue

        query_type, query, matches = selected
        row_preferred_accession = (
            row.get("resolved_accession", "") or preferred_accession
        )
        match = choose_match(matches, row_preferred_accession)
        if match is None:
            failures.append((row.get("raw_ID", row_index), f"no_selected_match:{query_type}"))
            records.append(None)
            continue
        if len(matches) > 1:
            ambiguous_resolved += 1

        transcript = match["transcript"]
        target_start = match["position"]
        target_end = target_start + len(query)
        local_start = max(target_start - up, 0)
        local_end = min(target_end + down, len(transcript))
        local_mrna = transcript[local_start:local_end].replace("T", "U")
        target_relative_start = target_start - local_start
        acc_mean, acc_seed, acc_min = compute_accessibility(
            local_mrna, target_relative_start, len(query)
        )
        fold_dg, fold_dg_norm = compute_local_fold_dg(local_mrna)
        records.append(
            {
                "resolved_accession": match["record_name"],
                "target_core": query.replace("T", "U"),
                "target_start_0based": target_start,
                "target_end_0based": target_end,
                "local_mrna_start_0based": local_start,
                "local_mrna_end_0based": local_end,
                "local_mrna": local_mrna,
                "local_mrna_len": len(local_mrna),
                "match_status": "unique_match" if len(matches) == 1 else "multi_match_resolved",
                "match_count": len(matches),
                "match_query_type": query_type,
                "acc_target_mean": acc_mean,
                "acc_seed_mean": acc_seed,
                "acc_target_min": acc_min,
                "local_fold_dg": fold_dg,
                "local_fold_dg_norm": fold_dg_norm,
            }
        )

    if failures:
        examples = ", ".join(f"{row_id}({reason})" for row_id, reason in failures[:10])
        raise ValueError(
            f"Could not uniquely map {len(failures)}/{len(out)} wet-lab rows to "
            f"the target FASTA. Examples: {examples}"
        )
    if ambiguous_resolved:
        print(
            f"[target] resolved {ambiguous_resolved} multi-record/multi-position matches "
            f"using preferred accession/first FASTA record",
            flush=True,
        )

    context = pd.DataFrame(records, index=out.index)
    for column in context.columns:
        out[column] = context[column]
    return out


def sequence_digest(sequences) -> str:
    payload = "\n".join(str(value).strip().upper() for value in sequences).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_rnafm_embeddings(
    sequences,
    cache_path,
    max_len,
    batch_size,
    rebuild,
    device,
    model_location=None,
):
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    digest = sequence_digest(sequences)
    expected_shape = (len(sequences), max_len, 640)

    if cache_path.exists() and not rebuild:
        array = np.load(cache_path, mmap_mode="r")
        if array.shape != expected_shape:
            raise ValueError(
                f"RNA-FM cache shape {array.shape} does not match {expected_shape}: "
                f"{cache_path}"
            )
        if not meta_path.exists():
            raise ValueError(f"RNA-FM cache metadata is missing: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("sequence_sha256") != digest:
            raise ValueError(
                "RNA-FM cache sequence/order mismatch. Use --rebuild_rnafm."
            )
        print(f"[RNA-FM] reusing aligned cache: {cache_path}", flush=True)
        return array

    try:
        import fm
    except ImportError as exc:
        raise ImportError(
            "RNA-FM is required to create the wet-lab embedding cache."
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rnafm_model, alphabet = fm.pretrained.rna_fm_t12(model_location)
    converter = alphabet.get_batch_converter()
    rnafm_model = rnafm_model.to(device).eval()
    stack = np.zeros(expected_shape, dtype=np.float32)
    prepared = []
    for row_index, raw in enumerate(sequences):
        normalized = str(raw).strip().upper().replace("T", "U")
        clean, valid_positions = prepare_clean_sequence(normalized, str(row_index))
        if len(normalized) > max_len:
            raise ValueError(
                f"local_mrna row {row_index} has length {len(normalized)} > "
                f"mrna_max_len={max_len}"
            )
        prepared.append((clean, valid_positions))

    print(f"[RNA-FM] generating embeddings for {len(prepared)} rows", flush=True)
    for start in range(0, len(prepared), batch_size):
        end = min(start + batch_size, len(prepared))
        batch = [(str(index), prepared[index][0]) for index in range(start, end)]
        _, _, tokens = converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            encoded = rnafm_model(tokens, repr_layers=[12])["representations"][12]
        encoded = encoded.detach().cpu().numpy()
        for batch_index, row_index in enumerate(range(start, end)):
            clean, valid_positions = prepared[row_index]
            token_embedding = encoded[batch_index, 1 : len(clean) + 1, :]
            stack[row_index, np.asarray(valid_positions, dtype=int), :] = token_embedding
        print(f"[RNA-FM] {end}/{len(prepared)}", flush=True)

    np.save(cache_path, stack)
    meta_path.write_text(
        json.dumps(
            {
                "rows": len(sequences),
                "max_len": max_len,
                "embedding_dim": 640,
                "sequence_sha256": digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del rnafm_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.load(cache_path, mmap_mode="r")


def concentration_nm(df):
    for column in [
        "concentration/nM",
        "concentration_nM",
        "concentration_nm",
        "Concentration_nM",
        "Concentration",
    ]:
        if column in df.columns:
            return (
                pd.to_numeric(df[column], errors="coerce")
                .fillna(10.0)
                .clip(lower=1e-9)
                .to_numpy(dtype=np.float32)
            )
    return np.full(len(df), 10.0, dtype=np.float32)


def labels_from(df):
    for column in [
        "inhibit/0-1",
        "norm_inhibit",
        "observed_inhibition_fraction_unclipped",
    ]:
        if column in df.columns:
            labels = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(labels).sum() < 2:
                raise ValueError(f"Fewer than two finite labels in {column}")
            return labels
    raise KeyError(
        "Wet-lab data must contain inhibit/0-1, norm_inhibit, or "
        "observed_inhibition_fraction_unclipped"
    )


def frame_dict(df, rnafm_indices=None):
    result = {
        "ori_as": df["ori_as"].astype(str).str.strip().to_numpy(),
        "ori_ss": df["ori_ss"].astype(str).str.strip().to_numpy(),
        "cm_as": df["cm_as"].astype(str).str.strip().to_numpy(),
        "cm_ss": df["cm_ss"].astype(str).str.strip().to_numpy(),
        "inhibition": labels_from(df).astype(np.float32),
        "concentration_nM": concentration_nm(df),
    }
    if rnafm_indices is not None:
        result["rnafm_ind"] = np.asarray(rnafm_indices, dtype=int)
    if all(column in df.columns for column in THERMO_COLS):
        result["mrna_thermo"] = (
            df[THERMO_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
    if "cell_type_idx" in df.columns:
        result["cell_type_idx"] = df["cell_type_idx"].to_numpy(dtype=np.int64)
    return result


def prepare_cell_type(df, default_cell_type):
    out = df.copy()
    if "Cell_Type" not in out.columns:
        out["Cell_Type"] = out["cell_type"] if "cell_type" in out.columns else ""
    out["Cell_Type"] = out["Cell_Type"].fillna("").astype(str).str.strip()
    out.loc[out["Cell_Type"] == "", "Cell_Type"] = default_cell_type
    unknown = sorted(set(out["Cell_Type"]) - set(LD.CELL_TYPE2IDX))
    if unknown:
        print(f"[warn] unknown cell types mapped to index 0: {unknown}", flush=True)
    out["cell_type_idx"] = (
        out["Cell_Type"].map(LD.CELL_TYPE2IDX).fillna(0).astype(int)
    )
    return out


def fold_checkpoints(pattern):
    found = {}
    for value in glob.glob(resolve_glob(pattern)):
        path = Path(value)
        fold = None
        for part in path.parts:
            match = re.fullmatch(r"([0-9]+)-inter", part)
            if match:
                fold = int(match.group(1))
                break
        if fold is not None:
            found[fold] = path
    missing = sorted(set(range(10)) - set(found))
    extra = sorted(set(found) - set(range(10)))
    if missing or extra:
        raise FileNotFoundError(
            f"Expected folds 0-9 for {pattern}; missing={missing}, extra={extra}"
        )
    return found


def fold_feature_context(train_df, fold):
    indices = np.arange(len(train_df))
    fold_train = train_df.iloc[indices[indices % 10 != fold]]
    data = frame_dict(fold_train)
    pssm_as = LD.create_pssm(data["ori_as"])
    pssm_ss = LD.create_pssm(data["ori_ss"])
    log_concentration = np.log10(
        np.clip(data["concentration_nM"].astype(float), 1e-9, None)
    )
    mean = float(log_concentration.mean())
    std = float(log_concentration.std())
    return pssm_as, pssm_ss, mean, std if std > 1e-8 else 1.0


def overlap_audit(train_df, wet_df):
    full_columns = ["ori_as", "ori_ss", "cm_as", "cm_ss"]
    duplex_columns = ["ori_as", "ori_ss"]

    def unique_keys(df, columns):
        return {
            tuple(str(row[column]).strip().upper() for column in columns)
            for _, row in df[columns].iterrows()
        }

    train_full = unique_keys(train_df, full_columns)
    train_duplex = unique_keys(train_df, duplex_columns)
    wet_full = [
        tuple(str(row[column]).strip().upper() for column in full_columns)
        for _, row in wet_df[full_columns].iterrows()
    ]
    wet_duplex = [
        tuple(str(row[column]).strip().upper() for column in duplex_columns)
        for _, row in wet_df[duplex_columns].iterrows()
    ]
    return {
        "wet_rows": len(wet_df),
        "exact_sequence_and_modification_overlap_rows": int(
            sum(key in train_full for key in wet_full)
        ),
        "same_unmodified_duplex_overlap_rows": int(
            sum(key in train_duplex for key in wet_duplex)
        ),
    }


def unwrap_state(state):
    if isinstance(state, dict) and "state_dict" in state:
        if isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def load_base0(args, checkpoint, device):
    model_class = getattr(base0_models, args.base0_model)
    model = model_class(
        device=device,
        dp=0.1,
        as_ts=args.as_ts,
        ss_ts=args.ss_ts,
        cell_embed_dim=args.cell_embed_dim,
    ).float().to(device)
    state = unwrap_state(torch.load(str(checkpoint), map_location="cpu"))
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    values = []
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    value = value.to(device)
                    if torch.is_floating_point(value):
                        value = value.float()
                    batch[key] = value
            values.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(values)


def metrics(labels, predictions, auc_threshold):
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    finite = np.isfinite(labels) & np.isfinite(predictions)
    labels = labels[finite]
    predictions = predictions[finite]
    result = {"n": int(len(labels))}
    result["pcc"] = float(pearsonr(labels, predictions)[0]) if len(labels) > 1 else None
    result["spcc"] = (
        float(spearmanr(labels, predictions)[0]) if len(labels) > 1 else None
    )
    result["rmse"] = float(np.sqrt(mean_squared_error(labels, predictions)))
    result["mae"] = float(mean_absolute_error(labels, predictions))
    binary = (labels >= auc_threshold).astype(int)
    result["auc"] = (
        float(roc_auc_score(binary, predictions))
        if len(np.unique(binary)) == 2
        else None
    )
    return result


def print_metrics(name, values):
    rendered = ", ".join(
        f"{key}={value:.4f}"
        for key, value in values.items()
        if key != "n" and value is not None
    )
    print(f"[{name}] n={values['n']}, {rendered}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Finetune BatchNorm and the full DNN of "
            "Base0_mrna_full_no_se on a fixed wet-lab feedback subset"
        )
    )
    parser.add_argument(
        "--wet_data",
        required=True,
        help="Complete wet-lab CSV used to construct finetune and holdout sets",
    )
    parser.add_argument(
        "--split_strategy",
        choices=["group_manifest", "group_fraction", "manifest"],
        default="group_manifest",
        help=(
            "Use the released sequence/modification group manifest by default; "
            "group_fraction regenerates it from the fixed seed, while manifest "
            "retains support for the earlier row-ID/tag format"
        ),
    )
    parser.add_argument("--finetune_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--split_seed", type=int, default=126)
    parser.add_argument("--strict_no_split_leakage", action="store_true", default=True)
    parser.add_argument(
        "--split_manifest",
        default="agt_adaptation/split_manifest.csv",
        help="Fixed group-level split manifest used with --split_strategy group_manifest",
    )
    parser.add_argument(
        "--finetune_manifest",
        default="",
        help="Optional manifest CSV used only when --split_strategy manifest",
    )
    parser.add_argument("--manifest_tag_col", default="gene_rescue_note")
    parser.add_argument("--manifest_tag", default="external_AGT72_finetune_candidate")
    parser.add_argument("--manifest_id_col", default="ID")
    parser.add_argument("--wet_id_col", default="raw_ID")
    parser.add_argument("--expected_finetune_rows", type=int, default=69)
    parser.add_argument("--expected_holdout_rows", type=int, default=138)
    parser.add_argument(
        "--target_fasta",
        default="",
        help="Target FASTA; required unless --use_existing_target_context is set",
    )
    parser.add_argument(
        "--use_existing_target_context",
        action="store_true",
        help=(
            "Reuse local_mrna and thermodynamic/accessibility columns already "
            "present in --wet_data instead of remapping against a target FASTA"
        ),
    )
    parser.add_argument(
        "--preferred_accession",
        default="",
        help="Preferred transcript accession when a target site maps to multiple FASTA records",
    )
    parser.add_argument(
        "--train_data",
        default="data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv",
    )
    parser.add_argument(
        "--base0_ckpt_glob",
        default=(
            "../liubin_pipeline/potency_prediction/pretrain_models/"
            "Base0_mrna_full_no_se/*-inter/model_best.pth.tar"
        ),
    )
    parser.add_argument(
        "--base0_model",
        default="Base0_mrna_full_no_se",
        choices=["Base0_mrna_full_no_se"],
    )
    parser.add_argument(
        "--rnafm_cache",
        default="agt_adaptation/cache/agt_207_rnafm_maxlen65.npy",
    )
    parser.add_argument(
        "--rnafm_model_location",
        default="",
        help="Optional local RNA-FM checkpoint path used when rebuilding the cache",
    )
    parser.add_argument("--rebuild_rnafm", action="store_true")
    parser.add_argument("--rnafm_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--pearson_weight", type=float, default=0.5)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument(
        "--folds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated pretrained fold indices",
    )
    parser.add_argument("--base0_max_len", type=int, default=25)
    parser.add_argument("--mrna_max_len", type=int, default=65)
    parser.add_argument("--window_up", type=int, default=20)
    parser.add_argument("--window_down", type=int, default=20)
    parser.add_argument("--min_query_len", type=int, default=16)
    parser.add_argument("--default_cell_type", default="HepG2")
    parser.add_argument("--holdout_exclude_source", default="siAGT_invivo")
    parser.add_argument("--as_ts", type=int, default=3)
    parser.add_argument("--ss_ts", type=int, default=3)
    parser.add_argument("--cell_embed_dim", type=int, default=8)
    parser.add_argument("--auc_threshold", type=float, default=0.7)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--no_save_checkpoints",
        action="store_true",
        help="Do not save each fold/seed finetuned checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="agt_adaptation/run_output",
    )
    args = parser.parse_args()
    args.seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not args.seeds:
        parser.error("--seeds must contain at least one integer")
    args.folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    if not args.folds or len(args.folds) != len(set(args.folds)):
        parser.error("--folds must contain unique fold indices")
    invalid_folds = sorted(set(args.folds) - set(range(10)))
    if invalid_folds:
        parser.error(f"--folds contains invalid indices: {invalid_folds}")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_fixed_split(wet: pd.DataFrame, manifest: pd.DataFrame, args):
    required = [args.manifest_tag_col, args.manifest_id_col]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise KeyError(f"Finetune manifest is missing columns: {missing}")
    if args.wet_id_col not in wet.columns:
        raise KeyError(f"Wet data is missing ID column: {args.wet_id_col}")

    tagged = manifest[
        manifest[args.manifest_tag_col]
        .fillna("")
        .astype(str)
        .str.contains(args.manifest_tag, regex=False)
    ].copy()
    ids = tagged[args.manifest_id_col].fillna("").astype(str).str.strip()
    if len(tagged) != args.expected_finetune_rows or ids.nunique() != len(tagged):
        raise ValueError(
            f"Expected {args.expected_finetune_rows} unique manifest rows, found "
            f"rows={len(tagged)}, unique_ids={ids.nunique()}"
        )

    wet_ids = wet[args.wet_id_col].fillna("").astype(str).str.strip()
    missing_ids = sorted(set(ids) - set(wet_ids))
    duplicate_ids = wet_ids[wet_ids.duplicated(keep=False)].unique().tolist()
    if missing_ids:
        raise ValueError(f"Manifest IDs missing from wet data: {missing_ids[:10]}")
    if duplicate_ids:
        raise ValueError(f"Wet IDs are not unique: {duplicate_ids[:10]}")

    finetune_mask = wet_ids.isin(set(ids))
    finetune = wet.loc[finetune_mask].copy()
    holdout = wet.loc[~finetune_mask].copy()
    if len(finetune) != args.expected_finetune_rows:
        raise ValueError(
            f"Expected {args.expected_finetune_rows} finetune rows after matching, "
            f"found {len(finetune)}"
        )
    return finetune, holdout, tagged


def normalized_key(row, columns):
    return tuple(str(row[column]).strip().upper() for column in columns)


def build_group_fraction_split(wet: pd.DataFrame, args):
    missing = [column for column in SPLIT_GROUP_KEY if column not in wet.columns]
    if missing:
        raise KeyError(f"Wet data is missing split group columns: {missing}")
    if not (0 < args.finetune_fraction < 1):
        raise ValueError("--finetune_fraction must be between 0 and 1")

    groups = []
    for key, indices in wet.groupby(SPLIT_GROUP_KEY, sort=False, dropna=False).groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        groups.append({"key": key, "indices": index_array, "size": len(index_array)})

    target_rows = int(round(len(wet) * args.finetune_fraction))
    rng = np.random.default_rng(args.split_seed)
    order = rng.permutation(len(groups))
    selected = []
    selected_rows = 0
    for group_index in order:
        group = groups[int(group_index)]
        if not selected:
            selected.append(group_index)
            selected_rows += group["size"]
            continue
        current_gap = abs(selected_rows - target_rows)
        new_gap = abs((selected_rows + group["size"]) - target_rows)
        if selected_rows < target_rows or new_gap < current_gap:
            selected.append(group_index)
            selected_rows += group["size"]
        if selected_rows >= target_rows and selected_rows >= 1:
            remaining_sizes = [
                groups[int(other)]["size"]
                for other in order
                if int(other) not in set(map(int, selected))
            ]
            if not remaining_sizes or min(remaining_sizes) >= abs(selected_rows - target_rows):
                break

    selected = set(map(int, selected))
    finetune_indices = np.concatenate(
        [groups[index]["indices"] for index in sorted(selected)]
    )
    finetune_mask = wet.index.isin(finetune_indices)
    finetune = wet.loc[finetune_mask].copy()
    holdout = wet.loc[~finetune_mask].copy()
    if finetune.empty or holdout.empty:
        raise ValueError(
            f"Invalid split: finetune={len(finetune)}, holdout={len(holdout)}"
        )
    if args.expected_finetune_rows > 0 and len(finetune) != args.expected_finetune_rows:
        raise ValueError(
            f"Expected {args.expected_finetune_rows} finetune rows, found {len(finetune)}"
        )
    if args.expected_holdout_rows > 0 and len(holdout) != args.expected_holdout_rows:
        raise ValueError(
            f"Expected {args.expected_holdout_rows} holdout rows, found {len(holdout)}"
        )

    manifest = pd.DataFrame(
        [
            {
                "split_group_id": index,
                "split": "finetune" if index in selected else "holdout",
                "group_size": group["size"],
                **{column: group["key"][i] for i, column in enumerate(SPLIT_GROUP_KEY)},
            }
            for index, group in enumerate(groups)
        ]
    )
    return finetune, holdout, manifest


def build_group_manifest_split(wet: pd.DataFrame, manifest: pd.DataFrame, args):
    """Apply a released split keyed by complete sequence/modification identity."""
    required = ["split", *SPLIT_GROUP_KEY]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise KeyError(f"Group split manifest is missing columns: {missing}")
    wet_missing = [column for column in SPLIT_GROUP_KEY if column not in wet.columns]
    if wet_missing:
        raise KeyError(f"Wet data is missing split group columns: {wet_missing}")

    assignments = {}
    declared_sizes = {}
    for _, row in manifest.iterrows():
        key = normalized_key(row, SPLIT_GROUP_KEY)
        split = str(row["split"]).strip().lower()
        if split not in {"finetune", "holdout"}:
            raise ValueError(f"Unsupported split label {row['split']!r} for group {key}")
        if key in assignments:
            raise ValueError(f"Duplicate sequence/modification group in manifest: {key}")
        assignments[key] = split
        if "group_size" in manifest.columns:
            declared_sizes[key] = int(row["group_size"])

    wet_keys = [normalized_key(row, SPLIT_GROUP_KEY) for _, row in wet.iterrows()]
    wet_key_set = set(wet_keys)
    missing_groups = wet_key_set - set(assignments)
    extra_groups = set(assignments) - wet_key_set
    if missing_groups or extra_groups:
        raise ValueError(
            "Group split manifest does not exactly cover the wet-lab data: "
            f"missing={len(missing_groups)}, extra={len(extra_groups)}"
        )

    if declared_sizes:
        observed_sizes = pd.Series(wet_keys).value_counts().to_dict()
        mismatches = {
            key: (declared_sizes[key], observed_sizes.get(key, 0))
            for key in assignments
            if declared_sizes[key] != observed_sizes.get(key, 0)
        }
        if mismatches:
            examples = list(mismatches.items())[:5]
            raise ValueError(f"Group-size mismatch in split manifest: {examples}")

    finetune_mask = np.asarray(
        [assignments[key] == "finetune" for key in wet_keys], dtype=bool
    )
    finetune = wet.loc[finetune_mask].copy()
    holdout = wet.loc[~finetune_mask].copy()
    if args.expected_finetune_rows > 0 and len(finetune) != args.expected_finetune_rows:
        raise ValueError(
            f"Expected {args.expected_finetune_rows} finetune rows, found {len(finetune)}"
        )
    if args.expected_holdout_rows > 0 and len(holdout) != args.expected_holdout_rows:
        raise ValueError(
            f"Expected {args.expected_holdout_rows} holdout rows, found {len(holdout)}"
        )
    return finetune, holdout, manifest.copy()


def split_group_overlap_audit(finetune: pd.DataFrame, holdout: pd.DataFrame):
    missing = [
        column
        for column in SPLIT_GROUP_KEY
        if column not in finetune.columns or column not in holdout.columns
    ]
    if missing:
        return {"checked": False, "missing_columns": missing}
    finetune_keys = {
        normalized_key(row, SPLIT_GROUP_KEY) for _, row in finetune.iterrows()
    }
    holdout_keys = {
        normalized_key(row, SPLIT_GROUP_KEY) for _, row in holdout.iterrows()
    }
    overlap = finetune_keys & holdout_keys
    duplex_columns = ["ori_as", "ori_ss"]
    finetune_duplexes = {
        normalized_key(row, duplex_columns) for _, row in finetune.iterrows()
    }
    holdout_duplexes = {
        normalized_key(row, duplex_columns) for _, row in holdout.iterrows()
    }
    duplex_overlap = finetune_duplexes & holdout_duplexes
    return {
        "checked": True,
        "group_columns": SPLIT_GROUP_KEY,
        "overlap_unique_sequence_modification_inputs": len(overlap),
        "finetune_unique_sequence_modification_inputs": len(finetune_keys),
        "holdout_unique_sequence_modification_inputs": len(holdout_keys),
        "overlap_unique_unmodified_duplexes": len(duplex_overlap),
        "finetune_unique_unmodified_duplexes": len(finetune_duplexes),
        "holdout_unique_unmodified_duplexes": len(holdout_duplexes),
    }


def model_input_overlap_audit(finetune: pd.DataFrame, holdout: pd.DataFrame):
    missing = [
        column
        for column in MODEL_INPUT_KEY
        if column not in finetune.columns or column not in holdout.columns
    ]
    if missing:
        return {"checked": False, "missing_columns": missing}

    def row_key(row):
        return tuple(str(row[column]).strip().upper() for column in MODEL_INPUT_KEY)

    finetune_keys = [row_key(row) for _, row in finetune.iterrows()]
    holdout_keys = [row_key(row) for _, row in holdout.iterrows()]
    overlap = set(finetune_keys) & set(holdout_keys)
    return {
        "checked": True,
        "overlap_unique_inputs": len(overlap),
        "finetune_rows_with_holdout_twin": sum(key in overlap for key in finetune_keys),
        "holdout_rows_with_finetune_twin": sum(key in overlap for key in holdout_keys),
    }


def configure_bn_dnn_finetuning(model):
    if not isinstance(model.se_block, nn.Identity):
        raise TypeError(
            "Expected Base0_mrna_full_no_se.se_block to be nn.Identity, "
            f"found {type(model.se_block).__name__}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.bn, model.DNN):
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    unexpected = [
        name
        for name in trainable_names
        if not (name.startswith("bn.") or name.startswith("DNN."))
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")

    trainable_parameters = [value for value in model.parameters() if value.requires_grad]
    return {
        "parameters": trainable_parameters,
        "names": trainable_names,
        "trainable_count": sum(value.numel() for value in trainable_parameters),
        "total_count": sum(value.numel() for value in model.parameters()),
    }


def set_bn_dnn_train_mode(model):
    model.eval()
    model.bn.train()
    model.DNN.train()


def move_batch(batch, device):
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            value = value.to(device)
            if torch.is_floating_point(value):
                value = value.float()
            batch[key] = value
    return batch, batch["inhibit"].float()


def pearson_loss(prediction, label):
    prediction = prediction - prediction.mean()
    label = label - label.mean()
    denominator = prediction.norm() * label.norm()
    return 1.0 - (prediction * label).sum() / (denominator + 1e-8)


def finetune_bn_dnn(model, dataset, args, device, seed):
    set_seed(seed)
    trainable = configure_bn_dnn_finetuning(model)
    optimizer = torch.optim.AdamW(
        trainable["parameters"],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.HuberLoss(reduction="mean")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    history = []
    for _ in range(args.n_epochs):
        set_bn_dnn_train_mode(model)
        epoch_losses = []
        for batch in loader:
            inputs, label = move_batch(batch, device)
            prediction = model(inputs)
            loss = criterion(prediction, label)
            if len(label) > 1 and args.pearson_weight > 0:
                loss = loss + args.pearson_weight * pearson_loss(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable["parameters"], max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)))

    model.eval()
    return {
        "finetuned_modules": ["bn", "DNN"],
        "trainable_parameter_names": trainable["names"],
        "trainable_parameters": trainable["trainable_count"],
        "total_parameters": trainable["total_count"],
        "first_epoch_loss": history[0],
        "final_epoch_loss": history[-1],
        "minimum_training_loss": float(min(history)),
    }


def make_dataset(data_dict, pssm_as, pssm_ss, conc_mean, conc_std, args):
    return LD.Generate_dataset(
        data_dict,
        pssm_as,
        pssm_ss,
        max_len=args.base0_max_len,
        conc_log_mean=conc_mean,
        conc_log_std=conc_std,
    )


def metric_subsets(labels, predictions, holdout: pd.DataFrame, args):
    result = {"all_remaining": metrics(labels, predictions, args.auc_threshold)}
    if args.holdout_exclude_source and "source_dataset" in holdout.columns:
        keep = holdout["source_dataset"].astype(str) != args.holdout_exclude_source
        result["invitro_only"] = metrics(
            np.asarray(labels)[keep.to_numpy()],
            np.asarray(predictions)[keep.to_numpy()],
            args.auc_threshold,
        )
    return result


def source_counts(df: pd.DataFrame):
    if "source_dataset" not in df.columns:
        return {}
    return df["source_dataset"].value_counts().to_dict()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    wet_path = resolve_path(args.wet_data)
    train_path = resolve_path(args.train_data)
    cache_path = resolve_path(args.rnafm_cache)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wet = pd.read_csv(wet_path)
    wet = prepare_cell_type(wet, args.default_cell_type)
    if args.use_existing_target_context:
        required_context = ["local_mrna", *THERMO_COLS]
        missing_context = [
            column for column in required_context if column not in wet.columns
        ]
        if missing_context:
            raise KeyError(
                "Existing target context is missing columns: "
                f"{missing_context}"
            )
        wet["local_mrna"] = wet["local_mrna"].fillna("").astype(str).str.strip()
        if (wet["local_mrna"] == "").any():
            raise ValueError("Empty local_mrna values in existing target context")
        print(
            f"[target] reusing existing target context for {len(wet)} rows",
            flush=True,
        )
    else:
        if not args.target_fasta:
            raise ValueError(
                "--target_fasta is required unless "
                "--use_existing_target_context is set"
            )
        fasta_path = resolve_path(args.target_fasta)
        transcript_records = read_fasta_records(fasta_path)
        print(
            "[target] loaded FASTA records: "
            + ", ".join(
                f"{record['name']}({len(record['sequence'])})"
                for record in transcript_records
            ),
            flush=True,
        )
        wet = add_target_context(
            wet,
            transcript_records,
            args.window_up,
            args.window_down,
            args.min_query_len,
            args.preferred_accession,
        )
    if wet[THERMO_COLS].isna().any().any():
        raise ValueError("Non-finite thermodynamic features remain after computation")

    wet["rnafm_ind"] = np.arange(len(wet), dtype=int)
    wet_rnafm = make_rnafm_embeddings(
        wet["local_mrna"].tolist(),
        cache_path,
        args.mrna_max_len,
        args.rnafm_batch_size,
        args.rebuild_rnafm,
        device,
        (
            str(resolve_path(args.rnafm_model_location))
            if args.rnafm_model_location
            else None
        ),
    )
    if args.split_strategy == "group_manifest":
        manifest_path = resolve_path(args.split_manifest)
        manifest = pd.read_csv(manifest_path, low_memory=False)
        finetune, holdout, split_manifest = build_group_manifest_split(
            wet, manifest, args
        )
    elif args.split_strategy == "manifest":
        if not args.finetune_manifest:
            raise ValueError("--finetune_manifest is required when --split_strategy manifest")
        manifest_path = resolve_path(args.finetune_manifest)
        manifest = pd.read_csv(manifest_path, low_memory=False)
        finetune, holdout, split_manifest = build_fixed_split(wet, manifest, args)
    else:
        finetune, holdout, split_manifest = build_group_fraction_split(wet, args)
    invitro_holdout = holdout
    if args.holdout_exclude_source and "source_dataset" in holdout.columns:
        invitro_holdout = holdout[
            holdout["source_dataset"].astype(str) != args.holdout_exclude_source
        ]

    split_overlap = model_input_overlap_audit(finetune, holdout)
    group_overlap = split_group_overlap_audit(finetune, holdout)
    print(
        f"[split] finetune={len(finetune)}, holdout_all={len(holdout)}, "
        f"holdout_invitro={len(invitro_holdout)}",
        flush=True,
    )
    print(f"[split] sequence/modification group overlap={group_overlap}", flush=True)
    print(f"[split] full model-input overlap={split_overlap}", flush=True)
    if group_overlap.get("overlap_unique_sequence_modification_inputs", 0):
        message = (
            "Sequence/modification inputs occur in both finetune and holdout sets; "
            "this would leak information."
        )
        if args.strict_no_split_leakage:
            raise ValueError(message)
        print(f"[warn] {message}", flush=True)
    elif split_overlap.get("overlap_unique_inputs", 0):
        print("[warn] Full model-visible inputs overlap despite group split.", flush=True)

    wet.to_csv(output_dir / "wet_data_with_mrna_features.csv", index=False)
    finetune.to_csv(output_dir / "finetune_feedback.csv", index=False)
    holdout.to_csv(output_dir / "holdout_remaining.csv", index=False)
    invitro_holdout.to_csv(output_dir / "holdout_invitro_only.csv", index=False)
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    train = LD._load_data(str(train_path))
    if len(train) != 22622:
        raise ValueError(f"Expected 22,622 training rows, found {len(train)}")
    LD.MRNA_RNAFM = wet_rnafm
    LD.MRNA_MAX_LEN = args.mrna_max_len

    finetune_indices = finetune.index.to_numpy(dtype=int)
    holdout_indices = holdout.index.to_numpy(dtype=int)
    finetune_dict = frame_dict(finetune, rnafm_indices=finetune_indices)
    holdout_dict = frame_dict(holdout, rnafm_indices=holdout_indices)
    holdout_labels = labels_from(holdout)
    checkpoints = fold_checkpoints(args.base0_ckpt_glob)

    summary = {
        "configuration": {
            "model": args.base0_model,
            "checkpoints": portable_argument_path(args.base0_ckpt_glob),
            "seeds": args.seeds,
            "folds": args.folds,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pearson_weight": args.pearson_weight,
            "split_strategy": args.split_strategy,
            "finetune_fraction": args.finetune_fraction,
            "split_seed": args.split_seed,
            "split_group_columns": SPLIT_GROUP_KEY,
            "finetuned_modules": ["BatchNorm", "full DNN"],
            "frozen_modules": "all remaining Base0 modules",
            "holdout_used_for_training_or_selection": False,
            "used_existing_target_context": args.use_existing_target_context,
            "target_fasta": (
                portable_path(args.target_fasta) if args.target_fasta else ""
            ),
            "preferred_accession": args.preferred_accession,
            "rnafm_model_location": (
                portable_path(args.rnafm_model_location)
                if args.rnafm_model_location
                else ""
            ),
            "saved_finetuned_checkpoints": not args.no_save_checkpoints,
        },
        "split_audit": {
            "wet_rows": len(wet),
            "finetune_rows": len(finetune),
            "holdout_all_rows": len(holdout),
            "holdout_invitro_rows": len(invitro_holdout),
            "finetune_sources": source_counts(finetune),
            "holdout_sources": source_counts(holdout),
            "sequence_modification_group_overlap": group_overlap,
            "model_input_overlap": split_overlap,
            "pretraining_overlap": overlap_audit(train, wet),
        },
        "folds": {},
        "seeds": {},
    }
    output_predictions = holdout.copy()
    zero_fold_predictions = []
    seed_fold_predictions = {seed: [] for seed in args.seeds}

    print("\n================ BASE0_MRNA_FULL_NO_SE ================", flush=True)
    for fold in args.folds:
        print(f"[fold {fold}] checkpoint={checkpoints[fold]}", flush=True)
        pssm_as, pssm_ss, conc_mean, conc_std = fold_feature_context(train, fold)
        finetune_set = make_dataset(
            finetune_dict, pssm_as, pssm_ss, conc_mean, conc_std, args
        )
        holdout_set = make_dataset(
            holdout_dict, pssm_as, pssm_ss, conc_mean, conc_std, args
        )
        fold_summary = {"checkpoint": portable_path(checkpoints[fold]), "seeds": {}}

        for seed_index, seed in enumerate(args.seeds):
            model = load_base0(args, checkpoints[fold], device)
            if seed_index == 0:
                zero_fold_predictions.append(
                    predict(model, holdout_set, args.batch_size, device)
                )
            training = finetune_bn_dnn(
                model, finetune_set, args, device, seed
            )
            print(
                f"[fold {fold} seed {seed}] trainable="
                f"{training['trainable_parameters']:,}/"
                f"{training['total_parameters']:,}",
                flush=True,
            )
            holdout_prediction = predict(
                model, holdout_set, args.batch_size, device
            )
            seed_fold_predictions[seed].append(holdout_prediction)
            fold_summary["seeds"][str(seed)] = {
                "training": training,
                "holdout": metric_subsets(
                    holdout_labels, holdout_prediction, holdout, args
                ),
            }

            if not args.no_save_checkpoints:
                save_dir = output_dir / "checkpoints" / f"fold{fold}" / f"seed{seed}"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    save_dir / "model_bn_dnn_finetuned.pth.tar",
                )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary["folds"][str(fold)] = fold_summary
        del finetune_set, holdout_set

    zero_fold_predictions = np.stack(zero_fold_predictions, axis=0)
    zero_mean = zero_fold_predictions.mean(axis=0)
    summary["zero_shot_10fold_ensemble"] = metric_subsets(
        holdout_labels, zero_mean, holdout, args
    )
    output_predictions["base0_zeroshot_mean"] = zero_mean
    output_predictions["base0_zeroshot_fold_std"] = zero_fold_predictions.std(axis=0)
    print_metrics(
        "Base0 zero-shot holdout all",
        summary["zero_shot_10fold_ensemble"]["all_remaining"],
    )
    if "invitro_only" in summary["zero_shot_10fold_ensemble"]:
        print_metrics(
            "Base0 zero-shot holdout in-vitro",
            summary["zero_shot_10fold_ensemble"]["invitro_only"],
        )

    seed_ensemble_predictions = []
    for seed in args.seeds:
        fold_values = np.stack(seed_fold_predictions[seed], axis=0)
        ensemble = fold_values.mean(axis=0)
        seed_ensemble_predictions.append(ensemble)
        seed_metrics = metric_subsets(holdout_labels, ensemble, holdout, args)
        summary["seeds"][str(seed)] = {"holdout": seed_metrics}
        output_predictions[f"base0_finetuned_seed{seed}_mean"] = ensemble
        output_predictions[f"base0_finetuned_seed{seed}_fold_std"] = fold_values.std(axis=0)
        print_metrics(
            f"Base0 finetuned seed{seed} holdout all",
            seed_metrics["all_remaining"],
        )

    seed_ensemble_predictions = np.stack(seed_ensemble_predictions, axis=0)
    final_mean = seed_ensemble_predictions.mean(axis=0)
    summary["finetuned_10fold_seed_mean"] = metric_subsets(
        holdout_labels, final_mean, holdout, args
    )
    output_predictions["base0_finetuned_mean"] = final_mean
    output_predictions["base0_finetuned_seed_std"] = seed_ensemble_predictions.std(axis=0)
    print_metrics(
        "Base0 FINAL finetuned holdout all",
        summary["finetuned_10fold_seed_mean"]["all_remaining"],
    )
    if "invitro_only" in summary["finetuned_10fold_seed_mean"]:
        print_metrics(
            "Base0 FINAL finetuned holdout in-vitro",
            summary["finetuned_10fold_seed_mean"]["invitro_only"],
        )

    output_predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    if args.holdout_exclude_source and "source_dataset" in output_predictions.columns:
        output_predictions[
            output_predictions["source_dataset"].astype(str)
            != args.holdout_exclude_source
        ].to_csv(output_dir / "holdout_predictions_invitro_only.csv", index=False)
    (output_dir / "finetune_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\n[saved] {output_dir / 'finetune_metrics.json'}", flush=True)
    print(f"[saved] {output_dir / 'holdout_predictions.csv'}", flush=True)


if __name__ == "__main__":
    main()
