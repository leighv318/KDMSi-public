"""Final KDMSi-Base0 inference backend for chemically modified siRNAs.

Default behavior:
    Base0_mrna_full_no_se, averaged over 0-inter ... 9-inter checkpoints
    under potency_prediction/pretrain_models/Base0_mrna_full_no_se.

The input CSV must contain ori_as, ori_ss, cm_as, and cm_ss. It may either
contain local_mrna directly or be paired with --base0_target_fasta so this
module can locate the target and construct the local mRNA context.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT_DIR / "liubin_pipeline"
BASE0_DIR = ROOT_DIR / "base0model"
DEFAULT_MODEL = "Base0_mrna_full_no_se"
DEFAULT_CKPT_DIR = PIPELINE_DIR / "potency_prediction" / "pretrain_models" / DEFAULT_MODEL
DEFAULT_TRAIN_DATA = BASE0_DIR / "data" / "CMsiRNAdb_Dsicm10_FINAL_trainready.csv"
REQUIRED_COLUMNS = ["ori_as", "ori_ss", "cm_as", "cm_ss"]

for path in (ROOT_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from base0model import load_data as base0_data
from base0model import model as base0_models
from base0model.compute_mrna_thermo import compute_accessibility, compute_local_fold_dg
from base0model.mrna_features import candidate_queries, find_all, prepare_clean_sequence


THERMO_COLUMNS = list(base0_data.MRNA_THERMO_COLS)


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clean_rna(value):
    return re.sub(r"[^ACGTUN]", "", str(value).strip().upper()).replace("T", "U")


def _parse_folds(value):
    if value is None or not str(value).strip():
        return [f"{i}-inter" for i in range(10)]
    folds = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            folds.append(item if item.endswith("-inter") else f"{item}-inter")
    return folds


def _checkpoint_for_fold(ckpt_dir, model_name, fold):
    ckpt_dir = Path(ckpt_dir)
    candidates = [
        ckpt_dir / fold / "model_best.pth.tar",
        ckpt_dir / fold / f"model_{fold}.pth.tar",
        ckpt_dir / f"{model_name}_{fold}.pth.tar",
        ckpt_dir / f"model_{fold}.pth.tar",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No Base0 checkpoint found for {fold}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _resolve_sirnacm_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PIPELINE_DIR / "output" / "sirnacm" / path).resolve()


def _resolve_optional_path(value):
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, PIPELINE_DIR / path, ROOT_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _read_fasta_records(path):
    records = []
    name = None
    pieces = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, _clean_rna("".join(pieces)).replace("U", "T")))
                name = line[1:].strip() or f"record_{len(records)}"
                pieces = []
            else:
                pieces.append(line)
    if name is not None:
        records.append((name, _clean_rna("".join(pieces)).replace("U", "T")))
    if not records:
        raise ValueError(f"No FASTA records found: {path}")
    return records


def _match_candidates(row, records, min_query_len):
    drug_id = row.get("raw_ID", row.name)
    for query_type, query in candidate_queries(
            row.get("ori_ss"), row.get("ori_as"), min_len=min_query_len):
        matches = []
        for record_name, transcript in records:
            for start in find_all(transcript, query):
                matches.append((record_name, transcript, start, query_type, query))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Multiple isoforms are acceptable only when they produce exactly the
            # same local context; otherwise the transcript choice is ambiguous.
            windows = set()
            for _, transcript, start, _, matched_query in matches:
                left = max(0, start - 20)
                right = min(len(transcript), start + len(matched_query) + 20)
                windows.add(transcript[left:right])
            if len(windows) == 1:
                return matches[0]
    raise ValueError(
        f"Could not uniquely map candidate {drug_id!r} to the supplied target FASTA. "
        "Use a single intended transcript FASTA or provide local_mrna in the CSV."
    )


def _thermo_for_existing_window(row, local_mrna, min_query_len):
    dna = local_mrna.replace("U", "T")
    for _, query in candidate_queries(
            row.get("ori_ss"), row.get("ori_as"), min_len=min_query_len):
        positions = find_all(dna, query)
        if len(positions) == 1:
            start = positions[0]
            acc = compute_accessibility(local_mrna, start, len(query))
            fold = compute_local_fold_dg(local_mrna)
            return query, start, acc, fold
    raise ValueError(
        f"Could not locate the target sequence inside local_mrna for row "
        f"{row.get('raw_ID', row.name)!r}."
    )


def _prepare_target_context(df, target_fasta, window_up, window_down, min_query_len):
    out = df.copy()
    has_local = "local_mrna" in out.columns and out["local_mrna"].notna().all()
    records = None
    if not has_local:
        if target_fasta is None:
            raise ValueError(
                "Base0_mrna_full_no_se requires mRNA context. Provide "
                "--base0_target_fasta or add local_mrna to the candidate CSV."
            )
        records = _read_fasta_records(target_fasta)

    prepared = []
    context_cache = {}
    for _, row in out.iterrows():
        cache_key = (
            str(row.get("ori_as", "")).strip(),
            str(row.get("ori_ss", "")).strip(),
            str(row.get("local_mrna", "")).strip() if has_local else "",
        )
        if cache_key in context_cache:
            prepared.append(context_cache[cache_key].copy())
            continue
        if has_local:
            local_mrna = _clean_rna(row["local_mrna"])
            query, target_rel_start, acc, fold = _thermo_for_existing_window(
                row, local_mrna, min_query_len
            )
            record_name = str(row.get("resolved_accession", "provided_local_mrna"))
            target_start = row.get("target_start_0based", np.nan)
            local_start = row.get("local_mrna_start_0based", np.nan)
        else:
            record_name, transcript, target_start, query_type, query = _match_candidates(
                row, records, min_query_len
            )
            target_end = target_start + len(query)
            local_start = max(target_start - window_up, 0)
            local_end = min(target_end + window_down, len(transcript))
            local_mrna = transcript[local_start:local_end].replace("T", "U")
            target_rel_start = target_start - local_start
            acc = compute_accessibility(local_mrna, target_rel_start, len(query))
            fold = compute_local_fold_dg(local_mrna)

        if len(local_mrna) > 65:
            raise ValueError(
                f"local_mrna length {len(local_mrna)} exceeds final model limit 65"
            )
        values = list(acc) + list(fold)
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(
                f"Non-finite mRNA biophysical features for "
                f"{row.get('raw_ID', row.name)!r}"
            )
        context = {
            "resolved_accession": record_name,
            "target_core": query.replace("T", "U"),
            "target_start_0based": target_start,
            "local_mrna_start_0based": local_start,
            "target_rel_start": target_rel_start,
            "local_mrna": local_mrna,
            "local_mrna_len": len(local_mrna),
            "acc_target_mean": acc[0],
            "acc_seed_mean": acc[1],
            "acc_target_min": acc[2],
            "local_fold_dg": fold[0],
            "local_fold_dg_norm": fold[1],
        }
        context_cache[cache_key] = context
        prepared.append(context.copy())

    prepared = pd.DataFrame(prepared, index=out.index)
    for column in prepared.columns:
        out[column] = prepared[column]
    return out


def _sequence_digest(sequences):
    payload = "\n".join(_clean_rna(seq) for seq in sequences).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rnafm_embeddings(sequences, cache_path, max_len, batch_size, rebuild, device):
    sequences = [_clean_rna(seq) for seq in sequences]
    digest = _sequence_digest(sequences)
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")

    if cache_path.exists() and not rebuild:
        array = np.load(cache_path, mmap_mode="r")
        if array.shape != (len(sequences), max_len, 640):
            raise ValueError(
                f"RNA-FM cache shape mismatch: {array.shape}; use --base0_rebuild_rnafm"
            )
        if not meta_path.exists():
            raise ValueError(f"RNA-FM metadata missing: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("sequence_sha256") != digest:
            raise ValueError(
                "RNA-FM cache does not match candidate mRNA order; "
                "use --base0_rebuild_rnafm"
            )
        print(f"[Base0] reusing RNA-FM cache: {cache_path}")
        return array

    import fm

    print(f"[Base0] loading RNA-FM for {len(sequences)} unique mRNA windows...")
    rnafm, alphabet = fm.pretrained.rna_fm_t12()
    converter = alphabet.get_batch_converter()
    rnafm = rnafm.to(device).eval()
    result = np.zeros((len(sequences), max_len, 640), dtype=np.float32)
    prepared = []
    for i, sequence in enumerate(sequences):
        if len(sequence) > max_len:
            raise ValueError(f"mRNA window {i} exceeds max_len={max_len}")
        clean, valid_positions = prepare_clean_sequence(sequence, str(i))
        prepared.append((clean, valid_positions))

    for start in range(0, len(prepared), batch_size):
        end = min(start + batch_size, len(prepared))
        batch = [(str(i), prepared[i][0]) for i in range(start, end)]
        _, _, tokens = converter(batch)
        with torch.no_grad():
            encoded = rnafm(tokens.to(device), repr_layers=[12])["representations"][12]
        encoded = encoded.detach().cpu().numpy()
        for j, i in enumerate(range(start, end)):
            clean, valid_positions = prepared[i]
            result[i, np.asarray(valid_positions, dtype=int), :] = encoded[
                j, 1:len(clean) + 1, :
            ]
        print(f"[Base0] RNA-FM {end}/{len(prepared)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result)
    meta_path.write_text(json.dumps({
        "rows": len(sequences),
        "max_len": max_len,
        "embedding_dim": 640,
        "sequence_sha256": digest,
    }, indent=2), encoding="utf-8")
    del rnafm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.load(cache_path, mmap_mode="r")


def _concentration_nm(df, default_value):
    for column in ["concentration/nM", "concentration_nM", "concentration_nm",
                   "Concentration_nM", "Concentration"]:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(
                default_value
            ).clip(lower=1e-9).to_numpy(dtype=np.float32)
    return np.full(len(df), float(default_value), dtype=np.float32)


def _cell_type_indices(df, default_cell_type):
    values = None
    for column in ["Cell_Type", "cell_type"]:
        if column in df.columns:
            values = df[column].fillna("").astype(str).str.strip()
            break
    if values is None:
        values = pd.Series([""] * len(df), index=df.index)
    if default_cell_type:
        values = values.mask(values.isin(["", "NA", "nan", "None"]), default_cell_type)
    return values.map(base0_data.CELL_TYPE2IDX).fillna(0).to_numpy(dtype=np.int64)


def _frame_dict(df, rnafm_indices, default_concentration, default_cell_type):
    def seq_values(column):
        return df[column].astype(str).str.strip().str.replace("T", "t", regex=False).to_numpy()

    return {
        "ori_as": seq_values("ori_as"),
        "ori_ss": seq_values("ori_ss"),
        "cm_as": seq_values("cm_as"),
        "cm_ss": seq_values("cm_ss"),
        "inhibition": np.zeros(len(df), dtype=np.float32),
        "concentration_nM": _concentration_nm(df, default_concentration),
        "rnafm_ind": np.asarray(rnafm_indices, dtype=int),
        "mrna_thermo": df[THERMO_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32),
        "cell_type_idx": _cell_type_indices(df, default_cell_type),
    }


def _fold_feature_context(train_df, fold):
    fold_number = int(fold.split("-")[0])
    indices = np.arange(len(train_df))
    fold_train = train_df.iloc[indices[indices % 10 != fold_number]]
    pssm_as = base0_data.create_pssm(np.asarray(fold_train["ori_as"]))
    pssm_ss = base0_data.create_pssm(np.asarray(fold_train["ori_ss"]))
    concentrations = _concentration_nm(fold_train, 10.0).astype(float)
    log_conc = np.log10(np.clip(concentrations, 1e-9, None))
    mean = float(log_conc.mean())
    std = float(log_conc.std())
    return pssm_as, pssm_ss, mean, std if std > 1e-8 else 1.0


def _unwrap_state(state):
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def _load_model(model_name, checkpoint, device, as_ts, ss_ts, cell_embed_dim):
    cls = getattr(base0_models, model_name)
    model = cls(
        device=device, dp=0.1, as_ts=as_ts, ss_ts=ss_ts,
        cell_embed_dim=cell_embed_dim,
    ).float().to(device)
    state = _unwrap_state(torch.load(str(checkpoint), map_location="cpu"))
    model.load_state_dict(state, strict=True)
    return model.eval()


def _predict(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    if torch.is_floating_point(value):
                        value = value.float()
                    value = value.to(device)
                    batch[key] = value
            predictions.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def predict_base0_csv(
    input_csv,
    output_csv=None,
    target_fasta=None,
    model_name=DEFAULT_MODEL,
    ckpt_dir=DEFAULT_CKPT_DIR,
    folds=None,
    train_data=DEFAULT_TRAIN_DATA,
    rnafm_cache=None,
    rebuild_rnafm=False,
    batch_size=128,
    rnafm_batch_size=64,
    max_len=25,
    mrna_max_len=65,
    window_up=20,
    window_down=20,
    min_query_len=16,
    concentration_nm=10.0,
    cell_type="",
    as_ts=3,
    ss_ts=3,
    cell_embed_dim=8,
    device=None,
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv) if output_csv is not None else input_csv
    target_fasta = Path(target_fasta) if target_fasta is not None else None
    ckpt_dir = Path(ckpt_dir)
    train_data = Path(train_data)
    folds = _parse_folds(folds)
    device = torch.device(device) if device else _default_device()

    df = pd.read_csv(input_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing Base0 columns: {missing}")
    print(f"[Base0] predicting {len(df)} modified candidates from {input_csv}")
    print(f"[Base0] model={model_name}")
    print(f"[Base0] ckpt_dir={ckpt_dir}")
    print(f"[Base0] folds={','.join(folds)}")
    print(f"[Base0] device={device}")

    df = _prepare_target_context(
        df, target_fasta, window_up, window_down, min_query_len
    )
    unique_windows = list(dict.fromkeys(df["local_mrna"].astype(str).tolist()))
    window_to_index = {window: i for i, window in enumerate(unique_windows)}
    rnafm_indices = np.asarray(
        [window_to_index[str(window)] for window in df["local_mrna"]], dtype=int
    )
    if rnafm_cache is None:
        rnafm_cache = input_csv.with_name(
            f".{input_csv.stem}_base0_mrna_rnafm_maxlen{mrna_max_len}.npy"
        )
    mrna_rnafm = _rnafm_embeddings(
        unique_windows, rnafm_cache, mrna_max_len, rnafm_batch_size,
        rebuild_rnafm, device,
    )

    train_df = base0_data._load_data(str(train_data))
    if len(train_df) != 22622:
        raise ValueError(
            f"Expected final 22,622-row Base0 training data, found {len(train_df)}"
        )
    candidate_dict = _frame_dict(
        df, rnafm_indices, concentration_nm, cell_type
    )
    base0_data.MRNA_RNAFM = mrna_rnafm
    base0_data.MRNA_MAX_LEN = mrna_max_len

    fold_predictions = []
    for fold in folds:
        checkpoint = _checkpoint_for_fold(ckpt_dir, model_name, fold)
        print(f"[Base0] loading {fold}: {checkpoint}")
        pssm_as, pssm_ss, conc_mean, conc_std = _fold_feature_context(train_df, fold)
        dataset = base0_data.Generate_dataset(
            candidate_dict, pssm_as, pssm_ss, max_len=max_len,
            conc_log_mean=conc_mean, conc_log_std=conc_std,
        )
        model = _load_model(
            model_name, checkpoint, device, as_ts, ss_ts, cell_embed_dim
        )
        prediction = _predict(model, dataset, batch_size, device)
        fold_predictions.append(prediction)
        df[f"Base0_{fold}"] = np.round(prediction, 6)
        del model, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_array = np.stack(fold_predictions, axis=0)
    df["Base0_avg"] = np.round(prediction_array.mean(axis=0), 6)
    df["Base0_std"] = np.round(prediction_array.std(axis=0), 6)
    df["Base0_rank"] = pd.Series(df["Base0_avg"]).rank(
        ascending=False, method="min"
    ).astype(int)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Base0] saved predictions to {output_csv}")
    return df


def pred_sirnacm(args):
    input_csv = _resolve_sirnacm_path(args.sirnacm_path)
    if not input_csv.exists():
        raise FileNotFoundError(f"Modified-siRNA CSV not found: {input_csv}")

    target_fasta = _resolve_optional_path(
        getattr(args, "base0_target_fasta", None)
    )
    if target_fasta is None and getattr(args, "g", None) and len(args.g) == 1:
        inferred = PIPELINE_DIR / "output" / "mrna" / f"{args.g[0]}.fasta"
        if inferred.exists():
            target_fasta = inferred

    cache_value = getattr(args, "base0_rnafm_cache", "")
    rnafm_cache = _resolve_optional_path(cache_value) if cache_value else None
    predict_base0_csv(
        input_csv=input_csv,
        output_csv=input_csv,
        target_fasta=target_fasta,
        model_name=getattr(args, "base0_model", DEFAULT_MODEL),
        ckpt_dir=getattr(args, "base0_ckpt_dir", str(DEFAULT_CKPT_DIR)),
        folds=getattr(args, "base0_folds", ",".join(str(i) for i in range(10))),
        train_data=getattr(args, "base0_train_data", str(DEFAULT_TRAIN_DATA)),
        rnafm_cache=rnafm_cache,
        rebuild_rnafm=getattr(args, "base0_rebuild_rnafm", False),
        batch_size=getattr(args, "base0_batch_size", 128),
        rnafm_batch_size=getattr(args, "base0_rnafm_batch_size", 64),
        max_len=getattr(args, "base0_max_len", 25),
        mrna_max_len=getattr(args, "base0_mrna_max_len", 65),
        window_up=getattr(args, "base0_window_up", 20),
        window_down=getattr(args, "base0_window_down", 20),
        concentration_nm=getattr(args, "base0_concentration_nm", 10.0),
        cell_type=getattr(args, "base0_cell_type", ""),
        as_ts=getattr(args, "base0_as_ts", 3),
        ss_ts=getattr(args, "base0_ss_ts", 3),
        cell_embed_dim=getattr(args, "base0_cell_embed_dim", 8),
        device=getattr(args, "device", None),
    )
