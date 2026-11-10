"""
New Base3 inference backend for liubin_pipeline.

Default behavior:
    Base3_cross_access_v2_src_no_se, averaged over 0-inter ... 9-inter checkpoints
    under liubin_pipeline/potency_prediction/pretrain_models/Base3_cross_access_v2_src_no_se.

The public entry point is pred_sirna(args), matching the old pipeline API.
"""

from __future__ import annotations

import math
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
BASE3_DIR = ROOT_DIR / "Base3model"
DEFAULT_MODEL = "Base3_cross_access_v2_src_no_se"
DEFAULT_CKPT_DIR = ROOT_DIR / "liubin_pipeline" / "potency_prediction" / "pretrain_models" / DEFAULT_MODEL
DEFAULT_TRAIN_DATA = BASE3_DIR / "data" / "normalized_sirna_with_mrna.csv"

BASES4 = ["A", "G", "C", "U"]
TRI_KMERS = ["".join(t) for t in product(BASES4, repeat=3)]
DI_KMERS = ["".join(t) for t in product(BASES4, repeat=2)]
RNAFM_VALID_CHARS = set("AUCGRYKMWSBDHVN-")
RNAFM_PADDING_CHARS = set(".")


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_base3_import_path():
    base3 = str(BASE3_DIR)
    if base3 not in sys.path:
        sys.path.insert(0, base3)


def _clean_rna(seq):
    return str(seq).upper().replace("T", "U").strip()


def _require_rna(seq, name):
    seq = _clean_rna(seq)
    bad = set(seq) - set("AUCG")
    if bad:
        raise ValueError(f"{name} contains unsupported RNA bases {sorted(bad)}: {seq}")
    return seq


def _parse_folds(value):
    if value is None or str(value).strip() == "":
        return [f"{i}-inter" for i in range(10)]
    folds = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        folds.append(item if item.endswith("-inter") or item in {"R", "T"} else f"{item}-inter")
    return folds


def _checkpoint_for_fold(ckpt_dir, model_name, fold):
    ckpt_dir = Path(ckpt_dir)
    candidates = [
        ckpt_dir / fold / f"model_{fold}.pth.tar",
        ckpt_dir / fold / "model_best.pth.tar",
        ckpt_dir / f"{model_name}_{fold}.pth.tar",
        ckpt_dir / f"model_{fold}.pth.tar",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No checkpoint found for fold "
        f"{fold}. Tried: " + ", ".join(str(p) for p in candidates)
    )


def _create_pssm(train_seq):
    seqs = [list(_require_rna(seq, "PSSM train sequence")) for seq in train_seq]
    arr = np.array(seqs)
    nr, nc = arr.shape
    pseudocount = nr ** 0.5
    denom = nr + 4 * pseudocount
    pssm = []
    for c in range(nc):
        col = arr[:, c].tolist()
        pssm.append([
            round((col.count("A") + pseudocount) / denom, 3),
            round((col.count("G") + pseudocount) / denom, 3),
            round((col.count("C") + pseudocount) / denom, 3),
            round((col.count("U") + pseudocount) / denom, 3),
        ])
    return np.asarray(pssm, dtype=np.float32).T


def _pssm_for_fold(train_data_path, fold):
    data = pd.read_csv(train_data_path)
    rows = len(data)
    if fold.endswith("-inter"):
        start = int(fold.split("-")[0])
        test_idx = set(np.arange(start, rows, 10).tolist())
    elif fold in {"R", "T"} and "source_paper" in data.columns:
        test_idx = set(np.where(np.asarray(data["source_paper"]) == fold)[0].tolist())
    else:
        test_idx = set()
    train_idx = [i for i in range(rows) if i not in test_idx]
    return _create_pssm(np.asarray(data.iloc[train_idx]["Antisense"]))


def _score_pssm(pssm, seq):
    nt_order = {"A": 0, "G": 1, "C": 2, "U": 3, "N": 0}
    seq = _clean_rna(seq)
    scores = [pssm[nt_order.get(nt, 0), i] for i, nt in enumerate(seq[: pssm.shape[1]])]
    return float(sum(-math.log2(max(float(v), 1e-10)) for v in scores))


def _kmer_percents(seq):
    seq = _require_rna(seq, "AS")
    n = len(seq)
    tri = np.asarray([seq.count(k) / max(n - 2, 1) for k in TRI_KMERS], dtype=np.float32)
    di = np.asarray([seq.count(k) / max(n - 1, 1) for k in DI_KMERS], dtype=np.float32)
    single = np.asarray([seq.count(b) / max(n, 1) for b in BASES4], dtype=np.float32)
    return tri, di, single


def _gc_content(seq):
    seq = _require_rna(seq, "AS")
    return float((seq.count("G") + seq.count("C")) / max(len(seq), 1))


def _gc_stretch(seq):
    seq = _require_rna(seq, "AS")
    best = cur = 0
    for nt in seq:
        if nt in "GC":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best / max(len(seq), 1))


def _gibbs_energy(seq):
    seq = _require_rna(seq, "AS")[:19]
    table = {
        "AA": -0.93, "AU": -1.10, "AG": -2.35, "AC": -2.11,
        "UA": -1.33, "UU": -0.93, "UG": -2.24, "UC": -2.08,
        "GA": -2.35, "GU": -2.24, "GG": -3.26, "GC": -3.42,
        "CA": -2.11, "CU": -2.08, "CG": -2.36, "CC": -3.26,
    }
    values = [table.get(seq[i:i + 2], 0.0) for i in range(max(len(seq) - 1, 0))]
    while len(values) < 18:
        values.append(0.0)
    values = values[:18]
    total = float(np.asarray(values).sum().round(3))
    end_diff = float(round(values[0] - values[-1], 3)) if values else 0.0
    return np.asarray(values + [total, end_diff], dtype=np.float32)[:, np.newaxis]


def _secondary_structure(seq):
    try:
        import RNA

        rna = _require_rna(seq, "AS")
        struct, mfe = RNA.fold(rna)
        n = max(len(struct), 1)
        paired = struct.count("(") + struct.count(")")
        unpaired = struct.count(".")
        return (
            np.asarray([[paired / n], [unpaired / n]], dtype=np.float32),
            np.asarray([[float(mfe)]], dtype=np.float32),
        )
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "RNA":
            raise RuntimeError(
                "ViennaRNA Python package is required for Base3 secondary-structure "
                "features. Install/import RNA before running Base3 prediction."
            ) from exc
        print(f"[warn] ViennaRNA fold failed for AS={seq}: {exc}")
        return np.zeros((2, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)


def _target_19_from_window(mrna_window, center=29):
    seq = _clean_rna(mrna_window)
    if len(seq) >= 59:
        target = seq[center - 9:center + 10]
    else:
        c = len(seq) // 2
        target = seq[max(0, c - 9):c + 10]
    return target.replace(".", "")


def _target_accessibility(mrna_window):
    try:
        import RNA

        clean = _clean_rna(mrna_window).replace(".", "")
        if len(clean) < 5:
            return np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32)
        try:
            _, full_mfe = RNA.fold(clean)
            full_mfe = float(full_mfe)
        except Exception:
            full_mfe = 0.0

        target_19 = _target_19_from_window(mrna_window)
        if len(target_19) < 5:
            return np.asarray([[full_mfe], [0.0], [1.0]], dtype=np.float32)
        try:
            local_struct, local_mfe = RNA.fold(target_19)
            local_mfe = float(local_mfe)
            unpaired_ratio = (
                local_struct.count(".") / len(local_struct)
                if len(local_struct) > 0 else 1.0
            )
        except Exception:
            local_mfe = 0.0
            unpaired_ratio = 1.0
        return np.asarray([[full_mfe], [local_mfe], [unpaired_ratio]], dtype=np.float32)
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "RNA":
            raise RuntimeError(
                "ViennaRNA Python package is required for Base3 target_accessibility "
                "features. Install/import RNA before running Base3 prediction."
            ) from exc
        print(f"[warn] target accessibility failed for mRNA={mrna_window}: {exc}")
        return np.zeros((3, 1), dtype=np.float32)


def _rnafm_mrna_batch(model, batch_converter, sequences, device):
    batch_data = []
    meta = []
    for i, raw in enumerate(sequences):
        raw_seq = _clean_rna(raw)
        bad = set(raw_seq) - RNAFM_VALID_CHARS - RNAFM_PADDING_CHARS
        if bad:
            raise ValueError(f"mRNA window contains unsupported chars {sorted(bad)}: {raw_seq}")
        valid_positions = [k for k, ch in enumerate(raw_seq) if ch in RNAFM_VALID_CHARS]
        clean_seq = "".join(raw_seq[k] for k in valid_positions)
        if not clean_seq:
            clean_seq = "N"
            valid_positions = [0]
        batch_data.append((str(i), clean_seq))
        meta.append((raw_seq, valid_positions, len(clean_seq)))

    _, _, tokens = batch_converter(batch_data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[12])
    token_emb = out["representations"][12].detach().cpu().numpy()

    result = []
    for j, (raw_seq, valid_positions, clean_len) in enumerate(meta):
        emb = token_emb[j, 1:clean_len + 1, :]
        full = np.zeros((len(raw_seq), emb.shape[1]), dtype=np.float32)
        usable = min(len(valid_positions), emb.shape[0])
        if usable:
            full[np.asarray(valid_positions[:usable]), :] = emb[:usable]
        result.append(full)
    return result


def _rnafm_as_batch(model, batch_converter, sequences, device):
    batch_data = []
    lengths = []
    for i, seq in enumerate(sequences):
        rna = _require_rna(seq, "AS")
        batch_data.append((str(i), rna))
        lengths.append(len(rna))
    _, _, tokens = batch_converter(batch_data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[12])
    token_emb = out["representations"][12].detach().cpu().numpy()
    return [token_emb[i, 1:length + 1, :].astype(np.float32) for i, length in enumerate(lengths)]


def _compute_rnafm(as_list, mrna_list, batch_size, device):
    import fm

    print("[Base3] loading RNA-FM...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    as_embeddings = []
    mrna_embeddings = []
    for start in range(0, len(as_list), batch_size):
        end = min(start + batch_size, len(as_list))
        as_embeddings.extend(_rnafm_as_batch(model, batch_converter, as_list[start:end], device))
        mrna_embeddings.extend(_rnafm_mrna_batch(model, batch_converter, mrna_list[start:end], device))
        print(f"[Base3] RNA-FM {end}/{len(as_list)}")
    return as_embeddings, mrna_embeddings


class Base3CandidateDataset(Dataset):
    def __init__(self, df, pssm, rnafm_as, rnafm_mrna):
        self.items = []
        for i, row in df.reset_index(drop=True).iterrows():
            as_seq = _require_rna(row["AS"], "AS")
            mrna = _clean_rna(row["local_mrna"])
            tri, di, single = _kmer_percents(as_seq)
            second_pct, second_energy = _secondary_structure(as_seq)
            self.items.append({
                "rnafm_encode": rnafm_as[i],
                "rnafm_encode_mrna": rnafm_mrna[i],
                "sirna_gibbs_energy": _gibbs_energy(as_seq),
                "pssm_score": np.asarray([[_score_pssm(pssm, as_seq)]], dtype=np.float32),
                "gc_sterch": np.asarray([[_gc_stretch(as_seq)]], dtype=np.float32),
                "sirna_second_percent": second_pct,
                "sirna_second_energy": second_energy,
                "tri_nt_percent": tri[:, np.newaxis],
                "di_nt_percent": di[:, np.newaxis],
                "single_nt_percent": single[:, np.newaxis],
                "gc_content": np.asarray([[_gc_content(as_seq)]], dtype=np.float32),
                "target_accessibility": _target_accessibility(mrna),
                "source_id": np.asarray([[3.0]], dtype=np.float32),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                value = value.float()
            value = value.to(device)
        out[key] = value
    return out


def _load_model(model_name, ckpt_path, device):
    _ensure_base3_import_path()
    import model_attention

    cls = getattr(model_attention, model_name)
    model = cls(dp=0.0, device=device).float().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _predict_one_model(model, loader, device):
    preds = []
    with torch.no_grad():
        for batch in loader:
            pred = model(_move_batch(batch, device))
            preds.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def predict_base3_csv(
    input_csv,
    output_csv=None,
    model_name=DEFAULT_MODEL,
    ckpt_dir=DEFAULT_CKPT_DIR,
    folds=None,
    train_data=DEFAULT_TRAIN_DATA,
    batch_size=128,
    rnafm_batch_size=64,
    device=None,
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv) if output_csv is not None else input_csv
    ckpt_dir = Path(ckpt_dir)
    train_data = Path(train_data)
    folds = _parse_folds(folds)
    device = torch.device(device) if device else _default_device()

    df = pd.read_csv(input_csv)
    required = {"AS", "local_mrna"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{input_csv} is missing required columns for Base3 prediction: {missing}")

    print(f"[Base3] predicting {len(df)} candidates from {input_csv}")
    print(f"[Base3] model={model_name}")
    print(f"[Base3] ckpt_dir={ckpt_dir}")
    print(f"[Base3] folds={','.join(folds)}")
    print(f"[Base3] device={device}")

    rnafm_as, rnafm_mrna = _compute_rnafm(
        np.asarray(df["AS"]),
        np.asarray(df["local_mrna"]),
        batch_size=int(rnafm_batch_size),
        device=device,
    )

    fold_preds = []
    for fold in folds:
        ckpt = _checkpoint_for_fold(ckpt_dir, model_name, fold)
        print(f"[Base3] loading {fold}: {ckpt}")
        pssm = _pssm_for_fold(train_data, fold)
        dataset = Base3CandidateDataset(df, pssm, rnafm_as, rnafm_mrna)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, drop_last=False)
        model = _load_model(model_name, ckpt, device)
        pred = _predict_one_model(model, loader, device)
        fold_preds.append(pred)
        df[f"Base3_{fold}"] = np.round(pred, 6)

    pred_arr = np.stack(fold_preds, axis=0)
    df["Base3_avg"] = np.round(pred_arr.mean(axis=0), 6)
    df["Base3_std"] = np.round(pred_arr.std(axis=0), 6)
    df.to_csv(output_csv, index=False)
    print(f"[Base3] saved predictions to {output_csv}")
    return df


def pred_sirna(args):
    if not getattr(args, "g", None):
        raise ValueError("Base3 prediction requires args.g gene list.")
    model_name = getattr(args, "base3_model", DEFAULT_MODEL)
    ckpt_dir = getattr(args, "base3_ckpt_dir", str(DEFAULT_CKPT_DIR))
    folds = getattr(args, "base3_folds", ",".join(str(i) for i in range(10)))
    train_data = getattr(args, "base3_train_data", str(DEFAULT_TRAIN_DATA))
    batch_size = getattr(args, "base3_batch_size", 128)
    rnafm_batch_size = getattr(args, "base3_rnafm_batch_size", 64)
    device = getattr(args, "device", None)

    for gene in args.g:
        cur_gene_path = Path("./output/sirna") / f"{gene}.csv"
        if not cur_gene_path.exists():
            print(f"[Base3] skip {gene}: {cur_gene_path} not found")
            continue
        predict_base3_csv(
            input_csv=cur_gene_path,
            output_csv=cur_gene_path,
            model_name=model_name,
            ckpt_dir=ckpt_dir,
            folds=folds,
            train_data=train_data,
            batch_size=batch_size,
            rnafm_batch_size=rnafm_batch_size,
            device=device,
        )
