#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for the Base0 per-residue encoding + energy index (TS parity).

Calls the real base0model.load_data.sequence_encoding_ / Energy_index on a few
(cm_as, ori_as) and (cm_ss, ori_ss) pairs from the training CSV.

Run: conda run -n kdmsi python web_export/reference/gen_base0_encoding.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "base0model"))
from load_data import sequence_encoding_, dict_onehot  # noqa: E402


def energy_index(cm_seq, max_len):
    """Replicates load_data.Energy_index with 0-padding (that file's
    `result[i] = np.array([0])` scalar assignment errors on numpy>=2.0;
    the intended and pre-numpy-2 behavior is padding with 0)."""
    cm_seq = str(cm_seq).strip()
    result = [0.0] * max_len
    seq_ind = res_ind = 0
    while seq_ind < len(cm_seq) and res_ind < max_len:
        sym = cm_seq[seq_ind]
        result[res_ind] = float(dict_onehot[sym] + 1)
        seq_ind += 1
        if sym != "q":
            res_ind += 1
    return result

CSV = REPO / "base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv"
OUT = REPO / "webapp/test/fixtures/base0_encoding.json"
N = 8
MAX_LEN = 25


def main():
    df = pd.read_csv(CSV).head(N)
    out = []
    for _, r in df.iterrows():
        for strand in ("as", "ss"):
            cm = str(r[f"cm_{strand}"]).strip()
            ori = str(r[f"ori_{strand}"]).strip()
            emb = sequence_encoding_(cm, ori, MAX_LEN)   # [25,162]
            en = energy_index(cm, MAX_LEN)                # [25]
            out.append({
                "strand": strand, "cm": cm, "ori": ori,
                "embedding": emb.astype(float).round(6).tolist(),
                "energy": en,
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote {len(out)} strand-encodings -> {OUT}")


if __name__ == "__main__":
    main()
