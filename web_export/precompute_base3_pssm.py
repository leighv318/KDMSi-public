#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute the 10 Base3 fold PSSM matrices to JSON (shipped, not the CSV).

Mirrors predictor._create_pssm / _pssm_for_fold exactly. Output:
  webapp/public/assets/base3_pssm.json = { "<fold>": [[A..21],[G..21],[C..21],[U..21]] }
(rows A,G,C,U; 3-decimal rounded; shape [4,21] per fold).

Run: conda run -n kdmsi python web_export/precompute_base3_pssm.py
(needs only pandas/numpy — no torch/RNA.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "Base3model/data/normalized_sirna_with_mrna.csv"
# Shipped asset (small): lives under webapp so it's bundled/served, not in the
# gitignored web_export/artifacts (which holds only the large ONNX/wasm files).
OUT = REPO / "webapp/public/assets/base3_pssm.json"
FOLDS = [f"{i}-inter" for i in range(10)]


def clean_rna(s: str) -> str:
    return str(s).strip().upper().replace("T", "U")


def create_pssm(train_seq):
    seqs = [list(clean_rna(s)) for s in train_seq]
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
    return np.asarray(pssm, dtype=np.float32).T  # [4, nc]


def main():
    data = pd.read_csv(CSV)
    rows = len(data)
    out = {}
    for fold in FOLDS:
        start = int(fold.split("-")[0])
        test_idx = set(np.arange(start, rows, 10).tolist())
        train_idx = [i for i in range(rows) if i not in test_idx]
        pssm = create_pssm(np.asarray(data.iloc[train_idx]["Antisense"]))
        out[fold] = pssm.tolist()  # [4][21], rows A,G,C,U
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote {len(out)} folds -> {OUT} (shape [4,{len(out[FOLDS[0]][0])}] each)")


if __name__ == "__main__":
    main()
