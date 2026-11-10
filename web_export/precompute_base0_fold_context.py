#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute the 10 Base0 per-fold feature contexts to JSON (shipped asset).

Mirrors predictor_base0._fold_feature_context EXACTLY (l.366-376): for fold k,
the training subset is `index % 10 != k` over the original row order of the
22,622-row training frame, from which we build 5-row PSSMs (rows A,G,C,U,t) for
AS and SS, plus the concentration log10 mean/std. To guarantee byte-identical
context to inference we import the real _load_data + create_pssm (so the row
filtering / reset_index and PSSM math are the pipeline's own), and replicate the
tiny _concentration_nm / _fold_feature_context bodies here (avoids importing the
full predictor/model/RNA-FM chain).

Output: webapp/public/assets/base0_fold_context.json =
  { "<fold>": { "pssm_as":[5][19], "pssm_ss":[5][19],
                "conc_log_mean":float, "conc_log_std":float } }

Run: conda run -n kdmsi python web_export/precompute_base0_fold_context.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "base0model"))
from load_data import _load_data, create_pssm  # noqa: E402

CSV = REPO / "base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv"
OUT = REPO / "webapp/public/assets/base0_fold_context.json"
FOLDS = [f"{i}-inter" for i in range(10)]
EXPECTED_ROWS = 22622


def _concentration_nm(df, default_value):
    """Replica of predictor_base0._concentration_nm (l.324-331)."""
    import pandas as pd
    for column in ["concentration/nM", "concentration_nM", "concentration_nm",
                   "Concentration_nM", "Concentration"]:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(
                default_value
            ).clip(lower=1e-9).to_numpy(dtype=np.float32)
    return np.full(len(df), float(default_value), dtype=np.float32)


def _fold_feature_context(train_df, fold):
    """Replica of predictor_base0._fold_feature_context (l.366-376)."""
    fold_number = int(fold.split("-")[0])
    indices = np.arange(len(train_df))
    fold_train = train_df.iloc[indices[indices % 10 != fold_number]]
    pssm_as = create_pssm(np.asarray(fold_train["ori_as"]))
    pssm_ss = create_pssm(np.asarray(fold_train["ori_ss"]))
    concentrations = _concentration_nm(fold_train, 10.0).astype(float)
    log_conc = np.log10(np.clip(concentrations, 1e-9, None))
    mean = float(log_conc.mean())
    std = float(log_conc.std())
    return pssm_as, pssm_ss, mean, std if std > 1e-8 else 1.0


def main():
    train_df = _load_data(str(CSV))
    if len(train_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS}-row Base0 training data, found {len(train_df)}"
        )
    out = {}
    for fold in FOLDS:
        pssm_as, pssm_ss, mean, std = _fold_feature_context(train_df, fold)
        assert pssm_as.shape == (5, 19), pssm_as.shape
        assert pssm_ss.shape == (5, 19), pssm_ss.shape
        out[fold] = {
            "pssm_as": pssm_as.astype(float).tolist(),  # [5][19] rows A,G,C,U,t
            "pssm_ss": pssm_ss.astype(float).tolist(),
            "conc_log_mean": mean,
            "conc_log_std": std,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote {len(out)} folds -> {OUT} (pssm [5,19]; "
          f"fold 0 conc_log_mean={out['0-inter']['conc_log_mean']:.4f})")


if __name__ == "__main__":
    main()
