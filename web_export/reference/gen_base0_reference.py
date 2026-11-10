#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth Base0_avg for Stage-4 modified candidates (Stage 5 TS parity).

Builds a small set of modified candidates via the REAL Stage 4
(_generate_modified_candidates on the first few AGT duplexes, seed 126), then
runs the REAL predict_base0_csv (torch RNA-FM + 10 Base0 fold models, has_local
path since the candidates carry local_mrna) and records Base0_avg. The TS verify
(verify_base0.ts) reranks the same candidates via the Base0 ONNX + RNA-FM ONNX +
ViennaRNA-WASM and must match within onnx-vs-torch tolerance.

Heavy: loads RNA-FM (~1.19GB) once. Run:
  conda run -n kdmsi python web_export/reference/gen_base0_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "liubin_pipeline"))
from potency_prediction.design_modified import _base3_candidates, _generate_modified_candidates  # noqa: E402
from potency_prediction.predictor_base0 import predict_base0_csv  # noqa: E402

AGT = REPO / "examples/AGT.fasta"
TRAIN = REPO / "base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv"
OUT = REPO / "webapp/test/fixtures/base0_reference.json"
TMP = REPO / "webapp/test/fixtures/.base0_ref_input.csv"
N_DUPLEX = 3
PATTERNS = 4
SEED = 126


def main():
    cand = _base3_candidates(str(AGT), "AGT").head(N_DUPLEX).copy().reset_index(drop=True)
    cand["Base3_rank"] = range(1, len(cand) + 1)
    train_df = pd.read_csv(TRAIN, low_memory=False)
    assert len(train_df) == 22622
    modified = _generate_modified_candidates(cand, train_df, 21, "target-derived", PATTERNS, SEED, "AGT")
    modified.to_csv(TMP, index=False)

    # has_local path (candidates carry local_mrna); concentration 10 nM, cell "".
    predict_base0_csv(input_csv=str(TMP), output_csv=str(TMP), target_fasta=None,
                      concentration_nm=10.0, cell_type="")
    scored = pd.read_csv(TMP)

    out = {
        "candidates": [
            {"ori_as": r["ori_as"], "ori_ss": r["ori_ss"], "cm_as": r["cm_as"],
             "cm_ss": r["cm_ss"], "local_mrna": r["local_mrna"]}
            for _, r in scored.iterrows()
        ],
        "base0_avg": [round(float(r["Base0_avg"]), 6) for _, r in scored.iterrows()],
    }
    OUT.write_text(json.dumps(out), encoding="utf-8")
    TMP.unlink(missing_ok=True)
    # predict_base0_csv writes an RNA-FM cache next to the input; clean it up.
    for p in TMP.parent.glob(f".{TMP.stem}_base0_mrna_rnafm_maxlen*.npy*"):
        p.unlink(missing_ok=True)
    print(f"[ok] {len(out['candidates'])} candidates, Base0_avg[:3]={out['base0_avg'][:3]} -> {OUT}")


if __name__ == "__main__":
    main()
