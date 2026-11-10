#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth Base3_avg for a subset of AGT candidates (Stage 3 TS parity).

Runs the REAL predict_base3_csv (torch RNA-FM + 10 Base3 fold models) on the
first K AGT windows and records Base3_avg. The TS verify script (verify_base3.ts)
scores the same candidates via the exported Base3 ONNX fold models + RNA-FM ONNX
and must match within onnx-vs-torch tolerance.

Heavy: loads RNA-FM (~1.19GB) once. Run:
  conda run -n kdmsi python web_export/reference/gen_base3_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "liubin_pipeline"))
from potency_prediction.design_modified import _base3_candidates  # noqa: E402
from potency_prediction.predictor import predict_base3_csv  # noqa: E402

AGT = REPO / "examples/AGT.fasta"
OUT = REPO / "webapp/test/fixtures/base3_reference.json"
TMP = REPO / "webapp/test/fixtures/.base3_ref_input.csv"
K = 24


def main():
    cand = _base3_candidates(str(AGT), "AGT").head(K).copy().reset_index(drop=True)
    cand.to_csv(TMP, index=False)
    predict_base3_csv(input_csv=str(TMP), output_csv=str(TMP))  # all 10 folds
    scored = pd.read_csv(TMP)
    out = {
        "candidates": [{"AS": r["AS"], "local_mrna": r["local_mrna"]} for _, r in scored.iterrows()],
        "base3_avg": [round(float(r["Base3_avg"]), 6) for _, r in scored.iterrows()],
    }
    OUT.write_text(json.dumps(out), encoding="utf-8")
    TMP.unlink(missing_ok=True)
    print(f"[ok] {len(out['candidates'])} candidates, Base3_avg[:3]={out['base3_avg'][:3]} -> {OUT}")


if __name__ == "__main__":
    main()
