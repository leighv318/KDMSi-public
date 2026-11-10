#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute the per-position modification distributions to JSON (shipped asset).

Imports the REAL design_modified._position_distributions so the shipped JSON is
byte-identical to what the pipeline builds at runtime. For each strand position
it emits (states, probs) where states preserve list(Counter) FIRST-INSERTION
order (inverse-CDF sampling depends on it) and each state is [q_count, symbol|null].
AS length = 23, SS length = 21 (design_ss_length 21, AS = SS+2).

Output: webapp/public/assets/mod_distributions.json =
  { "as_length":23, "ss_length":21,
    "as":[ {"states":[[q,sym|null],...], "probs":[...]} x23 ], "ss":[ ... x21 ] }

Run: conda run -n kdmsi python web_export/precompute_mod_distributions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "liubin_pipeline"))
from potency_prediction.design_modified import _position_distributions  # noqa: E402

CSV = REPO / "base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv"
OUT = REPO / "webapp/public/assets/mod_distributions.json"
SS_LEN = 21
AS_LEN = SS_LEN + 2  # 23
EXPECTED_ROWS = 22622


def serialize(distributions):
    return [
        {"states": [[int(q), sym] for (q, sym) in states], "probs": [float(p) for p in probs]}
        for (states, probs) in distributions
    ]


def main():
    train_df = pd.read_csv(CSV, low_memory=False)
    if len(train_df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} training rows, found {len(train_df)}")
    as_dist, as_examples = _position_distributions(train_df, "ori_as", "cm_as", AS_LEN)
    ss_dist, ss_examples = _position_distributions(train_df, "ori_ss", "cm_ss", SS_LEN)
    out = {
        "as_length": AS_LEN,
        "ss_length": SS_LEN,
        "as": serialize(as_dist),
        "ss": serialize(ss_dist),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] AS {AS_LEN}nt ({as_examples} rows), SS {SS_LEN}nt ({ss_examples} rows) -> {OUT}")


if __name__ == "__main__":
    main()
