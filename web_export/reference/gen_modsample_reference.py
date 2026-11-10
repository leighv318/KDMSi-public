#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for Stage-4 modification sampling (TS parity, seed 126).

Imports the REAL design_modified._base3_candidates + _generate_modified_candidates
so the fixture is exactly what the pipeline produces. Builds a small top_df from
the first few AGT windows (Base3_rank 1..K), samples patterns_per_sirna=6 with
seed 126, and records the input duplexes + the sampled ori/cm strands in order.
The TS test replays the SAME duplexes with webapp/public/assets/mod_distributions.json
+ MT19937(126) and must reproduce the strands byte-for-byte.

Run: conda run -n kdmsi python web_export/reference/gen_modsample_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "liubin_pipeline"))
from potency_prediction.design_modified import (  # noqa: E402
    _base3_candidates, _generate_modified_candidates,
)

AGT = REPO / "examples/AGT.fasta"
TRAIN = REPO / "base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv"
OUT = REPO / "webapp/test/fixtures/modsample_reference.json"

GENE = "AGT"
SENSE_LEN = 21
OVERHANG = "target-derived"
PATTERNS = 6
SEED = 126
N_DUPLEX = 3


def main():
    cand = _base3_candidates(str(AGT), GENE)
    top_df = cand.head(N_DUPLEX).copy().reset_index(drop=True)
    top_df["Base3_rank"] = range(1, len(top_df) + 1)

    train_df = pd.read_csv(TRAIN, low_memory=False)
    assert len(train_df) == 22622, len(train_df)

    result = _generate_modified_candidates(
        top_df, train_df, SENSE_LEN, OVERHANG, PATTERNS, SEED, GENE
    )

    candidates = [
        {
            "gene": r["gene"], "NM_id": r["NM_id"], "AS": r["AS"], "SS": r["SS"],
            "local_mrna": r["local_mrna"], "mrna_len": int(r["mrna_len"]),
            "Base3_rank": int(r["Base3_rank"]),
        }
        for _, r in top_df.iterrows()
    ]
    rows = [
        {
            "Base3_rank": int(r["Base3_rank"]),
            "pattern_index_within_core": int(r["pattern_index_within_core"]),
            "ori_as": r["ori_as"], "ori_ss": r["ori_ss"],
            "cm_as": r["cm_as"], "cm_ss": r["cm_ss"],
            "pattern_log_probability": round(float(r["pattern_log_probability"]), 6),
        }
        for _, r in result.iterrows()
    ]

    out = {
        "params": {"gene": GENE, "senseLength": SENSE_LEN, "overhangMode": OVERHANG,
                   "patternsPerSirna": PATTERNS, "seed": SEED},
        "candidates": candidates,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] {len(candidates)} duplexes -> {len(rows)} modified candidates -> {OUT}")


if __name__ == "__main__":
    main()
