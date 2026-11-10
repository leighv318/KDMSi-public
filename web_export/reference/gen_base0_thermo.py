#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for the Base0 mRNA thermodynamic features (mrna_thermo, 5).

Calls the REAL base0model.compute_mrna_thermo.compute_accessibility /
compute_local_fold_dg on a few local-mRNA windows (incl. edge cases: a T-bearing
window, an L<5 guard, and a target region clamped to the window end), then stores
the 5-scalar vector with NaN->0 exactly as load_data.load_sirna does. Column
order = load_data.MRNA_THERMO_COLS
[acc_target_mean, acc_seed_mean, acc_target_min, local_fold_dg, local_fold_dg_norm].

Run: conda run -n kdmsi python web_export/reference/gen_base0_thermo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "base0model"))
from compute_mrna_thermo import compute_accessibility, compute_local_fold_dg  # noqa: E402

OUT = REPO / "webapp/test/fixtures/base0_thermo.json"

CASES = [
    # normal RNA window, target in the middle
    {"localMrna": "GGCACUGAUCCGUAGCUAGCUAGGCAUCGAUCGGAUCGUAGCUAG", "targetRelStart": 13, "targetLen": 19},
    # T-bearing window (exercises upper() + T->U before folding)
    {"localMrna": "ATGCGATCGATCGATCGGCATCGATTAGCATCGATCGATCGATCGATGC", "targetRelStart": 15, "targetLen": 19},
    # L < 5 guard -> NaN triple + NaN fold -> all zeros
    {"localMrna": "AUGC", "targetRelStart": 0, "targetLen": 4},
    # target region reaching the window end (t1 clamped, small seed region)
    {"localMrna": "GGCCAUGGCCAUGGCCAUGGCCAUGG", "targetRelStart": 6, "targetLen": 19},
    # targetLen extends past window end (t1 = L clamp)
    {"localMrna": "GGGGGCCCCCAAAAAUUUUUGGGGGCCCCC", "targetRelStart": 5, "targetLen": 25},
]


def main():
    out = []
    for c in CASES:
        acc = compute_accessibility(c["localMrna"], c["targetRelStart"], c["targetLen"])
        fold = compute_local_fold_dg(c["localMrna"])
        thermo = np.nan_to_num(
            np.asarray([acc[0], acc[1], acc[2], fold[0], fold[1]], dtype=np.float32),
            nan=0.0,
        )
        out.append({
            **c,
            "thermo": [round(float(v), 6) for v in thermo],
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote {len(out)} thermo windows -> {OUT}")


if __name__ == "__main__":
    main()
