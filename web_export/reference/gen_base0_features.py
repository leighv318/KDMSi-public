#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for the Base0 remaining hand-crafted features (TS parity).

Calls the REAL base0model feature functions on a few (cm, ori) strand pairs from
the training CSV, applying predictor_base0._frame_dict's strip + T->t transform
so the inputs match inference. PSSM / concentration context is the fold
"0-inter" context loaded from the shipped
webapp/public/assets/base0_fold_context.json — so the fixture's pssm_score uses
the exact matrix the browser ships (as_pssm_score scores ori_as with pssm_as,
ss_pssm_score scores ori_ss with pssm_ss). Run AFTER
web_export/precompute_base0_fold_context.py.

Run: conda run -n kdmsi python web_export/reference/gen_base0_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "base0model"))
from load_data import (  # noqa: E402
    get_tri_comp_percent, get_di_comp_percent, get_single_comp_percent,
    get_gc_percentage, get_gc_sterch, score_seq_by_pssm, CELL_TYPE2IDX,
)
from mod_aux_features import compute_per_position_aux  # noqa: E402

CSV = REPO / "base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv"
CTX = REPO / "webapp/public/assets/base0_fold_context.json"
OUT = REPO / "webapp/test/fixtures/base0_features.json"
N = 8
MAX_LEN = 25
FOLD = "0-inter"
CONC_VALUES = [10.0, 100.0, 1.0, 50.0]


def frame_seq(x):
    """Mirror predictor_base0._frame_dict seq_values: strip + T->t."""
    return str(x).strip().replace("T", "t")


def main():
    df = pd.read_csv(CSV).head(N)
    ctx = json.loads(CTX.read_text())[FOLD]
    pssm_as = np.asarray(ctx["pssm_as"], dtype=float)  # (5,19) rows A,G,C,U,t
    pssm_ss = np.asarray(ctx["pssm_ss"], dtype=float)
    conc_mean = float(ctx["conc_log_mean"])
    conc_std = float(ctx["conc_log_std"])
    conc_std_eff = conc_std if conc_std > 1e-8 else 1.0

    rows = []
    for _, r in df.iterrows():
        ori_as, ori_ss = frame_seq(r["ori_as"]), frame_seq(r["ori_ss"])
        cm_as, cm_ss = frame_seq(r["cm_as"]), frame_seq(r["cm_ss"])
        for strand, ori, cm, pssm in (
            ("as", ori_as, cm_as, pssm_as),
            ("ss", ori_ss, cm_ss, pssm_ss),
        ):
            mod_aux = compute_per_position_aux(cm, MAX_LEN, len(ori))  # (25,15)
            rows.append({
                "strand": strand, "cm": cm, "ori": ori, "seqLen": len(ori),
                "modAux": mod_aux.astype(float).round(6).tolist(),
                "tri": get_tri_comp_percent(ori)[:, 0].astype(float).round(6).tolist(),
                "di": get_di_comp_percent(ori)[:, 0].astype(float).round(6).tolist(),
                "single": get_single_comp_percent(cm)[:, 0].astype(float).round(6).tolist(),
                "gcContent": float(round(float(get_gc_percentage(ori)[0, 0]), 6)),
                "gcStretch": float(round(float(get_gc_sterch(ori)[0, 0]), 6)),
                "pssmScore": float(round(float(score_seq_by_pssm(pssm, ori[:19])[0, 0]), 6)),
            })

    conc = []
    for c in CONC_VALUES:
        log_conc = float(np.log10(max(c, 1e-9)))
        z = (log_conc - conc_mean) / conc_std_eff
        conc.append({"nm": c, "feature": [round(log_conc, 6), round(z, 6)]})

    out = {
        "fold": FOLD,
        "conc_log_mean": conc_mean,
        "conc_log_std": conc_std,
        "rows": rows,
        "concentration": conc,
        "cell_type2idx": {str(k): int(v) for k, v in CELL_TYPE2IDX.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote {len(rows)} strand-features + {len(conc)} conc -> {OUT}")


if __name__ == "__main__":
    main()
