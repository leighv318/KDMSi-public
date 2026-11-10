#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for the Base3 ViennaRNA features (TS parity).

Replicates predictor._secondary_structure (sirna_second_percent[2] +
sirna_second_energy[1]) and _target_accessibility (target_accessibility[3]) +
_target_19_from_window, calling the real ViennaRNA (import RNA) exactly as the
pipeline does. (The two functions are thin RNA.fold wrappers; replicated here to
avoid importing predictor.py's torch/RNA-FM chain.)

Run: conda run -n kdmsi python web_export/reference/gen_base3_vienna_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import RNA

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "webapp/test/fixtures/base3_vienna_reference.json"


def clean_rna(seq):
    return str(seq).upper().replace("T", "U").strip()


def secondary_structure(seq):
    rna = clean_rna(seq)
    struct, mfe = RNA.fold(rna)
    n = max(len(struct), 1)
    paired = struct.count("(") + struct.count(")")
    unpaired = struct.count(".")
    return [paired / n, unpaired / n], float(mfe)


def target_19_from_window(mrna_window, center=29):
    seq = clean_rna(mrna_window)
    if len(seq) >= 59:
        target = seq[center - 9:center + 10]
    else:
        c = len(seq) // 2
        target = seq[max(0, c - 9):c + 10]
    return target.replace(".", "")


def target_accessibility(mrna_window):
    clean = clean_rna(mrna_window).replace(".", "")
    if len(clean) < 5:
        return [0.0, 0.0, 1.0]
    try:
        _, full_mfe = RNA.fold(clean)
        full_mfe = float(full_mfe)
    except Exception:
        full_mfe = 0.0
    target_19 = target_19_from_window(mrna_window)
    if len(target_19) < 5:
        return [full_mfe, 0.0, 1.0]
    try:
        local_struct, local_mfe = RNA.fold(target_19)
        local_mfe = float(local_mfe)
        unpaired_ratio = local_struct.count(".") / len(local_struct) if len(local_struct) > 0 else 1.0
    except Exception:
        local_mfe = 0.0
        unpaired_ratio = 1.0
    return [full_mfe, local_mfe, unpaired_ratio]


AS_SEQS = ["UUAGAAAGCUGAUGGACCAUA", "GCUACGUAGCUAGCUAGCUAA", "AAAAGGGGCCCCUUUUAAAAG"]
WINDOWS = [
    # 59-nt AUCG window (len>=59 branch)
    "GCACUGAUCCGUAGCUAGCUAGGCAUCGAUCGGAUCGUAGCUAGCUAGCAUCGUAGCUA",
    # 59-char window with leading/trailing dots (dot removal + target core)
    "...GAUCCGUAGCUAGCUAGGCAUCGAUCGGAUCGUAGCUAGCUAGCAUCGUAGCUA...",
    # short window (<59 -> centered target branch)
    "GGCCAUGGCCAUGGCCAUGGCCAUGGCC",
    # degenerate: clean <5 -> [0,0,1]
    "AU.C",
]


def main():
    out = {
        "as": [{"seq": s, "percent": [round(v, 6) for v in p], "energy": round(e, 6)}
               for s in AS_SEQS for (p, e) in [secondary_structure(s)]],
        "windows": [{"window": w, "accessibility": [round(v, 6) for v in target_accessibility(w)]}
                    for w in WINDOWS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote AS={len(out['as'])} windows={len(out['windows'])} -> {OUT}")


if __name__ == "__main__":
    main()
