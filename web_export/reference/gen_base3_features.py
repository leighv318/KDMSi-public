#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for Base3 pure-TS hand features (no RNA-FM / no ViennaRNA).

Replicates predictor.py: _score_pssm, _gibbs_energy, _gc_content, _gc_stretch,
_kmer_percents (TRI/DI/single with BASES4=[A,G,C,U] product order). PSSM uses
fold 0-inter from base3/pssm.json. Emits per-candidate expected values for the
TS parity test.

Run: conda run -n kdmsi python web_export/reference/gen_base3_features.py
"""
from __future__ import annotations

import csv
import json
import math
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CANDS = REPO / "webapp/test/fixtures/stage2_AGT.csv"
PSSM_JSON = REPO / "webapp/public/assets/base3_pssm.json"
OUT = REPO / "webapp/test/fixtures/base3_features_AGT.json"
N = 60

BASES4 = ["A", "G", "C", "U"]
TRI_KMERS = ["".join(p) for p in product(BASES4, repeat=3)]
DI_KMERS = ["".join(p) for p in product(BASES4, repeat=2)]
GIBBS = {
    "AA": -0.93, "AU": -1.10, "AG": -2.35, "AC": -2.11,
    "UA": -1.33, "UU": -0.93, "UG": -2.24, "UC": -2.08,
    "GA": -2.35, "GU": -2.24, "GG": -3.26, "GC": -3.42,
    "CA": -2.11, "CU": -2.08, "CG": -2.36, "CC": -3.26,
}
NT_ORDER = {"A": 0, "G": 1, "C": 2, "U": 3, "N": 0}


def clean(s):
    return str(s).strip().upper().replace("T", "U")


def score_pssm(pssm, seq):  # pssm [4][21] rows A,G,C,U
    seq = clean(seq)
    ncol = len(pssm[0])
    scores = [pssm[NT_ORDER.get(nt, 0)][i] for i, nt in enumerate(seq[:ncol])]
    return float(sum(-math.log2(max(float(v), 1e-10)) for v in scores))


def gibbs_energy(seq):
    seq = clean(seq)[:19]
    values = [GIBBS.get(seq[i:i + 2], 0.0) for i in range(max(len(seq) - 1, 0))]
    while len(values) < 18:
        values.append(0.0)
    values = values[:18]
    total = round(sum(values), 3)
    end_diff = round(values[0] - values[-1], 3) if values else 0.0
    return values + [total, end_diff]


def gc_content(seq):
    seq = clean(seq)
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def gc_stretch(seq):
    seq = clean(seq)
    best = cur = 0
    for nt in seq:
        if nt in "GC":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best / max(len(seq), 1)


def kmers(seq):
    seq = clean(seq)
    n = len(seq)
    tri = [seq.count(k) / max(n - 2, 1) for k in TRI_KMERS]
    di = [seq.count(k) / max(n - 1, 1) for k in DI_KMERS]
    single = [seq.count(b) / max(n, 1) for b in BASES4]
    return tri, di, single


def main():
    pssm = json.loads(PSSM_JSON.read_text())["0-inter"]
    with CANDS.open() as f:
        rows = list(csv.DictReader(f))[:N]
    out = []
    for r in rows:
        AS = r["AS"]
        tri, di, single = kmers(AS)
        out.append({
            "AS": clean(AS),
            "pssm_score_0inter": score_pssm(pssm, AS),
            "gibbs": gibbs_energy(AS),
            "gc_content": gc_content(AS),
            "gc_stretch": gc_stretch(AS),
            "tri": tri, "di": di, "single": single,
        })
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote {len(out)} candidates -> {OUT}")


if __name__ == "__main__":
    main()
