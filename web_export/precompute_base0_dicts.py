#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the pickled Base0 modification dicts to a shipped JSON.

fingerprints.npy (RDKit, 101-dim) + gvae.npy (GVAE latent, 56-dim, latin1 pickle)
are keyed by single-char modification symbol. Emits webapp/public/assets/
base0_mod_features.json = { "<symbol>": { "rdkit": [101], "gvae": [56] } } so the
browser can build the 162-dim per-residue vectors without pickled numpy.

Run: conda run -n kdmsi python web_export/precompute_base0_dicts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RD = REPO / "base0model/data/fingerprints.npy"
GV = REPO / "base0model/data/gvae.npy"
OUT = REPO / "webapp/public/assets/base0_mod_features.json"


def main():
    rd = np.load(str(RD), allow_pickle=True).item()
    gv = np.load(str(GV), allow_pickle=True, encoding="latin1").item()
    symbols = sorted(set(rd) | set(gv))
    out = {}
    for s in symbols:
        entry = {}
        if s in rd:
            entry["rdkit"] = np.asarray(rd[s], dtype=float).ravel().tolist()
        if s in gv:
            entry["gvae"] = np.asarray(gv[s], dtype=float).ravel().tolist()
        out[s] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    dims = {s: (len(v.get("rdkit", [])), len(v.get("gvae", []))) for s, v in out.items()}
    print(f"[ok] wrote {len(out)} symbols -> {OUT}")
    print(f"     dims sample: {dict(list(dims.items())[:5])}")


if __name__ == "__main__":
    main()
