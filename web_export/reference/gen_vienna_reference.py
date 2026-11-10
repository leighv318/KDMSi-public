#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 Spike A: ViennaRNA ground-truth reference for the WASM parity check.

Emits JSON with the exact ViennaRNA outputs the pipeline relies on, so the
Emscripten/WASM build can be validated against Python `import RNA`:

  1. RNA.fold(seq) -> (structure, mfe)         [Base3 _secondary_structure /
                                                 _target_accessibility;
                                                 Base0 compute_local_fold_dg]
  2. RNAplfold unpaired probabilities via
     fold_compound(seq, md, OPTION_WINDOW) + probs_window(u=1, PROBS_WINDOW_UP)
                                                [Base0 compute_accessibility]

The probs_window logic mirrors base0model/compute_mrna_thermo.compute_accessibility
(W=min(80,L), span=min(40,L), u=1). Keep in lockstep with that file.

Run inside the `kdmsi` conda env:
    conda run -n kdmsi python web_export/reference/gen_vienna_reference.py \
        --out webapp/test/fixtures/vienna_reference.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import RNA


def clean(seq: str) -> str:
    return str(seq).strip().upper().replace("T", "U")


def fold(seq: str):
    structure, mfe = RNA.fold(clean(seq))
    return {"seq": clean(seq), "structure": structure, "mfe": round(float(mfe), 6)}


def plfold_unpaired(seq: str, u: int = 1):
    """Per-position unpaired probability, mirroring compute_accessibility."""
    s = clean(seq)
    L = len(s)
    unpaired = [None] * L
    md = RNA.md()
    fc = RNA.fold_compound(s, md, RNA.OPTION_WINDOW)

    def callback(v, v_size, i, maxsize, what, data):  # noqa: ANN001
        if (what & RNA.PROBS_WINDOW_UP) and v is not None:
            if len(v) > u and v[u] is not None:
                data[i - 1] = v[u]  # i is 1-based

    fc.probs_window(u, RNA.PROBS_WINDOW_UP, callback, unpaired)
    return {
        "seq": s,
        "u": u,
        "unpaired": [None if x is None else round(float(x), 6) for x in unpaired],
    }


# 21-nt antisense, 19-nt target, and 59-nt mRNA windows drawn from AGT.
FOLD_SEQS = [
    "UACCCUUCUGCUGUAGUACCC",          # AS 21
    "ACUACAGCAGAAGGGUAUG",            # target_19
    "GAAGAAGCUGCCGUUGUUCUGGGUACUACAGCAGAAGGGUAUGCGGAAGCGAGCACCCC",  # mRNA 59
    "GCGCGCGCGCGCGCGC",               # high-GC stress
    "AAAAAAAAAAAAAAAA",               # unstructured
]
PLFOLD_SEQS = [
    "GAAGAAGCUGCCGUUGUUCUGGGUACUACAGCAGAAGGGUAUGCGGAAGCGAGCACCCC",  # mRNA 59
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = {
        "viennarna_version": getattr(RNA, "__version__", "unknown"),
        "fold": [fold(s) for s in FOLD_SEQS],
        "plfold_unpaired": [plfold_unpaired(s) for s in PLFOLD_SEQS],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[ok] wrote ViennaRNA reference (v{data['viennarna_version']}) -> {out}")


if __name__ == "__main__":
    main()
