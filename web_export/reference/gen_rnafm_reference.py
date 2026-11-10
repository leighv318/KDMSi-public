#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for the RNA-FM tokenizer + embedding builder (TS parity).

Two things are validated:
  1. Tokenization — uses the REAL RNA-FM Alphabet
     (fm.data.Alphabet.from_architecture("roberta_large","rna")) batch_converter
     to produce token-id matrices ([cls] residues [eos], right-padded).
  2. End-to-end embeddings — runs the exported fp32 RNA-FM ONNX
     (web_export/artifacts/rnafm_t12.onnx) via onnxruntime, then replicates the
     exact slice/scatter of predictor.py (_rnafm_as_batch / _rnafm_mrna_batch)
     and predictor_base0.py (_rnafm_embeddings -> 65). Both the Python reference
     and the TS test run the SAME ONNX graph, so residual diffs are tiny.

To keep the fixture small we store a few "probe" dims across ALL output
positions (fully validates positional slice/scatter; model per-dim fidelity was
proven separately in the ONNX export spike).

Run: conda run -n kdmsi python web_export/reference/gen_rnafm_reference.py
(needs the fp32 rnafm_t12.onnx on disk; no torch/RNA-FM weights required.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "liubin_pipeline"))
from fm.data import Alphabet  # noqa: E402

ONNX = REPO / "web_export/artifacts/rnafm_t12.onnx"
OUT = REPO / "webapp/test/fixtures/rnafm_reference.json"
PROBE_DIMS = [0, 1, 2, 159, 320, 639]

RNAFM_VALID = set("AUCGRYKMSWBDHVN-")
RNAFM_PAD = set(".")

AS_SEQS = ["UUAGAAAGCUGAUGGACCAUA", "GCUACGUAGCUAGCUAGCU"]  # 21, 19 nt (AUCG)
MRNA3_SEQS = [
    "GCACUGAUCCGUAGCUAGCUAGGCAUCGAUCGGAUCGUAGCUAGCUAGCAUCGUAGCUA",     # 59, no dots
    "...GAUCCGUAGCUAGCUAGGCAUCGAUCGGAUCGUAGCUAGCUAGCAUCGUAGCUA...",   # 59 with leading/trailing dots
]
MRNA0_SEQS = [
    "GCACUGAUCCGUAGCUAGCUAGGCAUCGAUCGGAUCGUAGCUA",                     # 43 -> pad to 65
    "GCACUGAUCCGUAGCUAGCUAGGCAUCGAUCGGAUCGUAGCUAGCUAGCAUCGUAGCUAGCUAGC",  # 65
]


def clean_rna(s: str) -> str:
    return s.strip().upper().replace("T", "U")


def prepare_mrna(raw: str, fixed_len=None):
    """Replicates predictor._rnafm_mrna_batch preprocessing + Base0 fixed_len."""
    raw_seq = clean_rna(raw)
    bad = set(raw_seq) - RNAFM_VALID - RNAFM_PAD
    if bad:
        raise ValueError(f"mRNA window unsupported chars {sorted(bad)}: {raw_seq}")
    valid_positions = [k for k, ch in enumerate(raw_seq) if ch in RNAFM_VALID]
    clean_seq = "".join(raw_seq[k] for k in valid_positions)
    if not clean_seq:
        clean_seq, valid_positions = "N", [0]
    out_len = fixed_len if fixed_len is not None else len(raw_seq)
    return clean_seq, valid_positions, out_len


def main():
    if not ONNX.exists():
        raise SystemExit(f"missing {ONNX}; export RNA-FM fp32 ONNX first")
    alphabet = Alphabet.from_architecture("roberta_large", "rna")
    bc = alphabet.get_batch_converter()
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    def tokens_np(clean_seqs):
        _, _, toks = bc([(str(i), s) for i, s in enumerate(clean_seqs)])
        return toks.numpy().astype(np.int64)

    def repr12(clean_seqs):
        toks = tokens_np(clean_seqs)
        out = sess.run(["repr12"], {"tokens": toks})[0]  # [B, L+2, 640] float32
        return toks, out

    fixture = {"probeDims": PROBE_DIMS, "tokenize": [], "as": [], "mrna3": [], "mrna0": []}

    # --- AS group (drop BOS, slice [1:len+1], output length = AS length) ---
    as_clean = [clean_rna(s) for s in AS_SEQS]
    for s in as_clean:
        bad = set(s) - set("AUCG")
        if bad:
            raise ValueError(f"AS unsupported bases {sorted(bad)}: {s}")
    toks, emb = repr12(as_clean)
    fixture["tokenize"].append({"seqs": as_clean, "dims": list(toks.shape), "tokens": toks.tolist()})
    for j, s in enumerate(as_clean):
        e = emb[j, 1:len(s) + 1, :]  # [L, 640]
        fixture["as"].append({
            "seq": AS_SEQS[j], "outputLen": len(s),
            "probe": np.round(e[:, PROBE_DIMS].astype(float), 5).tolist(),
        })

    # --- mRNA groups (scatter into zeros at valid positions) ---
    def do_mrna(raw_list, fixed_len, key):
        preps = [prepare_mrna(r, fixed_len) for r in raw_list]
        clean_seqs = [p[0] for p in preps]
        toks, emb = repr12(clean_seqs)
        fixture["tokenize"].append({"seqs": clean_seqs, "dims": list(toks.shape), "tokens": toks.tolist()})
        for j, (clean_seq, valid_positions, out_len) in enumerate(preps):
            e = emb[j, 1:len(clean_seq) + 1, :]  # [cleanLen, 640]
            full = np.zeros((out_len, emb.shape[2]), dtype=np.float32)
            usable = min(len(valid_positions), e.shape[0])
            if usable:
                full[np.asarray(valid_positions[:usable]), :] = e[:usable]
            fixture[key].append({
                "seq": raw_list[j], "outputLen": out_len,
                **({"fixedLen": fixed_len} if fixed_len is not None else {}),
                "probe": np.round(full[:, PROBE_DIMS].astype(float), 5).tolist(),
            })

    do_mrna(MRNA3_SEQS, None, "mrna3")
    do_mrna(MRNA0_SEQS, 65, "mrna0")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"[ok] wrote AS={len(fixture['as'])} mRNA3={len(fixture['mrna3'])} "
          f"mRNA0={len(fixture['mrna0'])} tokenize={len(fixture['tokenize'])} -> {OUT}")


if __name__ == "__main__":
    main()
