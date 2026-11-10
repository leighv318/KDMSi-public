#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the 10 Base0 fold models (Base0_mrna_full_no_se) to ONNX.

Wraps the dict-based forward into positional tensors (22 inputs; see plan Base0
contract). TS assembles these; the ONNX graph carries conv/attention/cross-attn/
BatchNorm/DNN. Run: conda run -n kdmsi python web_export/base0_to_onnx.py
Outputs: web_export/artifacts/base0/{fold}.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
# base0model/model.py uses non-relative `from load_data import ...` / `from
# model_layers import ...`, so put base0model/ on the path (as predictor_base0 does).
sys.path.insert(0, str(REPO / "base0model"))

import model as base0_models  # noqa: E402  (resolves to base0model/model.py)

CKPT_DIR = REPO / "liubin_pipeline/potency_prediction/pretrain_models/Base0_mrna_full_no_se"
OUT_DIR = REPO / "web_export/artifacts/base0"
FOLDS = [f"{i}-inter" for i in range(10)]

# (name, shape-without-batch, dtype, kind) — kind: 'f'=randn float, 'e'=energy idx, 'c'=cell idx
INPUT_SPEC = [
    ("cm_as_embedding", (25, 162), "f"),
    ("as_mod_aux", (25, 15), "f"),
    ("cm_ss_embedding", (25, 162), "f"),
    ("ss_mod_aux", (25, 15), "f"),
    ("mrna_rnafm", (65, 640), "f"),
    ("as_enenrgy_index", (25,), "e"),
    ("ss_enenrgy_index", (25,), "e"),
    ("as_tri_nt_percent", (125, 1), "f"),
    ("as_di_nt_percent", (25, 1), "f"),
    ("as_single_nt_percent", (28, 1), "f"),
    ("as_gc_content", (1, 1), "f"),
    ("as_gc_sterch", (1, 1), "f"),
    ("as_pssm_score", (1, 1), "f"),
    ("ss_tri_nt_percent", (125, 1), "f"),
    ("ss_di_nt_percent", (25, 1), "f"),
    ("ss_single_nt_percent", (28, 1), "f"),
    ("ss_gc_content", (1, 1), "f"),
    ("ss_gc_sterch", (1, 1), "f"),
    ("ss_pssm_score", (1, 1), "f"),
    ("concentration_feature", (2,), "f"),
    ("mrna_thermo", (5,), "f"),
    ("cell_type_idx", (1,), "c"),
]
INPUT_NAMES = [n for n, _, _ in INPUT_SPEC]


class Base0Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model({name: args[i] for i, name in enumerate(INPUT_NAMES)})


def _dummy(batch=2):
    torch.manual_seed(0)
    out = []
    for _, shape, kind in INPUT_SPEC:
        if kind == "f":
            out.append(torch.randn(batch, *shape))
        elif kind == "e":
            out.append(torch.randint(0, 32, (batch, *shape)).float())
        else:  # cell idx
            out.append(torch.randint(0, 19, (batch, *shape)).to(torch.int64))
    return tuple(out)


def _load(fold):
    ckpt = CKPT_DIR / fold / "model_best.pth.tar"
    if not ckpt.exists():
        ckpt = CKPT_DIR / fold / f"model_{fold}.pth.tar"
    model = base0_models.Base0_mrna_full_no_se(
        device="cpu", dp=0.1, as_ts=3, ss_ts=3, cell_embed_dim=8
    ).float()
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval(), ckpt


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort

    dummy = _dummy()
    all_ok = True
    for fold in FOLDS:
        model, ckpt = _load(fold)
        wrapper = Base0Wrapper(model).eval()
        with torch.no_grad():
            ref = wrapper(*dummy).cpu().numpy()
        out_path = OUT_DIR / f"{fold}.onnx"
        torch.onnx.export(
            wrapper, dummy, str(out_path),
            input_names=INPUT_NAMES, output_names=["score"],
            dynamic_axes={n: {0: "batch"} for n in INPUT_NAMES} | {"score": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        feeds = {n: dummy[i].numpy() for i, n in enumerate(INPUT_NAMES)}
        got = sess.run(["score"], feeds)[0]
        max_abs = float(np.max(np.abs(got - ref)))
        ok = max_abs < 1e-4
        all_ok = all_ok and ok
        print(f"[base0] {fold}: {ckpt.name} size={out_path.stat().st_size/1e6:.1f}MB "
              f"max_abs_diff={max_abs:.2e} {'PASS' if ok else 'FAIL'}")
    print(f"[base0] {'ALL PASS' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
