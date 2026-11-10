#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the 10 Base3 fold models (Base3_cross_access_v2_src_no_se) to ONNX.

Wraps the model's dict-based forward into a positional-tensor forward so it can
be exported and run in onnxruntime-web. TS builds these 13 input tensors (see
plan Base3 input contract); the ONNX graph carries attention/BatchNorm/DNN.

Run inside kdmsi env:
    conda run -n kdmsi python web_export/base3_to_onnx.py
Outputs: web_export/artifacts/base3/{fold}.onnx  (fold = 0-inter … 9-inter)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Base3model"))  # model_attention uses `from model_layers import *`

import model_attention  # noqa: E402

CKPT_DIR = REPO / "liubin_pipeline/potency_prediction/pretrain_models/Base3_cross_access_v2_src_no_se"
OUT_DIR = REPO / "web_export/artifacts/base3"
FOLDS = [f"{i}-inter" for i in range(10)]

# (name, per-sample-shape-without-batch) in the exact order the wrapper accepts.
INPUT_SPEC = [
    ("rnafm_encode", (21, 640)),
    ("rnafm_encode_mrna", (59, 640)),
    ("sirna_gibbs_energy", (20, 1)),
    ("pssm_score", (1, 1)),
    ("gc_sterch", (1, 1)),
    ("sirna_second_percent", (2, 1)),
    ("sirna_second_energy", (1, 1)),
    ("tri_nt_percent", (64, 1)),
    ("di_nt_percent", (16, 1)),
    ("single_nt_percent", (4, 1)),
    ("gc_content", (1, 1)),
    ("target_accessibility", (3, 1)),
    ("source_id", (1, 1)),
]
INPUT_NAMES = [n for n, _ in INPUT_SPEC]


class Base3Wrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rnafm_encode, rnafm_encode_mrna, sirna_gibbs_energy, pssm_score,
                gc_sterch, sirna_second_percent, sirna_second_energy, tri_nt_percent,
                di_nt_percent, single_nt_percent, gc_content, target_accessibility, source_id):
        inputs = {
            "rnafm_encode": rnafm_encode,
            "rnafm_encode_mrna": rnafm_encode_mrna,
            "sirna_gibbs_energy": sirna_gibbs_energy,
            "pssm_score": pssm_score,
            "gc_sterch": gc_sterch,
            "sirna_second_percent": sirna_second_percent,
            "sirna_second_energy": sirna_second_energy,
            "tri_nt_percent": tri_nt_percent,
            "di_nt_percent": di_nt_percent,
            "single_nt_percent": single_nt_percent,
            "gc_content": gc_content,
            "target_accessibility": target_accessibility,
            "source_id": source_id,
        }
        return self.model(inputs)


def _dummy(batch=2):
    torch.manual_seed(0)
    tensors = []
    for name, shape in INPUT_SPEC:
        if name == "source_id":
            t = torch.full((batch, *shape), 3.0)  # source always 3 (padding_idx -> zero embed)
        else:
            t = torch.randn(batch, *shape)
        tensors.append(t)
    return tuple(tensors)


def _load(fold: str):
    ckpt = CKPT_DIR / fold / f"model_{fold}.pth.tar"
    if not ckpt.exists():
        alt = CKPT_DIR / fold / "model_best.pth.tar"
        ckpt = alt if alt.exists() else ckpt
    model = model_attention.Base3_cross_access_v2_src_no_se(dp=0.0, device="cpu").float()
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
        wrapper = Base3Wrapper(model).eval()
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
        print(f"[base3] {fold}: {ckpt.name} size={out_path.stat().st_size/1e6:.1f}MB "
              f"max_abs_diff={max_abs:.2e} {'PASS' if ok else 'FAIL'}")

    print(f"[base3] {'ALL PASS' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
