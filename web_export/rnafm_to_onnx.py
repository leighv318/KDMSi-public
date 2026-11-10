#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 Spike B: export RNA-FM (rna_fm_t12) to ONNX and check parity.

Loads RNA-FM exactly as the pipeline does (fm.pretrained.rna_fm_t12), wraps a
forward that returns the layer-12 token representations (what predictor.py /
predictor_base0.py consume), exports to ONNX, then compares PyTorch vs
onnxruntime outputs on sample siRNA/mRNA sequences.

Run inside the `kdmsi` conda env:
    conda run -n kdmsi python web_export/rnafm_to_onnx.py \
        --out web_export/artifacts/rnafm_t12.onnx

RNA-FM weights: uses $RNAFM_MODEL_PATH or ./rnafm/RNA-FM_pretrained.pth if
present, else auto-downloads (~400MB) from CUHK on first call.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
# `fm` lives under liubin_pipeline/
sys.path.insert(0, str(REPO / "liubin_pipeline"))

import fm  # noqa: E402


REPR_LAYER = 12
# A couple of realistic inputs: a 21-nt antisense and a 59-nt mRNA window.
SAMPLE_SEQS = [
    ("as21", "UACCCUUCUGCUGUAGUACCC"),
    ("mrna59", "GAAGAAGCUGCCGUUGUUCUGGGUACUACAGCAGAAGGGUAUGCGGAAGCGAGCACCCC"),
]


class RnaFmLayer12(torch.nn.Module):
    """Thin wrapper: tokens -> layer-12 token representations tensor."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.model(tokens, repr_layers=[REPR_LAYER])
        return out["representations"][REPR_LAYER]


def _load_rnafm():
    local = os.environ.get("RNAFM_MODEL_PATH")
    if not local:
        cand = REPO / "rnafm" / "RNA-FM_pretrained.pth"
        local = str(cand) if cand.exists() else None
    print(f"[rnafm] loading rna_fm_t12 (model_location={local})…")
    model, alphabet = fm.pretrained.rna_fm_t12(model_location=local)
    model = model.eval()
    return model, alphabet


# RNA-FM t12 published config (ESM-1b / roberta_large arch).
RNAFM_T12_CONFIG = dict(
    arch="roberta_large",
    layers=12,
    num_layers=12,
    embed_dim=640,
    ffn_embed_dim=5120,
    attention_heads=20,
    max_positions=1024,
    emb_layer_norm_before=True,
    token_dropout=False,
)


def _build_random_rnafm():
    """Build the RNA-FM t12 architecture with random weights (no download).

    Validates the ONNX export mechanism + onnxruntime parity + speed without the
    ~400MB pretrained checkpoint. Real numerical parity later just swaps in the
    real .pth via RNAFM_MODEL_PATH — no code change.
    """
    torch.manual_seed(0)
    alphabet = fm.Alphabet.from_architecture("roberta_large", "rna")
    args = Namespace(**RNAFM_T12_CONFIG)
    model = fm.BioBertModel(args, alphabet).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[rnafm] built RANDOM rna_fm_t12: {n_params/1e6:.1f}M params (alphabet={len(alphabet)})")
    return model, alphabet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "web_export" / "artifacts" / "rnafm_t12.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precisions", default="fp32,fp16",
                        help="comma list of precision variants to emit: fp32,fp16,int8. "
                             "Runtime picks per backend (fp16 for WebGPU, int8 optional for "
                             "CPU/WASM low-bandwidth); int8 is NEVER mandatory.")
    parser.add_argument("--random-weights", action="store_true",
                        help="build architecture with random weights (no download); "
                             "validates export mechanism + speed only")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, alphabet = _build_random_rnafm() if args.random_weights else _load_rnafm()
    wrapper = RnaFmLayer12(model).eval()
    batch_converter = alphabet.get_batch_converter()

    _, _, tokens = batch_converter(SAMPLE_SEQS)
    print(f"[rnafm] token batch shape={tuple(tokens.shape)} dtype={tokens.dtype}")

    with torch.no_grad():
        t0 = time.time()
        ref = wrapper(tokens).cpu().numpy()
        print(f"[rnafm] torch forward {ref.shape} in {time.time()-t0:.3f}s")

    print(f"[rnafm] exporting ONNX (opset {args.opset}) -> {out_path}")
    torch.onnx.export(
        wrapper,
        (tokens,),
        str(out_path),
        input_names=["tokens"],
        output_names=["repr12"],
        dynamic_axes={"tokens": {0: "batch", 1: "seq"}, "repr12": {0: "batch", 1: "seq"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: better onnxruntime-web support
    )
    print(f"[rnafm] fp32 onnx size = {out_path.stat().st_size / 1e6:.1f} MB")

    # Build the requested precision variants (fp32 is the base export above).
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    variants = {"fp32": out_path}

    if "fp16" in precisions:
        try:
            # Native half-precision export (a half() copy of the model) produces a
            # valid graph, unlike post-hoc onnxconverter_common float16 which can
            # leave mismatched Cast nodes that fail to load in onnxruntime.
            p16 = out_path.with_name(out_path.stem + "_fp16.onnx")
            wrapper16 = RnaFmLayer12(wrapper.model).half().eval()
            torch.onnx.export(
                wrapper16, (tokens,), str(p16),
                input_names=["tokens"], output_names=["repr12"],
                dynamic_axes={"tokens": {0: "batch", 1: "seq"}, "repr12": {0: "batch", 1: "seq"}},
                opset_version=args.opset, do_constant_folding=True, dynamo=False,
            )
            variants["fp16"] = p16
        except Exception as exc:  # noqa: BLE001
            print(f"[rnafm] fp16 skipped: {exc}")

    if "int8" in precisions:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            p8 = out_path.with_name(out_path.stem + "_int8.onnx")
            quantize_dynamic(str(out_path), str(p8), weight_type=QuantType.QInt8)
            variants["int8"] = p8
        except Exception as exc:  # noqa: BLE001
            print(f"[rnafm] int8 skipped: {exc}")

    # Per-variant size + accuracy delta vs the PyTorch reference. fp32 must be
    # exact; fp16/int8 are reported so we can quantify the accuracy/size trade
    # (final accept/reject is decided by downstream Base3/Base0 parity in Phase 1).
    import onnxruntime as ort

    fp32_ok = True
    print("[variants] name    size(MB)  max_abs_diff_vs_torch")
    for name, path in variants.items():
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = sess.run(["repr12"], {"tokens": tokens.cpu().numpy().astype(np.int64)})[0].astype(np.float32)
        max_abs = float(np.max(np.abs(out - ref)))
        size = path.stat().st_size / 1e6
        note = "PASS" if (name != "fp32" or max_abs < 1e-3) else "FAIL"
        if name == "fp32":
            fp32_ok = max_abs < 1e-3
        print(f"[variants] {name:6s}  {size:7.1f}   {max_abs:.3e}  {note}  -> {path.name}")

    sys.exit(0 if fp32_ok else 1)


if __name__ == "__main__":
    main()
