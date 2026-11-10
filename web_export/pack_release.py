#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repack all browser artifacts + a manifest for a GitHub Release (or R2/Zenodo).

Gathers the exported ONNX models, the ViennaRNA WASM, and zips the raw .pth.tar
checkpoints; computes sha256 + size for each; writes manifest.json that the
browser cache layer uses to fetch + verify + cache (OPFS). Does NOT upload —
publishing is a separate, confirmed `gh release` step.

Run: conda run -n kdmsi python web_export/pack_release.py   (or plain python3)
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "web_export/artifacts"
WASM = REPO / "webapp/wasm/dist/vrna.wasm"
CKPT_DIR = REPO / "liubin_pipeline/potency_prediction/pretrain_models"
OUT_MANIFEST = ART / "manifest.json"
VERSION = "web-assets-v1"
RNAFM_VARIANTS = (
    ("rnafm_t12.onnx", "fp32", True),
    ("rnafm_t12_fp16.onnx", "fp16", False),
    ("rnafm_t12_int8.onnx", "int8", False),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path: Path, kind: str, **extra) -> dict:
    # `asset` is the flat, collision-free name used when uploading (base3/base0
    # folds otherwise share e.g. "0-inter.onnx"); `name` is the original file.
    prefix = f"{kind}_" if kind in ("base3", "base0") else ""
    return {
        "asset": f"{prefix}{path.name}",
        "name": path.name,
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        **extra,
    }


def main():
    assets = []

    # The browser UI currently selects fp32 unconditionally, so a release
    # without rnafm_t12.onnx would produce a manifest that the app cannot use.
    # fp16/int8 remain optional alternatives and are included when present.
    for filename, prec, required in RNAFM_VARIANTS:
        p = ART / filename
        if required and not p.exists():
            raise FileNotFoundError(
                f"Required browser model is missing: {p}. "
                "Run web_export/rnafm_to_onnx.py before packing the release."
            )
        if p.exists():
            assets.append(entry(p, "rnafm", precision=prec))
    for fold in range(10):
        b3 = ART / "base3" / f"{fold}-inter.onnx"
        b0 = ART / "base0" / f"{fold}-inter.onnx"
        if b3.exists():
            assets.append(entry(b3, "base3", fold=f"{fold}-inter"))
        if b0.exists():
            assets.append(entry(b0, "base0", fold=f"{fold}-inter"))

    # ViennaRNA WASM
    if WASM.exists():
        assets.append(entry(WASM, "wasm"))

    # Raw checkpoints -> zip
    ckpts = sorted(CKPT_DIR.rglob("*.pth.tar"))
    if ckpts:
        zip_path = ART / "kdmsi_checkpoints.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for c in ckpts:
                z.write(c, arcname=str(c.relative_to(REPO)))
        assets.append(entry(zip_path, "checkpoints", count=len(ckpts)))

    manifest = {
        "version": VERSION,
        "note": "KDMSi browser inference assets. See the repository README for the upstream RNA-FM checkpoint.",
        "assets": assets,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(a["bytes"] for a in assets)
    print(f"[pack] manifest -> {OUT_MANIFEST}")
    for a in assets:
        print(f"  {a['name']:28s} {a['kind']:12s} {a['bytes']/1e6:8.1f} MB")
    print(f"[pack] {len(assets)} assets, total {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
