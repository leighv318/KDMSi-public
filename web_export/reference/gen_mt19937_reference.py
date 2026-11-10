#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth for the CPython-compatible MT19937 port.

Emits random.Random(seed).random() sequences for several seeds so the TS
MT19937 can be validated byte-exact. Seed 126 is the pipeline default.

Run: python3 web_export/reference/gen_mt19937_reference.py   (stdlib only)
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "webapp/test/fixtures/mt19937_reference.json"
SEEDS = [126, 0, 1, 42, 2147483647, 123456789]
COUNT = 2000


def main():
    out = {}
    for s in SEEDS:
        r = random.Random(s)  # seed ONCE, then draw COUNT times
        out[str(s)] = [r.random() for _ in range(COUNT)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] wrote {len(SEEDS)} seeds x {COUNT} draws -> {OUT}")


if __name__ == "__main__":
    main()
