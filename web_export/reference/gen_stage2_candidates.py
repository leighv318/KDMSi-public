#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stdlib-only ground-truth generator for Stage 2 candidate generation.

Faithfully mirrors ``liubin_pipeline/potency_prediction/design_modified.py``
functions ``_clean_rna`` / ``_reverse_complement`` / ``_read_fasta`` /
``_base3_candidates`` WITHOUT importing torch/ViennaRNA/pandas, so the browser
(TypeScript) port can be validated against it in any environment.

Any change here must stay in lockstep with design_modified.py. See that file
for the authoritative implementation.

Usage:
    python3 web_export/reference/gen_stage2_candidates.py \
        --fasta examples/AGT.fasta --gene AGT \
        --out webapp/test/fixtures/stage2_AGT.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

CORE_SS_LEN = 19


def _clean_rna(sequence: str) -> str:
    return str(sequence).strip().upper().replace("T", "U")


def _reverse_complement(sequence: str) -> str:
    sequence = _clean_rna(sequence)
    invalid = sorted(set(sequence).difference("AUCG"))
    if invalid:
        raise ValueError(f"Cannot reverse-complement symbols {invalid} in {sequence}")
    return sequence.translate(str.maketrans("AUCG", "UAGC"))[::-1]


def _read_fasta(path: str):
    records = []
    name = None
    pieces = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, _clean_rna("".join(pieces))))
                name = line[1:].split()[0]
                pieces = []
            else:
                pieces.append(line)
    if name is not None:
        records.append((name, _clean_rna("".join(pieces))))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def base3_candidates(target_fasta: str, gene: str):
    rows = []
    skipped = 0
    for transcript_id, transcript in _read_fasta(target_fasta):
        if len(transcript) < 59:
            continue
        for window_start in range(len(transcript) - 59 + 1):
            local_mrna = transcript[window_start:window_start + 59]
            if set(local_mrna).difference("AUCG"):
                skipped += 1
                continue
            target_21 = local_mrna[20:41]
            sense_19 = target_21[2:]
            antisense_21 = _reverse_complement(target_21)
            rows.append({
                "gene": gene,
                "NM_id": transcript_id,
                "AS": antisense_21,
                "SS": sense_19,
                "local_mrna": local_mrna,
                "mrna_len": len(transcript),
                "ss_pos(_prefer_middle)": window_start + 23,
                "ss_GC%(_prefer_30-60)": round(
                    100.0 * (sense_19.count("G") + sense_19.count("C")) / CORE_SS_LEN, 3
                ),
                "ss_GU%(_prefer_30-65)": round(
                    100.0 * (sense_19.count("G") + sense_19.count("U")) / CORE_SS_LEN, 3
                ),
                "as_immune_GUCCUUCAA|UGUGU(_prefer_N)": "{}|{}".format(
                    "Y" if "GUCCUUCAA" in antisense_21 else "N",
                    "Y" if "UGUGU" in antisense_21 else "N",
                ),
                "ss_start_AA(_prefer_Y)": "Y" if sense_19.startswith("AA") else "N",
                "4_same_nts(_prefer_N)": "Y" if any(
                    base * 4 in antisense_21 for base in "AUCG"
                ) else "N",
            })
    if not rows:
        raise ValueError(f"No valid 59-nt windows from {target_fasta}")
    if skipped:
        print(f"[design] skipped {skipped} window(s) containing non-A/U/C/G symbols")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--gene", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = base3_candidates(args.fasta, args.gene)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {len(rows)} candidates -> {out_path}")


if __name__ == "__main__":
    main()
