"""Shared KDMSi-Mod mRNA helpers and the training-cache command."""

import argparse
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import List, Tuple

import numpy as np
import pandas as pd


RNAFM_VALID_CHARS = set("AUCGRYKMWSBDHVN-")
PADDING_CHARS = set(".")


def normalize_raw_mrna(seq) -> str:
    if pd.isna(seq):
        raise ValueError("local_mrna is NaN; expected a matched mRNA row.")
    raw_seq = str(seq).strip().upper().replace("T", "U")
    if raw_seq == "" or raw_seq == "NAN":
        raise ValueError("local_mrna is empty/NAN; expected a matched mRNA row.")
    return raw_seq


def prepare_clean_sequence(raw_seq: str, row_label: str) -> Tuple[str, List[int]]:
    """Return clean sequence sent to RNA-FM and raw positions for each token."""
    bad_chars = set(raw_seq) - RNAFM_VALID_CHARS - PADDING_CHARS
    if bad_chars:
        raise ValueError(
            f"Invalid mRNA chars {bad_chars} for RNAFM_ind={row_label}, seq={raw_seq}"
        )

    valid_positions = [k for k, ch in enumerate(raw_seq) if ch in RNAFM_VALID_CHARS]
    clean_seq = "".join(raw_seq[k] for k in valid_positions)
    if len(clean_seq) == 0:
        raise ValueError(f"Empty mRNA after removing '.' padding for RNAFM_ind={row_label}")
    return clean_seq, valid_positions


def clean_seq_to_dna(seq: object) -> str:
    """Keep A/C/G/T/U only and convert RNA to DNA alphabet."""
    if pd.isna(seq):
        return ""
    s = str(seq).strip().upper()
    s = re.sub(r"[^ACGTU]", "", s)
    return s.replace("U", "T")


def revcomp_dna(seq: str) -> str:
    tab = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tab)[::-1].upper()


def strip_terminal_t(seq: str) -> str:
    """Remove likely terminal DNA T overhangs from both ends."""
    return re.sub(r"^T+|T+$", "", seq)


def candidate_queries(ori_ss: object, ori_as: object, min_len: int = 16) -> List[Tuple[str, str]]:
    """
    Generate search candidates.
    Prefer ori_ss because it should be same orientation as mRNA target sequence.
    Use reverse complement of ori_as as backup.
    """
    ss = clean_seq_to_dna(ori_ss)
    asq = clean_seq_to_dna(ori_as)
    rc_as = revcomp_dna(asq) if asq else ""

    raw_candidates: List[Tuple[str, str]] = []
    for label, s in [("ori_ss_full", ss), ("rc_ori_as_full", rc_as)]:
        if not s:
            continue
        raw_candidates.append((label, s))
        st = strip_terminal_t(s)
        if st and st != s:
            raw_candidates.append((label + "_strip_terminal_T", st))
        # Common siRNA core choices. These are backups only.
        for L in (23, 21, 20, 19):
            if len(s) > L:
                raw_candidates.append((f"{label}_first{L}", s[:L]))
                raw_candidates.append((f"{label}_last{L}", s[-L:]))
                mid_start = max((len(s) - L) // 2, 0)
                raw_candidates.append((f"{label}_mid{L}", s[mid_start:mid_start + L]))

    # Deduplicate while preserving order; discard too-short candidates.
    seen = set()
    out: List[Tuple[str, str]] = []
    for label, q in raw_candidates:
        if len(q) < min_len:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append((label, q))
    return out


def find_all(seq: str, query: str) -> List[int]:
    """Return all start positions, including overlapping matches."""
    if not seq or not query:
        return []
    starts = []
    start = 0
    while True:
        idx = seq.find(query, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts


def main():
    parser = argparse.ArgumentParser(description="Generate the row-aligned KDMSi-Mod RNA-FM cache.")
    parser.add_argument("--datafile", required=True, type=Path, help="Training CSV containing local_mrna")
    parser.add_argument("--out-stack", required=True, type=Path, help="Output .npy file")
    parser.add_argument("--max-len", type=int, default=65)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model-location", help="Path to RNA-FM_pretrained.pth")
    args = parser.parse_args()
    if args.max_len < 1 or args.batch_size < 1:
        parser.error("--max-len and --batch_size must be positive")

    local_mrnas = pd.read_csv(args.datafile, usecols=["local_mrna"])["local_mrna"]
    if local_mrnas.empty:
        parser.error("training CSV has no data rows")
    prepared = [
        prepare_clean_sequence(normalize_raw_mrna(raw), str(i))
        for i, raw in enumerate(local_mrnas)
    ]

    # The vendored package is imported as fm by its own modules.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liubin_pipeline"))
    import fm
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, alphabet = fm.pretrained.rna_fm_t12(args.model_location)
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()

    output = args.out_stack
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp_dir:
        temporary = Path(temp_dir) / "cache.npy"
        stack = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(len(prepared), args.max_len, 640)
        )
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start:start + args.batch_size]
            _, _, tokens = converter([(str(start + j), seq) for j, (seq, _) in enumerate(batch)])
            with torch.no_grad():
                encoded = model(tokens.to(device), repr_layers=[12])["representations"][12]
            embeddings = encoded.detach().cpu().numpy()
            for j, (seq, positions) in enumerate(batch):
                positions = np.asarray(positions, dtype=int)
                keep = positions < args.max_len
                stack[start + j] = 0
                stack[start + j, positions[keep], :] = embeddings[j, 1:len(seq) + 1, :][keep]
            print(f"[RNA-FM] {start + len(batch)}/{len(prepared)}", flush=True)
        stack.flush()
        del stack
        temporary.replace(output)
    print(f"Saved RNA-FM cache to {output}")


if __name__ == "__main__":
    main()
