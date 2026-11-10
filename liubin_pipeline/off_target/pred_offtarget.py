#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .Smith_Waterman import Homo_offtarget_Based_on_substr
from .Smith_Waterman import LocalAligner_multi_findings_based_on_ss


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REFSEQ_RE = re.compile(r"(?:NM|NR|XM|XR)_\d+(?:\.\d+)?", re.IGNORECASE)


def normalize_gene_arg(g):
    """Return a plain string gene name from args.g, which may be a list."""
    if isinstance(g, list):
        if len(g) == 1:
            return str(g[0])
        return "_".join(map(str, g))
    return str(g)


def _normalize_rna(sequence):
    if pd.isna(sequence):
        raise ValueError("Off-target sequence is empty")
    return str(sequence).strip().upper().replace("T", "U")


def _resolve_candidate_path(path):
    candidate = Path(path).expanduser()
    alternatives = [candidate]
    if not candidate.is_absolute():
        alternatives.extend([
            Path.cwd() / candidate,
            PIPELINE_DIR / candidate,
            PIPELINE_DIR / "output" / "sirnacm" / candidate,
        ])
    for alternative in alternatives:
        if alternative.exists():
            return alternative.resolve()
    raise FileNotFoundError("Modified-siRNA input not found: {}".format(path))


def _resolve_modified_gene(args, data):
    genes = [str(g) for g in (args.g or []) if str(g).strip()]
    if len(genes) == 1:
        return genes[0]
    if len(genes) > 1:
        raise ValueError("sirnacm off-target analysis requires exactly one --g target gene")

    if "target_gene" in data.columns:
        values = data["target_gene"].dropna().astype(str).str.strip()
        values = [value for value in values.unique() if value]
        if len(values) == 1:
            return values[0]
    raise ValueError(
        "Cannot identify the true target for sirnacm off-target analysis. "
        "Pass exactly one target with --g GENE or provide one unique target_gene column."
    )


def _candidate_contexts(args):
    """Yield target gene, output label, data and naked duplex column names."""
    if getattr(args, "filter", "sirna") == "sirnacm":
        if not args.sirnacm_path:
            raise ValueError("--sirnacm_path is required for sirnacm off-target analysis")
        input_path = _resolve_candidate_path(args.sirnacm_path)
        data = pd.read_csv(input_path)
        gene = _resolve_modified_gene(args, data)
        required = {"ori_as", "ori_ss"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(
                "sirnacm off-target analysis requires naked-sequence columns {}; missing {}".format(
                    sorted(required), missing
                )
            )
        yield gene, input_path.stem, data, "ori_as", "ori_ss", "Base0_avg"
        return

    if not args.g:
        raise ValueError("--g is required for unmodified-siRNA off-target analysis")
    for raw_gene in args.g:
        gene = normalize_gene_arg(raw_gene)
        input_path = PIPELINE_DIR / "output" / "sirna" / (gene + ".csv")
        if not input_path.exists():
            raise FileNotFoundError("Potency result not found: {}".format(input_path))
        data = pd.read_csv(input_path)
        required = {"AS", "SS"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError("Unmodified-siRNA input is missing columns {}".format(missing))
        yield gene, gene, data, "AS", "SS", "Base3_avg"


def _rank_top_candidates(data, args, score_column):
    ranked = data.copy()
    if score_column in ranked.columns:
        ranked = ranked.sort_values(score_column, ascending=False, kind="mergesort")
    elif args.k != -1:
        raise ValueError(
            "Top-k off-target analysis requires potency column '{}'. "
            "Run method 1 first or include it in --m.".format(score_column)
        )

    if args.k == -1:
        k = len(ranked)
    elif args.k <= 0:
        raise ValueError("--k must be -1 or a positive integer")
    else:
        k = min(int(args.k), len(ranked))
    return ranked.iloc[:k].reset_index(drop=True)


def _accession_keys(value):
    value = str(value).strip().split()[0].upper()
    keys = {value, value.split(".")[0]}
    for accession in REFSEQ_RE.findall(value):
        accession = accession.upper()
        keys.update([accession, accession.split(".")[0]])
    return keys


def _read_fasta_target(path):
    accessions = set()
    sequences = []
    current = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if current:
                    sequences.append(_normalize_rna("".join(current)))
                    current = []
                accessions.update(_accession_keys(line[1:]))
            else:
                current.append(line.strip())
    if current:
        sequences.append(_normalize_rna("".join(current)))
    return accessions, sequences


def _target_fasta(args, gene):
    explicit = getattr(args, "base0_target_fasta", None)
    if getattr(args, "filter", "sirna") == "sirnacm" and explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            candidates = [Path.cwd() / path, PIPELINE_DIR / path]
            path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        path = PIPELINE_DIR / "output" / "mrna" / (gene + ".fasta")
    if not path.exists():
        raise FileNotFoundError(
            "Target FASTA is required to exclude true on-target transcripts: {}".format(path)
        )
    return path.resolve()


def _target_accessions(args, gene):
    path = _target_fasta(args, gene)
    accessions, sequences = _read_fasta_target(path)
    refseq_accessions = {key for key in accessions if REFSEQ_RE.fullmatch(key)}
    if not refseq_accessions:
        raise ValueError(
            "No RefSeq transcript accession was found in target FASTA headers: {}".format(path)
        )
    return accessions, sequences, path


def _filter_utr_reference(utr_seq, gene, target_accessions, target_sequences):
    gene_upper = str(gene).strip().upper()
    keep = []
    for value in utr_seq["gene"]:
        keys = _accession_keys(value)
        keep.append(gene_upper not in keys and keys.isdisjoint(target_accessions))
    for index, should_keep in enumerate(keep):
        if not should_keep:
            continue
        utr = _normalize_rna(utr_seq["utr_seqs"].iloc[index])
        if utr and any(utr in target for target in target_sequences):
            keep[index] = False
    filtered = utr_seq.loc[keep].reset_index(drop=True)
    print("[off-target] UTR reference: kept {}, excluded {} on-target record(s)".format(
        len(filtered), len(utr_seq) - len(filtered)
    ))
    return filtered


def _filter_transcript_reference(nm_seq, args, target_accessions):
    allowed = {str(prefix).strip().upper() for prefix in args.obj if str(prefix).strip()}
    if not allowed:
        raise ValueError("--obj must contain at least one RefSeq prefix")

    keep = []
    excluded_target = 0
    for value in nm_seq["nm_id"]:
        keys = _accession_keys(value)
        accession = str(value).strip().split()[0].upper()
        prefix = accession[:2]
        is_target = not keys.isdisjoint(target_accessions)
        if is_target:
            excluded_target += 1
        keep.append(prefix in allowed and not is_target)
    filtered = nm_seq.loc[keep].reset_index(drop=True)
    print(
        "[off-target] transcript reference: kept {} ({}) and excluded {} on-target record(s)".format(
            len(filtered), ",".join(sorted(allowed)), excluded_target
        )
    )
    return filtered


def _detail_path_for_output(path):
    try:
        return "./" + str(Path(path).resolve().relative_to(PIPELINE_DIR))
    except ValueError:
        return str(Path(path).resolve())


def _duplex_key(row, as_column, ss_column):
    return _normalize_rna(row[as_column]), _normalize_rna(row[ss_column])


def mirna_like_offtarget(args, UTR_SEQ, NM_INFO=None):
    output_root = PIPELINE_DIR / "output" / "mirna-like_offtarget"
    output_root.mkdir(parents=True, exist_ok=True)

    for gene, label, data, as_column, ss_column, score_column in _candidate_contexts(args):
        print("mirna-like off target predicting {}".format(label))
        data = _rank_top_candidates(data, args, score_column)
        target_ids, target_sequences, fasta_path = _target_accessions(args, gene)
        reference = _filter_utr_reference(UTR_SEQ, gene, target_ids, target_sequences)
        print("[off-target] excluding target transcript(s) from {}".format(fasta_path))

        detail_dir = output_root / label
        detail_dir.mkdir(parents=True, exist_ok=True)
        cache = {}
        results = []
        for _, row in data.iterrows():
            key = _duplex_key(row, as_column, ss_column)
            if key not in cache:
                detail_path = detail_dir / ("duplex_{}_utr.csv".format(len(cache)))
                counts = cal_utr_paires_num(key[1], reference, detail_path)
                cache[key] = counts + (_detail_path_for_output(detail_path),)
            results.append(cache[key])

        columns = ["utr_num", "mer_8", "mer_7_m8", "mer_7_a1", "utr_info_path"]
        for index, column in enumerate(columns):
            data[column] = [result[index] for result in results]
        data.to_csv(output_root / (label + ".csv"), index=False)
        print("[off-target] saved {} top candidate(s); {} unique naked duplex(es)".format(
            len(data), len(cache)
        ))


def homo_substr_offtarget(args, NM_SEQ):
    _run_homology_offtarget(args, NM_SEQ, method="substr")


def homo_sw_offtarget(args, NM_SEQ):
    _run_homology_offtarget(args, NM_SEQ, method="sw")


def _run_homology_offtarget(args, NM_SEQ, method):
    if method == "substr":
        output_name = "homo_substr_offtarget"
        count_column = "homo_substr_num"
        path_column = "homo_substr_info_path"
        suffix = "homo_substr"
        calculator = cal_homo_substr_num
        description = "substring"
    else:
        output_name = "homo_sw_offtarget"
        count_column = "homo_sw_num"
        path_column = "homo_sw_info_path"
        suffix = "homo_sw"
        calculator = cal_homo_sw_num
        description = "Smith-Waterman"

    output_root = PIPELINE_DIR / "output" / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    for gene, label, data, as_column, ss_column, score_column in _candidate_contexts(args):
        print("homology off target by {} predicting {}".format(description, label))
        data = _rank_top_candidates(data, args, score_column)
        target_ids, _, fasta_path = _target_accessions(args, gene)
        reference = _filter_transcript_reference(NM_SEQ, args, target_ids)
        print("[off-target] excluding target transcript(s) from {}".format(fasta_path))

        detail_dir = output_root / label
        detail_dir.mkdir(parents=True, exist_ok=True)
        cache = {}
        results = []
        for _, row in data.iterrows():
            key = _duplex_key(row, as_column, ss_column)
            if key not in cache:
                detail_path = detail_dir / ("duplex_{}_{}.csv".format(len(cache), suffix))
                count = calculator(key[1], reference, args, detail_path)
                cache[key] = count, _detail_path_for_output(detail_path)
            results.append(cache[key])

        data[count_column] = [result[0] for result in results]
        data[path_column] = [result[1] for result in results]
        data.to_csv(output_root / (label + ".csv"), index=False)
        print("[off-target] saved {} top candidate(s); {} unique naked duplex(es)".format(
            len(data), len(cache)
        ))


def cal_homo_substr_num(ss, NM_SEQ, args, path):
    homo_all_num = 0
    ss = _normalize_rna(ss)
    rows = []
    for _, transcript in NM_SEQ.iterrows():
        cur_id = str(transcript["nm_id"]).strip().split()[0]
        cur_seq = _normalize_rna(transcript["sequence"])
        for match in Homo_offtarget_Based_on_substr(ss, cur_seq, min_match=16):
            homo_all_num += 1
            rows.append({
                "homo_nm_id": cur_id,
                "ss_site": match["query"],
                "mrna_site": match["subject"],
                "pairs": match["pairs"],
                "amtch_ntd_num": match["num_pairs"],
                "pos_on_mrna": match["sub_index"],
            })
    pd.DataFrame(rows, columns=[
        "homo_nm_id", "ss_site", "mrna_site", "pairs", "amtch_ntd_num", "pos_on_mrna"
    ]).to_csv(path, index=False)
    return homo_all_num


def cal_homo_sw_num(ss, NM_SEQ, args, path):
    homo_all_num = 0
    ss = _normalize_rna(ss)
    aligner = LocalAligner_multi_findings_based_on_ss(
        min_match=16, match=5, mismatch=-3, gap=-2
    )
    rows = []
    for _, transcript in tqdm(NM_SEQ.iterrows(), total=len(NM_SEQ), desc="Smith-Waterman"):
        cur_id = str(transcript["nm_id"]).strip().split()[0]
        cur_seq = _normalize_rna(transcript["sequence"])
        for match in aligner.reset_forwarding(seq_query=ss, seq_subject=cur_seq):
            homo_all_num += 1
            rows.append({
                "homo_nm_id": cur_id,
                "ss_site": match["query"],
                "mrna_site": match["subject"],
                "pairs": match["pairs"],
                "amtch_ntd_num": match["num_pairs"],
                "pos_on_mrna": match["sub_index"],
            })
    pd.DataFrame(rows, columns=[
        "homo_nm_id", "ss_site", "mrna_site", "pairs", "amtch_ntd_num", "pos_on_mrna"
    ]).to_csv(path, index=False)
    return homo_all_num


def cal_utr_paires_num(ss_seq, UTR_SEQ, path):
    """Count canonical seed matches using the original pipeline's SS positions."""
    ss_seq = _normalize_rna(ss_seq)
    if len(ss_seq) < 18:
        raise ValueError("Sense sequence must contain at least 18 nt")
    ss_12_18 = ss_seq[11:18]
    ss_13_18 = ss_seq[12:18]
    rows = []
    counts = {"8mer": 0, "7mer-m8": 0, "7mer-A1": 0}
    for _, transcript in UTR_SEQ.iterrows():
        cur_id = str(transcript["gene"]).strip()
        cur_seq = _normalize_rna(transcript["utr_seqs"])
        pos = cur_seq.find(ss_12_18)
        if pos != -1 and pos + 7 < len(cur_seq):
            pair_type = "8mer" if cur_seq[pos + 7] == "A" else "7mer-m8"
            counts[pair_type] += 1
            rows.append({
                "utr_nm_id": cur_id,
                "utr_seed": cur_seq[pos:pos + 8],
                "pair_type": pair_type,
                "pos_on_utr": pos,
            })
            continue

        pos = cur_seq.find(ss_13_18)
        if pos != -1 and pos + 6 < len(cur_seq) and cur_seq[pos + 6] == "A":
            counts["7mer-A1"] += 1
            rows.append({
                "utr_nm_id": cur_id,
                "utr_seed": cur_seq[pos:pos + 7],
                "pair_type": "7mer-A1",
                "pos_on_utr": pos,
            })

    pd.DataFrame(rows, columns=[
        "utr_nm_id", "utr_seed", "pair_type", "pos_on_utr"
    ]).to_csv(path, index=False)
    total = sum(counts.values())
    return total, counts["8mer"], counts["7mer-m8"], counts["7mer-A1"]


def get_energy(seq):
    result = 0.0
    energy_dict = {"A": 0, "C": 1, "G": 2, "U": 3, "table": np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}
    for index in range(len(seq) - 1):
        result += energy_dict["table"][energy_dict[seq[index]], energy_dict[seq[index + 1]]]
    return result
