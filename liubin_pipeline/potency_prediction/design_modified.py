#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Design chemically modified siRNAs from a FASTA or user-provided naked siRNAs."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from .predictor import predict_base3_csv
from .predictor_base0 import predict_base0_csv


PIPELINE_DIR = Path(__file__).resolve().parents[1]
CORE_SS_LEN = 19
CORE_AS_LEN = 21
MIN_PATTERN_EXAMPLES = 5
RECOMMENDED_PATTERN_EXAMPLES = 20
FINAL_DISPLAY_COLUMNS = [
    "design_rank",
    "target_gene",
    "ori_as",
    "ori_ss",
    "cm_as",
    "cm_ss",
    "scoring_concentration_nm",
    "scoring_cell_type",
]
MRNA_DISPLAY_COLUMNS = [
    "NM_id",
    "local_mrna",
    "mrna_len",
    "ss_pos(_prefer_middle)",
]


def _display_columns(dataframe, include_mrna=False):
    requested = list(FINAL_DISPLAY_COLUMNS)
    if include_mrna:
        requested.extend(MRNA_DISPLAY_COLUMNS)
    return [column for column in requested if column in dataframe.columns]


def _clean_rna(sequence):
    return str(sequence).strip().upper().replace("T", "U")


def _reverse_complement(sequence):
    sequence = _clean_rna(sequence)
    invalid = sorted(set(sequence).difference("AUCG"))
    if invalid:
        raise ValueError("Cannot reverse-complement symbols {} in {}".format(invalid, sequence))
    return sequence.translate(str.maketrans("AUCG", "UAGC"))[::-1]


def _read_fasta(path):
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
        raise ValueError("No FASTA records found in {}".format(path))
    return records


def _base3_candidates(target_fasta, gene):
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
        raise ValueError("No valid 59-nt windows could be generated from {}".format(target_fasta))
    if skipped:
        print("[design] skipped {} window(s) containing non-A/U/C/G symbols".format(skipped))
    return pd.DataFrame(rows)


def _extend_duplex(row, sense_length, overhang_mode):
    local_mrna = _clean_rna(row["local_mrna"])
    if sense_length < CORE_SS_LEN:
        raise ValueError("--design_ss_length must be at least {}".format(CORE_SS_LEN))
    if 22 + sense_length > len(local_mrna):
        return None

    sense = local_mrna[22:22 + sense_length]
    if sense[:CORE_SS_LEN] != _clean_rna(row["SS"]):
        raise ValueError("Base3 SS does not match its local_mrna context")

    if overhang_mode == "target-derived":
        overhang = _reverse_complement(local_mrna[20:22])
    elif overhang_mode == "UU":
        overhang = "UU"
    elif overhang_mode == "tt":
        overhang = "tt"
    else:
        raise ValueError("Unsupported AS overhang mode: {}".format(overhang_mode))

    antisense = _reverse_complement(sense) + overhang
    return antisense, sense, overhang


def _tokenize_pattern(original, modified):
    original = str(original).strip()
    modified = str(modified).strip()
    tokens = []
    q_count = 0
    for symbol in modified:
        if symbol == "q":
            q_count += 1
            continue
        position = len(tokens)
        if position >= len(original):
            return None
        state_symbol = None if symbol == original[position] else symbol
        tokens.append((q_count, state_symbol))
        q_count = 0
    if q_count or len(tokens) != len(original):
        return None
    return tokens


def _position_distributions(train_df, original_column, modified_column, length):
    counters = [Counter() for _ in range(length)]
    valid_rows = 0
    for original, modified in zip(train_df[original_column], train_df[modified_column]):
        original = str(original).strip()
        if len(original) != length:
            continue
        tokens = _tokenize_pattern(original, modified)
        if tokens is None:
            continue
        valid_rows += 1
        for position, state in enumerate(tokens):
            counters[position][state] += 1

    if valid_rows < MIN_PATTERN_EXAMPLES:
        raise ValueError(
            "Only {} valid {}-nt examples were found for {}/{}; at least {} are required".format(
                valid_rows, length, original_column, modified_column, MIN_PATTERN_EXAMPLES
            )
        )

    distributions = []
    for counter in counters:
        states = list(counter)
        counts = [counter[state] for state in states]
        total = float(sum(counts))
        distributions.append((states, [count / total for count in counts]))
    if valid_rows < RECOMMENDED_PATTERN_EXAMPLES:
        print(
            "[design][warning] only {} exact-length examples support {}-nt {} patterns; "
            "sampling uncertainty will be high".format(valid_rows, length, modified_column)
        )
    return distributions, valid_rows


def _weighted_state(rng, states, probabilities):
    threshold = rng.random()
    cumulative = 0.0
    for state, probability in zip(states, probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return state, probability
    return states[-1], probabilities[-1]


def _sample_pattern(original, distributions, rng):
    pieces = []
    log_probability = 0.0
    changed = False
    for base, (states, probabilities) in zip(original, distributions):
        (q_count, state_symbol), probability = _weighted_state(
            rng, states, probabilities
        )
        symbol = base if state_symbol is None else state_symbol
        pieces.append("q" * q_count + symbol)
        log_probability += math.log(max(probability, 1e-12))
        changed = changed or q_count > 0 or state_symbol is not None
    return "".join(pieces), log_probability, changed


def _first_existing_column(dataframe, names):
    for name in names:
        if name in dataframe.columns:
            return name
    return None


def _standardize_user_naked_sirnas(input_csv, gene, sense_length):
    data = pd.read_csv(input_csv)
    as_column = _first_existing_column(
        data,
        ["ori_as", "AS", "antisense", "Antisense", "antisense_strand", "Antisense strand (naked)"],
    )
    ss_column = _first_existing_column(
        data,
        ["ori_ss", "SS", "sense", "Sense", "sense_strand", "Sense strand (naked)"],
    )
    if as_column is None or ss_column is None:
        raise ValueError(
            "Method 6 input CSV must contain naked antisense and sense columns. "
            "Accepted examples: ori_as/ori_ss, AS/SS, antisense/sense."
        )

    out = data.copy()
    out["ori_as"] = out[as_column].map(_clean_rna)
    out["ori_ss"] = out[ss_column].map(_clean_rna)
    out = out.loc[(out["ori_as"] != "") & (out["ori_ss"] != "")].reset_index(drop=True)
    if out.empty:
        raise ValueError("Method 6 input CSV contains no usable naked siRNA rows")

    antisense_length = sense_length + 2
    bad_lengths = out.loc[
        (out["ori_ss"].str.len() != sense_length) |
        (out["ori_as"].str.len() != antisense_length)
    ]
    if not bad_lengths.empty:
        example = bad_lengths.iloc[0]
        raise ValueError(
            "Method 6 expects ori_ss length {} and ori_as length {} for the current "
            "--design_ss_length. First mismatch: ori_ss={} nt, ori_as={} nt. "
            "Adjust --design_ss_length or provide matching naked duplexes.".format(
                sense_length, antisense_length, len(example["ori_ss"]), len(example["ori_as"])
            )
        )

    for column in ["ori_as", "ori_ss"]:
        invalid_rows = out[column].map(lambda value: sorted(set(value).difference("AUCG")))
        if invalid_rows.map(bool).any():
            first = invalid_rows[invalid_rows.map(bool)].iloc[0]
            raise ValueError("Method 6 {} contains unsupported symbols {}".format(column, first))

    if "target_gene" not in out.columns:
        out["target_gene"] = gene
    out["user_input_rank"] = range(1, len(out) + 1)
    return out


def _generate_modified_candidates(
        top_df, train_df, sense_length, overhang_mode, patterns_per_sirna, seed, gene):
    antisense_length = sense_length + 2
    as_distributions, as_examples = _position_distributions(
        train_df, "ori_as", "cm_as", antisense_length
    )
    ss_distributions, ss_examples = _position_distributions(
        train_df, "ori_ss", "cm_ss", sense_length
    )
    print(
        "[design] pattern statistics: AS {} nt from {} rows; SS {} nt from {} rows".format(
            antisense_length, as_examples, sense_length, ss_examples
        )
    )

    rng = random.Random(seed)
    output_rows = []
    for _, source_row in top_df.iterrows():
        duplex = _extend_duplex(source_row, sense_length, overhang_mode)
        if duplex is None:
            continue
        ori_as, ori_ss, overhang = duplex
        observed = set()
        attempts = 0
        while len(observed) < patterns_per_sirna and attempts < patterns_per_sirna * 100:
            attempts += 1
            cm_as, as_logp, as_changed = _sample_pattern(
                ori_as, as_distributions, rng
            )
            cm_ss, ss_logp, ss_changed = _sample_pattern(
                ori_ss, ss_distributions, rng
            )
            key = (cm_as, cm_ss)
            if key in observed or not (as_changed or ss_changed):
                continue
            observed.add(key)
            row = source_row.to_dict()
            row.update({
                "target_gene": gene,
                "Base3_core_AS": row.pop("AS"),
                "Base3_core_SS": row.pop("SS"),
                "Base3_rank": int(source_row["Base3_rank"]),
                "ori_as": ori_as,
                "ori_ss": ori_ss,
                "cm_as": cm_as,
                "cm_ss": cm_ss,
                "as_overhang": overhang,
                "as_overhang_mode": overhang_mode,
                "design_ss_length": sense_length,
                "design_as_length": antisense_length,
                "pattern_log_probability": as_logp + ss_logp,
                "pattern_index_within_core": len(observed),
            })
            output_rows.append(row)
        if len(observed) < patterns_per_sirna:
            raise RuntimeError(
                "Could generate only {} unique patterns for Base3 rank {} after {} attempts".format(
                    len(observed), int(source_row["Base3_rank"]), attempts
                )
            )
    if not output_rows:
        raise ValueError("No modified candidates were generated")
    return pd.DataFrame(output_rows)


def _generate_modified_candidates_from_naked(
        naked_df, train_df, sense_length, patterns_per_sirna, seed, gene):
    antisense_length = sense_length + 2
    as_distributions, as_examples = _position_distributions(
        train_df, "ori_as", "cm_as", antisense_length
    )
    ss_distributions, ss_examples = _position_distributions(
        train_df, "ori_ss", "cm_ss", sense_length
    )
    print(
        "[design] pattern statistics: AS {} nt from {} rows; SS {} nt from {} rows".format(
            antisense_length, as_examples, sense_length, ss_examples
        )
    )

    rng = random.Random(seed)
    output_rows = []
    for _, source_row in naked_df.iterrows():
        observed = set()
        attempts = 0
        ori_as = _clean_rna(source_row["ori_as"])
        ori_ss = _clean_rna(source_row["ori_ss"])
        while len(observed) < patterns_per_sirna and attempts < patterns_per_sirna * 100:
            attempts += 1
            cm_as, as_logp, as_changed = _sample_pattern(
                ori_as, as_distributions, rng
            )
            cm_ss, ss_logp, ss_changed = _sample_pattern(
                ori_ss, ss_distributions, rng
            )
            key = (cm_as, cm_ss)
            if key in observed or not (as_changed or ss_changed):
                continue
            observed.add(key)
            row = source_row.to_dict()
            row.update({
                "target_gene": row.get("target_gene") or gene,
                "ori_as": ori_as,
                "ori_ss": ori_ss,
                "cm_as": cm_as,
                "cm_ss": cm_ss,
                "design_ss_length": sense_length,
                "design_as_length": antisense_length,
                "pattern_log_probability": as_logp + ss_logp,
                "pattern_index_within_input": len(observed),
            })
            output_rows.append(row)
        if len(observed) < patterns_per_sirna:
            raise RuntimeError(
                "Could generate only {} unique patterns for user input rank {} "
                "after {} attempts".format(
                    len(observed), int(source_row["user_input_rank"]), attempts
                )
            )
    if not output_rows:
        raise ValueError("No modified candidates were generated from user input")
    return pd.DataFrame(output_rows)


def _score_and_save_modified_candidates(
        modified_df, candidates_csv, all_scored_csv, final_csv, args, target_fasta):
    modified_df["scoring_concentration_nm"] = float(args.base0_concentration_nm)
    modified_df["scoring_cell_type"] = (
        str(args.base0_cell_type).strip() if str(args.base0_cell_type).strip() else "unknown"
    )
    modified_df.to_csv(candidates_csv, index=False)
    print("[design] generated {} modified candidates".format(len(modified_df)))

    cache_value = str(args.base0_rnafm_cache).strip()
    rnafm_cache = Path(cache_value) if cache_value else None
    scored = predict_base0_csv(
        input_csv=candidates_csv,
        output_csv=all_scored_csv,
        target_fasta=target_fasta,
        model_name=args.base0_model,
        ckpt_dir=args.base0_ckpt_dir,
        folds=args.base0_folds,
        train_data=args.base0_train_data,
        rnafm_cache=rnafm_cache,
        rebuild_rnafm=args.base0_rebuild_rnafm,
        batch_size=args.base0_batch_size,
        rnafm_batch_size=args.base0_rnafm_batch_size,
        max_len=args.base0_max_len,
        mrna_max_len=args.base0_mrna_max_len,
        window_up=args.base0_window_up,
        window_down=args.base0_window_down,
        concentration_nm=args.base0_concentration_nm,
        cell_type=args.base0_cell_type,
        as_ts=args.base0_as_ts,
        ss_ts=args.base0_ss_ts,
        cell_embed_dim=args.base0_cell_embed_dim,
        device=args.device,
    )
    scored = scored.sort_values("Base0_avg", ascending=False, kind="mergesort").reset_index(drop=True)
    scored["design_rank"] = range(1, len(scored) + 1)
    scored.to_csv(all_scored_csv, index=False)
    final_n = int(args.design_final_top_n)
    if final_n != -1:
        if final_n <= 0:
            raise ValueError("--design_final_top_n must be -1 or positive")
        scored = scored.head(final_n).copy()
    scored[_display_columns(scored)].to_csv(final_csv, index=False)
    print("[design] saved final ranked candidates to {}".format(final_csv))
    return all_scored_csv


def _validate_design_lengths(args):
    sense_length = int(args.design_ss_length)
    antisense_length = sense_length + 2
    if antisense_length > int(args.base0_max_len):
        raise ValueError(
            "Requested AS length {} exceeds Base0 max_len {}. "
            "Use SS length <= {} for the final pad{} model.".format(
                antisense_length, args.base0_max_len,
                int(args.base0_max_len) - 2, args.base0_max_len,
            )
        )
    if args.design_patterns_per_sirna <= 0:
        raise ValueError("--design_patterns_per_sirna must be positive")
    return sense_length, antisense_length


def design_modified_sirnas(args):
    if not args.g or len(args.g) != 1:
        raise ValueError("Method 5 requires exactly one target gene via --g GENE")
    gene = str(args.g[0])
    target_fasta = Path(args.base0_target_fasta)
    sense_length, _ = _validate_design_lengths(args)
    if args.design_base3_top_n <= 0 or args.design_patterns_per_sirna <= 0:
        raise ValueError("Design top-N and patterns-per-siRNA must be positive")

    sirna_dir = PIPELINE_DIR / "output" / "sirna"
    design_dir = PIPELINE_DIR / "output" / "sirnacm_design"
    sirna_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)
    base3_csv = sirna_dir / (gene + "_design_base3.csv")
    candidates_csv = design_dir / (gene + "_modified_candidates.csv")
    all_scored_csv = design_dir / (gene + "_modified_all_scored.csv")
    final_csv = design_dir / (gene + "_modified_ranked.csv")

    base3_df = _base3_candidates(target_fasta, gene)
    base3_df.to_csv(base3_csv, index=False)
    predict_base3_csv(
        input_csv=base3_csv,
        output_csv=base3_csv,
        model_name=args.base3_model,
        ckpt_dir=args.base3_ckpt_dir,
        folds=args.base3_folds,
        train_data=args.base3_train_data,
        batch_size=args.base3_batch_size,
        rnafm_batch_size=args.base3_rnafm_batch_size,
        device=args.device,
    )
    base3_df = pd.read_csv(base3_csv).sort_values(
        "Base3_avg", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    base3_df["Base3_rank"] = range(1, len(base3_df) + 1)
    top_df = base3_df.head(int(args.design_base3_top_n)).copy()
    print("[design] selected Base3 top {}/{} cores".format(len(top_df), len(base3_df)))

    train_df = pd.read_csv(args.base0_train_data, low_memory=False)
    if len(train_df) != 22622:
        raise ValueError("Expected 22,622 Base0 training rows, found {}".format(len(train_df)))
    modified_df = _generate_modified_candidates(
        top_df=top_df,
        train_df=train_df,
        sense_length=sense_length,
        overhang_mode=args.design_as_overhang,
        patterns_per_sirna=int(args.design_patterns_per_sirna),
        seed=int(args.design_seed),
        gene=gene,
    )
    return _score_and_save_modified_candidates(
        modified_df, candidates_csv, all_scored_csv, final_csv, args, target_fasta
    )


def design_modified_sirnas_from_naked(args):
    if not args.g or len(args.g) != 1:
        raise ValueError("Method 6 requires exactly one target gene via --g GENE")
    if not args.sirnacm_path:
        raise ValueError("Method 6 requires --sirnacm_path pointing to a naked-siRNA CSV")
    gene = str(args.g[0])
    target_fasta = Path(args.base0_target_fasta) if args.base0_target_fasta else None
    sense_length, _ = _validate_design_lengths(args)

    design_dir = PIPELINE_DIR / "output" / "sirnacm_design"
    design_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = design_dir / (gene + "_user_modified_candidates.csv")
    all_scored_csv = design_dir / (gene + "_user_modified_all_scored.csv")
    final_csv = design_dir / (gene + "_user_modified_ranked.csv")

    naked_df = _standardize_user_naked_sirnas(args.sirnacm_path, gene, sense_length)
    train_df = pd.read_csv(args.base0_train_data, low_memory=False)
    if len(train_df) != 22622:
        raise ValueError("Expected 22,622 Base0 training rows, found {}".format(len(train_df)))
    modified_df = _generate_modified_candidates_from_naked(
        naked_df=naked_df,
        train_df=train_df,
        sense_length=sense_length,
        patterns_per_sirna=int(args.design_patterns_per_sirna),
        seed=int(args.design_seed),
        gene=gene,
    )
    return _score_and_save_modified_candidates(
        modified_df, candidates_csv, all_scored_csv, final_csv, args, target_fasta
    )
