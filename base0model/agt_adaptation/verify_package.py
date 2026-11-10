#!/usr/bin/env python3
"""Verify the released AGT 207 -> 69/138 adaptation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUP_COLUMNS = ("ori_as", "ori_ss", "cm_as", "cm_ss")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def group_key(row):
    return tuple(str(row[column]).strip().upper() for column in GROUP_COLUMNS)


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator


def metrics(labels, predictions, threshold=0.7):
    residuals = [prediction - label for label, prediction in zip(labels, predictions)]
    binary = [label >= threshold for label in labels]
    prediction_ranks = average_ranks(predictions)
    positive_ranks = sum(rank for rank, positive in zip(prediction_ranks, binary) if positive)
    positives = sum(binary)
    negatives = len(binary) - positives
    auc = (positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)
    return {
        "n": len(labels),
        "pcc": pearson(labels, predictions),
        "spcc": pearson(average_ranks(labels), prediction_ranks),
        "rmse": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "mae": sum(abs(value) for value in residuals) / len(residuals),
        "auc": auc,
    }


def assert_metrics(actual, expected, tolerance=1e-6):
    assert actual["n"] == expected["n"]
    for name in ("pcc", "spcc", "rmse", "mae", "auc"):
        assert abs(actual[name] - expected[name]) <= tolerance, (
            name,
            actual[name],
            expected[name],
        )


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    data = read_csv(ROOT / "data" / "agt_207.csv")
    manifest = read_csv(ROOT / "split_manifest.csv")
    predictions = read_csv(ROOT / "results" / "holdout_predictions.csv")
    summary = json.loads((ROOT / "results" / "metrics_summary.json").read_text())

    assert len(data) == 207
    assert len(manifest) == 182
    assert len(predictions) == 138
    assert len({row["raw_ID"] for row in data}) == 207

    assignments = {}
    declared_sizes = {}
    for row in manifest:
        key = group_key(row)
        assert key not in assignments
        assert row["split"] in {"finetune", "holdout"}
        assignments[key] = row["split"]
        declared_sizes[key] = int(row["group_size"])

    observed_sizes = Counter(group_key(row) for row in data)
    assert set(assignments) == set(observed_sizes)
    assert declared_sizes == observed_sizes
    finetune = [row for row in data if assignments[group_key(row)] == "finetune"]
    holdout = [row for row in data if assignments[group_key(row)] == "holdout"]
    assert (len(finetune), len(holdout)) == (69, 138)
    assert {group_key(row) for row in finetune}.isdisjoint(
        {group_key(row) for row in holdout}
    )
    assert (len({group_key(row) for row in finetune}), len({group_key(row) for row in holdout})) == (65, 117)
    assert {row["raw_ID"] for row in predictions} == {row["raw_ID"] for row in holdout}

    labels = [float(row["inhibit/0-1"]) for row in predictions]
    zero_shot = [float(row["base0_zeroshot_mean"]) for row in predictions]
    finetuned = [float(row["base0_finetuned_mean"]) for row in predictions]
    assert_metrics(metrics(labels, zero_shot), summary["zero_shot_10fold_ensemble"])
    assert_metrics(metrics(labels, finetuned), summary["finetuned_10fold_seed_mean"])

    rounded = summary["manuscript_rounded"]
    assert rounded["zero_shot"] == {"pcc": 0.595, "spcc": 0.357, "auc": 0.659}
    assert rounded["finetuned"] == {"pcc": 0.652, "spcc": 0.629, "auc": 0.814}

    for relative_path, expected in summary["artifact_sha256"].items():
        assert sha256(ROOT / relative_path) == expected

    print("PASS: 207 rows -> 69 finetune / 138 holdout")
    print("PASS: 0 sequence/modification groups overlap")
    print("PASS: zero-shot PCC/SPCC/AUC = 0.595/0.357/0.659")
    print("PASS: finetuned PCC/SPCC/AUC = 0.652/0.629/0.814")


if __name__ == "__main__":
    main()
