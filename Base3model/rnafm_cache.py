"""Generate and validate the per-residue RNA-FM cache used by Base3.

The original Base3 training code expected a separate ``RNAFM_ENCODING`` source
checkout and thousands of pre-generated ``.npy`` files.  This module produces
the same arrays directly with the RNA-FM implementation already vendored under
``liubin_pipeline/fm`` and the separately released pretrained weight.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BASE3_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BASE3_ROOT.parent
DEFAULT_DATAFILE = BASE3_ROOT / "data" / "normalized_sirna_with_mrna.csv"
DEFAULT_CACHE_DIR = BASE3_ROOT / "data" / "rnafm_cache"
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "rnafm" / "RNA-FM_pretrained.pth"

RNAFM_DIM = 640
AS_LENGTH = 21
MRNA_LENGTH = 59
RNAFM_AS_CHARS = frozenset("AUCG")
RNAFM_MRNA_CHARS = frozenset("AUCGRYKMWSBDHVN-")
MRNA_PADDING_CHARS = frozenset(".")
CACHE_FORMAT_VERSION = 1
RELEASED_MODEL_SHA256 = (
    "5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99"
)

_PREPARED = set()


def _clean_rna(sequence):
    return str(sequence).upper().replace("T", "U")


def _normalised_records(datafile):
    table = pd.read_csv(datafile, header=0)
    required = {"Antisense", "mrna", "RNAFM_ind"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "Base3 training data is missing required column(s): {}".format(
                ", ".join(missing)
            )
        )

    records = []
    seen_indices = set()
    for row_number, row in table.iterrows():
        try:
            rnafm_index = int(row["RNAFM_ind"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid RNAFM_ind at Base3 row {}: {!r}".format(
                    row_number, row["RNAFM_ind"]
                )
            ) from exc
        if rnafm_index in seen_indices:
            raise ValueError("Duplicate RNAFM_ind in Base3 data: {}".format(rnafm_index))
        seen_indices.add(rnafm_index)

        antisense = _clean_rna(row["Antisense"])
        mrna = _clean_rna(row["mrna"])
        bad_as = sorted(set(antisense) - RNAFM_AS_CHARS)
        bad_mrna = sorted(set(mrna) - RNAFM_MRNA_CHARS - MRNA_PADDING_CHARS)
        if bad_as:
            raise ValueError(
                "Unsupported antisense character(s) {} at RNAFM_ind={}".format(
                    bad_as, rnafm_index
                )
            )
        if bad_mrna:
            raise ValueError(
                "Unsupported mRNA character(s) {} at RNAFM_ind={}".format(
                    bad_mrna, rnafm_index
                )
            )
        if len(antisense) != AS_LENGTH:
            raise ValueError(
                "Base3 requires {}-nt antisense inputs; RNAFM_ind={} has {} nt".format(
                    AS_LENGTH, rnafm_index, len(antisense)
                )
            )
        if len(mrna) != MRNA_LENGTH:
            raise ValueError(
                "Base3 requires {}-nt mRNA windows; RNAFM_ind={} has {} nt".format(
                    MRNA_LENGTH, rnafm_index, len(mrna)
                )
            )
        records.append((rnafm_index, antisense, mrna))
    return records


def _records_sha256(records):
    digest = hashlib.sha256()
    for rnafm_index, antisense, mrna in records:
        digest.update(
            "{}\t{}\t{}\n".format(rnafm_index, antisense, mrna).encode("ascii")
        )
    return digest.hexdigest()


def _file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_path(model_path=None):
    candidates = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    env_path = os.environ.get("RNAFM_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DEFAULT_MODEL_PATH)
    candidates.append(
        Path(torch.hub.get_dir()) / "checkpoints" / "RNA-FM_pretrained.pth"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "RNA-FM pretrained weight was not found. Download the released "
        "RNA-FM_pretrained.pth asset, place it at {}, or pass "
        "--rnafm-model-path /path/to/RNA-FM_pretrained.pth.\nChecked:\n  - {}".format(
            DEFAULT_MODEL_PATH, checked
        )
    )


def _cache_paths(cache_dir, rnafm_index):
    name = "{:04d}.npy".format(int(rnafm_index))
    cache_dir = Path(cache_dir)
    return (
        cache_dir / "results_as" / "representations" / name,
        cache_dir / "results_mrna" / "representations" / name,
    )


def _array_is_valid(path, expected_shape):
    if not path.is_file():
        return False
    try:
        value = np.load(str(path), mmap_mode="r", allow_pickle=False)
        return (
            value.shape == expected_shape
            and value.dtype == np.float32
            and bool(np.isfinite(value).all())
        )
    except (OSError, ValueError):
        return False


def _cache_state(cache_dir, records):
    missing_as = []
    missing_mrna = []
    for rnafm_index, _, _ in records:
        as_path, mrna_path = _cache_paths(cache_dir, rnafm_index)
        if not _array_is_valid(as_path, (AS_LENGTH, RNAFM_DIM)):
            missing_as.append(rnafm_index)
        if not _array_is_valid(mrna_path, (MRNA_LENGTH, RNAFM_DIM)):
            missing_mrna.append(rnafm_index)
    return missing_as, missing_mrna


def _atomic_save_npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(str(temporary), "wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
    os.replace(str(temporary), str(path))


def _write_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _read_metadata(path):
    if not path.is_file():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _import_repository_fm():
    pipeline_root = REPOSITORY_ROOT / "liubin_pipeline"
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))
    import fm  # pylint: disable=import-outside-toplevel

    fm_path = Path(fm.__file__).resolve()
    expected_root = (pipeline_root / "fm").resolve()
    if expected_root not in fm_path.parents:
        raise ImportError(
            "Imported an unexpected fm package from {}; expected the repository copy "
            "under {}".format(fm_path, expected_root)
        )
    return fm


def _generate_kind(
    records,
    indices_to_generate,
    kind,
    cache_dir,
    model,
    batch_converter,
    device,
    batch_size,
):
    wanted = set(indices_to_generate)
    selected = [record for record in records if record[0] in wanted]
    total = len(selected)
    for start in range(0, total, batch_size):
        batch = selected[start:start + batch_size]
        batch_data = []
        batch_meta = []
        for rnafm_index, antisense, mrna in batch:
            if kind == "as":
                sequence = antisense
                positions = None
            else:
                positions = [
                    position
                    for position, character in enumerate(mrna)
                    if character in RNAFM_MRNA_CHARS
                ]
                sequence = "".join(mrna[position] for position in positions)
                if not sequence:
                    raise ValueError(
                        "Empty mRNA after removing dot padding at RNAFM_ind={}".format(
                            rnafm_index
                        )
                    )
            batch_data.append((str(rnafm_index), sequence))
            batch_meta.append((rnafm_index, sequence, positions))

        _, _, tokens = batch_converter(batch_data)
        with torch.no_grad():
            output = model(tokens.to(device), repr_layers=[12])
        embeddings = output["representations"][12].detach().cpu().numpy()

        for offset, (rnafm_index, sequence, positions) in enumerate(batch_meta):
            residue_embedding = embeddings[
                offset, 1:len(sequence) + 1, :
            ].astype(np.float32, copy=False)
            as_path, mrna_path = _cache_paths(cache_dir, rnafm_index)
            if kind == "as":
                value = residue_embedding
                output_path = as_path
            else:
                value = np.zeros((MRNA_LENGTH, RNAFM_DIM), dtype=np.float32)
                value[np.asarray(positions, dtype=int), :] = residue_embedding
                output_path = mrna_path
            _atomic_save_npy(output_path, value)
        print(
            "[RNA-FM] {} cache: {}/{}".format(
                "antisense" if kind == "as" else "mRNA", min(start + batch_size, total), total
            ),
            flush=True,
        )


def prepare_rnafm_cache(
    datafile=None,
    cache_dir=None,
    model_path=None,
    batch_size=64,
    rebuild=False,
    device=None,
):
    """Ensure that the complete Base3 RNA-FM cache exists and return its root."""
    datafile = Path(datafile or DEFAULT_DATAFILE).expanduser().resolve()
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve()
    if batch_size < 1:
        raise ValueError("RNA-FM batch size must be at least 1")

    records = _normalised_records(datafile)
    dataset_sha256 = _records_sha256(records)
    ready_key = (str(datafile), str(cache_dir), dataset_sha256)
    # A training process asks once per split. Even with --rebuild-rnafm, rebuild
    # only on the first split rather than regenerating the cache 12 times.
    if ready_key in _PREPARED:
        return cache_dir

    resolved_model_path = _resolve_model_path(model_path)
    model_sha256 = _file_sha256(resolved_model_path)
    if model_sha256 != RELEASED_MODEL_SHA256:
        raise RuntimeError(
            "RNA-FM weight checksum mismatch for {}: got {}, expected the "
            "checkpoint SHA-256 {}".format(
                resolved_model_path, model_sha256, RELEASED_MODEL_SHA256
            )
        )
    metadata_path = cache_dir / "metadata.json"
    metadata = _read_metadata(metadata_path)
    expected_identity = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "dataset_sha256": dataset_sha256,
        "model_sha256": model_sha256,
        "rows": len(records),
    }

    if metadata is not None and not rebuild:
        mismatches = [
            key
            for key, expected in expected_identity.items()
            if metadata.get(key) != expected
        ]
        if mismatches:
            raise RuntimeError(
                "Base3 RNA-FM cache metadata does not match the selected data/model "
                "({}). Re-run with --rebuild-rnafm.".format(", ".join(mismatches))
            )

    missing_as, missing_mrna = _cache_state(cache_dir, records)
    if metadata is None and (missing_as or missing_mrna):
        existing_count = 2 * len(records) - len(missing_as) - len(missing_mrna)
        if existing_count:
            raise RuntimeError(
                "Found a partial legacy RNA-FM cache without metadata at {}. "
                "Its sequence identity cannot be verified; re-run with "
                "--rebuild-rnafm.".format(cache_dir)
            )

    if rebuild:
        missing_as = [record[0] for record in records]
        missing_mrna = list(missing_as)

    if not missing_as and not missing_mrna:
        complete_metadata = dict(expected_identity)
        complete_metadata.update({
            "status": "complete",
            "datafile": str(datafile),
            "model_path": str(resolved_model_path),
            "antisense_shape": [AS_LENGTH, RNAFM_DIM],
            "mrna_shape": [MRNA_LENGTH, RNAFM_DIM],
        })
        _write_metadata(metadata_path, complete_metadata)
        _PREPARED.add(ready_key)
        print("[RNA-FM] using validated Base3 cache: {}".format(cache_dir), flush=True)
        return cache_dir

    incomplete_metadata = dict(expected_identity)
    incomplete_metadata.update({
        "status": "incomplete",
        "datafile": str(datafile),
        "model_path": str(resolved_model_path),
        "antisense_shape": [AS_LENGTH, RNAFM_DIM],
        "mrna_shape": [MRNA_LENGTH, RNAFM_DIM],
    })
    _write_metadata(metadata_path, incomplete_metadata)

    fm = _import_repository_fm()
    selected_device = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print("[RNA-FM] loading pretrained weight: {}".format(resolved_model_path), flush=True)
    print("[RNA-FM] embedding device: {}".format(selected_device), flush=True)
    model, alphabet = fm.pretrained.rna_fm_t12(str(resolved_model_path))
    model = model.to(selected_device).eval()
    batch_converter = alphabet.get_batch_converter()

    if missing_as:
        _generate_kind(
            records, missing_as, "as", cache_dir, model, batch_converter,
            selected_device, batch_size,
        )
    if missing_mrna:
        _generate_kind(
            records, missing_mrna, "mrna", cache_dir, model, batch_converter,
            selected_device, batch_size,
        )

    del model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()

    missing_as, missing_mrna = _cache_state(cache_dir, records)
    if missing_as or missing_mrna:
        raise RuntimeError(
            "RNA-FM cache generation finished incomplete: {} antisense and {} mRNA "
            "arrays are missing or malformed".format(len(missing_as), len(missing_mrna))
        )

    complete_metadata = dict(incomplete_metadata)
    complete_metadata["status"] = "complete"
    _write_metadata(metadata_path, complete_metadata)
    _PREPARED.add(ready_key)
    print("[RNA-FM] Base3 cache complete: {}".format(cache_dir), flush=True)
    return cache_dir


def load_cached_embedding(cache_dir, rnafm_index, kind):
    """Load one validated Base3 embedding with its architecture-required shape."""
    as_path, mrna_path = _cache_paths(cache_dir, rnafm_index)
    if kind == "as":
        path = as_path
        expected_shape = (AS_LENGTH, RNAFM_DIM)
    elif kind == "mrna":
        path = mrna_path
        expected_shape = (MRNA_LENGTH, RNAFM_DIM)
    else:
        raise ValueError("Unknown RNA-FM cache kind: {}".format(kind))
    value = np.load(str(path), allow_pickle=False)
    if value.shape != expected_shape or value.dtype != np.float32:
        raise ValueError(
            "Malformed RNA-FM cache {}: got {} {}, expected {} float32".format(
                path, value.shape, value.dtype, expected_shape
            )
        )
    return value
