#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
KDMSi external asset downloader

This script downloads the upstream RNA-FM checkpoint. For the optional
off-target references, place kdmsi_external_assets_v1.zip in the repository
root, or provide the archive or individual file URLs as environment variables.
MSG

RNAFM_URL="${RNAFM_URL:-https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth}"
RNAFM_SHA256="5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99"
KDMSI_EXTERNAL_ASSETS_ZIP_URL="${KDMSI_EXTERNAL_ASSETS_ZIP_URL:-}"
REFERENCE_FASTA_URL="${REFERENCE_FASTA_URL:-}"
NM_SEQ_URL="${NM_SEQ_URL:-}"

mkdir -p rnafm
mkdir -p liubin_pipeline/genomes/GRCh38_rna_ref
mkdir -p liubin_pipeline/off_target

download_if_set() {
  local url="$1"
  local out="$2"
  if [ -z "$url" ]; then
    echo "skip $out: URL not set"
    return 0
  fi
  echo "downloading $out"
  curl --fail --location "$url" -o "$out"
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$path" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$path" | awk '{print $1}')"
  else
    echo "ERROR: sha256sum or shasum is required to verify $path" >&2
    return 1
  fi
  if [ "$actual" != "$expected" ]; then
    echo "ERROR: SHA-256 mismatch for $path" >&2
    echo "  got:      $actual" >&2
    echo "  expected: $expected" >&2
    return 1
  fi
  echo "verified SHA-256 $path"
}

if [ -f "rnafm/RNA-FM_pretrained.pth" ]; then
  echo "using existing rnafm/RNA-FM_pretrained.pth"
  verify_sha256 "rnafm/RNA-FM_pretrained.pth" "$RNAFM_SHA256"
else
  curl --fail --location --retry 3 "$RNAFM_URL" -o "rnafm/RNA-FM_pretrained.pth.part"
  verify_sha256 "rnafm/RNA-FM_pretrained.pth.part" "$RNAFM_SHA256"
  mv "rnafm/RNA-FM_pretrained.pth.part" "rnafm/RNA-FM_pretrained.pth"
fi

if [ -f kdmsi_external_assets_v1.zip ] || [ -n "$KDMSI_EXTERNAL_ASSETS_ZIP_URL" ]; then
  if unzip -tq kdmsi_external_assets_v1.zip >/dev/null 2>&1; then
    echo "using existing kdmsi_external_assets_v1.zip"
  else
    echo "downloading kdmsi_external_assets_v1.zip"
    download_if_set "$KDMSI_EXTERNAL_ASSETS_ZIP_URL" kdmsi_external_assets_v1.zip
  fi
  if ! unzip -tq kdmsi_external_assets_v1.zip >/dev/null 2>&1; then
    cat <<'MSG'
ERROR: kdmsi_external_assets_v1.zip is not a valid zip archive.

Place a valid archive in the repository root or set
KDMSI_EXTERNAL_ASSETS_ZIP_URL to its download URL, then rerun the script.
MSG
    exit 1
  fi
  unzip -o kdmsi_external_assets_v1.zip
else
  download_if_set "$REFERENCE_FASTA_URL" "liubin_pipeline/genomes/GRCh38_rna_ref/reference.fasta"
  download_if_set "$NM_SEQ_URL" "liubin_pipeline/off_target/NM_SEQ.csv"
fi

echo "external asset check:"
test -f rnafm/RNA-FM_pretrained.pth && echo "ok RNA-FM_pretrained.pth" || echo "missing RNA-FM_pretrained.pth"
test -f liubin_pipeline/genomes/GRCh38_rna_ref/reference.fasta && echo "ok reference.fasta" || echo "missing reference.fasta"
test -f liubin_pipeline/off_target/NM_SEQ.csv && echo "ok NM_SEQ.csv" || echo "missing NM_SEQ.csv"
