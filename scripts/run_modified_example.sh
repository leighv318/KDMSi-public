#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../liubin_pipeline"

python workflow.py \
  --filter sirnacm \
  --g AGT \
  --m 5 \
  --base0_target_fasta ../examples/AGT.fasta \
  --design_base3_top_n 100 \
  --design_patterns_per_sirna 20 \
  --design_final_top_n 100 \
  --design_ss_length 21 \
  --design_as_overhang target-derived
