# AGT feedback adaptation (207 -> 69/138)

This directory contains the compact reproducibility package for the AGT
feedback-adaptation experiment reported in the manuscript. The input contains
207 chemically modified siRNAs. A fixed, sequence/modification-group-disjoint
split assigns 69 records to feedback fine-tuning and keeps 138 records as the
locked holdout set.

## Protocol

- Group identity is the complete tuple `ori_as`, `ori_ss`, `cm_as`, `cm_ss`.
- `split_manifest.csv` fixes all 182 unique groups: 65 groups/69 rows for
  fine-tuning and 117 groups/138 rows for holdout.
- The ten pretrained Base0 sample-split models are evaluated as an ensemble.
- For each fold, only BatchNorm and the full downstream DNN are updated; all
  other Base0 modules remain frozen.
- Fine-tuning uses seeds 42, 123, and 456 for 100 epochs. The holdout set is
  never used for training or checkpoint selection.
- AUC uses the fixed inhibition threshold 0.7.

## Included files

```text
data/agt_207.csv                 207 experimental records
data/AGT.fasta                   AGT transcript sequences used for context mapping
split_manifest.csv               fixed group-level 69/138 split
results/holdout_predictions.csv  compact locked-holdout predictions
results/metrics_summary.json     exact and manuscript-rounded metrics
verify_package.py                dependency-free artifact and metric audit
```

Train the ten KDMSi-Mod sample-fold models using the
[training instructions](../../README.md#train-kdmsi-from-the-included-data).
Use those models to initialize the AGT adaptation. The command below reads
the ten fold directories at
`../../liubin_pipeline/potency_prediction/pretrain_models/Base0_mrna_full_no_se/`.

## Reproduce the run

Create the repository environment as described in the root `README.md`, place
the RNA-FM weight at `rnafm/RNA-FM_pretrained.pth`, and run from the repository
root:

```bash
# The AGT context features require the ViennaRNA Python binding.
python -c "import RNA; print(RNA.__version__)"

python base0model/finetune_wet72_base0_no_se.py \
  --wet_data agt_adaptation/data/agt_207.csv \
  --target_fasta agt_adaptation/data/AGT.fasta \
  --preferred_accession NM_001382817.3 \
  --split_strategy group_manifest \
  --split_manifest agt_adaptation/split_manifest.csv \
  --expected_finetune_rows 69 \
  --expected_holdout_rows 138 \
  --rnafm_model_location ../rnafm/RNA-FM_pretrained.pth \
  --rebuild_rnafm \
  --no_save_checkpoints \
  --output_dir agt_adaptation/run_output
```

Paths passed to this script are resolved relative to `base0model/`. On later
runs, omit `--rebuild_rnafm` to reuse the locally generated cache. The cache and
`run_output/` are ignored by Git.

To verify the released split, predictions, hashes, and reported metrics without
loading PyTorch or RNA-FM:

```bash
python base0model/agt_adaptation/verify_package.py
```

## Reported locked-holdout performance

| State | PCC | SPCC | AUC@0.7 |
|---|---:|---:|---:|
| Zero-shot 10-fold ensemble | 0.595 | 0.357 | 0.659 |
| Fine-tuned 10-fold/3-seed ensemble | 0.652 | 0.629 | 0.814 |

The exact unrounded values and RMSE/MAE are retained in
`results/metrics_summary.json`.
