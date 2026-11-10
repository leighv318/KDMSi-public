# KDMSi

KDMSi is a siRNA design workflow for ranking naked and chemically modified siRNA candidates.

This public snapshot is the source-code companion for the manuscript. It includes the
KDMSi training and inference code, the processed data used by the model scripts,
and a fixed AGT adaptation example with a metric verifier.

## Repository contents

| Path | Contents |
| --- | --- |
| `Base3model/` | KDMSi-Site training and feature preparation |
| `base0model/` | KDMSi-Mod training and target adaptation |
| `liubin_pipeline/` | Python design, inference and off-target workflows |
| `examples/` | Example inputs and reference outputs |
| `web_export/` | Optional ONNX and browser parity tools |
| [Data availability](docs/data_availability.md) | Training tables, split rules and external data |

The [AGT adaptation package](base0model/agt_adaptation/README.md) provides the
207-record input, fixed 69/138 split and holdout predictions. The metric
verifier runs from the repository root without model checkpoints or PyTorch:

```bash
python base0model/agt_adaptation/verify_package.py
```

## Web App

Use the [KDMSi Web App](https://leighv318-kdmsi-webapp.static.hf.space/).
The browser source and deployment are maintained in the separate
[KDMSi-webapp repository](https://github.com/leighv318/KDMSi-webapp).

## Installation

Create the Python environment before running the workflow. A conda environment is
recommended:

```bash
git clone https://github.com/leighv318/KDMSi-public.git
cd KDMSi-public

conda env create -f environment.yml
conda activate kdmsi
```

Alternatively, install the Python packages with pip:

```bash
python -m pip install -r requirements.txt
```

For the Base3/Base0 training entry points, install the retained allRank loss as
a second, explicit step:

```bash
python -m pip install --no-deps -r requirements-training.txt
```

Do not omit `--no-deps`: allRank 1.4.3 publishes obsolete
PyTorch/torchvision upper bounds, which would otherwise downgrade a modern
CUDA-matched installation. The import-time prerequisites needed by its
ApproxNDCG module remain declared in the root environment.

The root dependency set covers the main KDMSi/Base3/Base0 workflow. To export
the browser models and run the Python ONNX parity checks, install the optional
export layer instead (it includes the root requirements):

```bash
python -m pip install -r web_export/requirements.txt
```

## Train KDMSi from the included data

### Obtain RNA-FM

KDMSi uses the upstream RNA-FM checkpoint to generate sequence features.
Download it from the RNA-FM team's Hugging Face repository and verify it before
running the trainers:

```bash
mkdir -p rnafm
curl --fail --location --retry 3 \
  'https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth' \
  -o rnafm/RNA-FM_pretrained.pth
printf '%s  %s\n' '5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99' \
  rnafm/RNA-FM_pretrained.pth | shasum -a 256 -c -
```

A checkpoint already stored elsewhere can be selected with
`RNAFM_MODEL_PATH`. The following commands assume that the checkpoint is in
`rnafm/` and that a CUDA device is available.

### KDMSi-Site

Base3 training now generates its `(21, 640)` antisense and `(59, 640)` mRNA
per-residue caches directly from this weight; the old external
`RNAFM_ENCODING/redevelop` source tree is not needed. Cache preparation runs
automatically before the first training split, or it can be run separately:

```bash
CUDA_VISIBLE_DEVICES=0 python -u Base3model/prepare_rnafm.py \
  --device cuda:0 --batch-size 64

CUDA_VISIBLE_DEVICES=0 python -u Base3model/train.py \
  --model Base3_cross_access_v2_src_no_se --loss_type huber \
  --run_splits all --seed 126 --source_checkpoint_selection heldout_spcc \
  --output_dir liubin_pipeline/potency_prediction/pretrain_models
```

Generated arrays are validated against the training-table identity, pretrained
weight checksum, dtype, and architecture-required shapes, then reused for every
unchanged split. Use `--rebuild-rnafm` with `train.py` (or `--rebuild` with the
preparation command) only when intentionally regenerating the cache. A weight
stored elsewhere can be selected with `RNAFM_MODEL_PATH` or
`--rnafm-model-path`.

### KDMSi-Mod

KDMSi-Mod requires a separate row-aligned
`(22622, 65, 640)` cache. They now stop with an error if that cache is missing
or has the wrong shape; zero embeddings are never used as a silent substitute.
Generate the cache from the same released weight and the published Base0 table:

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  base0model/mrna_features.py \
  --datafile base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv \
  --out-stack base0model/data/mrna_rnafm_22622_maxlen65.npy \
  --max-len 65 \
  --model-location rnafm/RNA-FM_pretrained.pth \
  --batch_size 32
```

Train the ten sample folds and the eleven leave-one-gene-out splits. The
last command puts the resulting checkpoints where the inference and adaptation
scripts expect them.

```bash
(
  cd base0model
  CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --model Base0_mrna_full_no_se --loss_type ndcg --seed 126 --run_leave_gene 1
)
mkdir -p liubin_pipeline/potency_prediction/pretrain_models
cp -R base0model/output_ndcg/Base0_mrna_full_no_se \
  liubin_pipeline/potency_prediction/pretrain_models/
```

To rerun the across-target adaptation, use the freshly trained gene-specific
checkpoints:

```bash
python base0model/finetune_ori.py --gene all --model Base0_mrna_full_no_se \
  --n_finetune 300 --max_finetune_pct 0.5 --seeds 42,123,456 \
  --output_dir runs/kdmsi_mod_target_adaptation
```

See the [AGT instructions](base0model/agt_adaptation/README.md) for the
separate locked-holdout analysis. Newly trained scores can vary with software
versions and hardware. For direct metric verification on the recorded AGT
predictions, use the verifier above.

## Optional reference data

Training and the example FASTA workflow use the included tables and upstream
RNA-FM. Substring and Smith–Waterman annotation and reference lookup by gene
require additional reference files:

```text
liubin_pipeline/off_target/NM_SEQ.csv
liubin_pipeline/genomes/GRCh38_rna_ref/reference.fasta
```

Place the files at those paths. If you have a copy of
`kdmsi_external_assets_v1.zip`, put it in the repository root and unpack it
with `unzip kdmsi_external_assets_v1.zip`. The optional downloader in
`scripts/download_assets.sh` also accepts a local archive, an archive URL in
`KDMSI_EXTERNAL_ASSETS_ZIP_URL`, or individual URLs in
`NM_SEQ_URL` and `REFERENCE_FASTA_URL`.

## Input FASTA And Outputs

Runtime output folders are created automatically. When the workflow is launched
from `liubin_pipeline/`, generated files are written under:

```text
liubin_pipeline/output/mrna/
liubin_pipeline/output/sirna/
liubin_pipeline/output/sirnacm/
```

Users can provide a target transcript FASTA from any readable path with
`--base0_target_fasta`. For example, place a custom FASTA in `examples/` and
refer to it from `liubin_pipeline/`:

```bash
cd liubin_pipeline

python workflow.py \
  --filter sirnacm \
  --g TARGET_GENE \
  --m 5 \
  --base0_target_fasta ../examples/TARGET_GENE.fasta
```

If `--base0_target_fasta` is omitted, the workflow attempts to retrieve the
target transcript by gene name and saves it to `liubin_pipeline/output/mrna/`.

## Quick Example

After training the Site and Mod checkpoints, run the Python workflow on the
example target FASTA:

```bash
cd liubin_pipeline

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
```

To start from user-selected naked siRNAs and only design/rank chemical
modification patterns with Base0, provide a CSV with naked antisense/sense
columns. Accepted column pairs include `ori_as`/`ori_ss`, `AS`/`SS`, or
`antisense`/`sense`.

```csv
target_gene,ori_as,ori_ss
AGT,CUUUUUUGUUUCACAAACAAGCU,CUUGUUUGUGAAACAAAAAAG
AGT,UUGUUCUCAACUUGAAAAGGGAA,CCCUUUUCAAGUUGAGAACAA
```

Then run method 6:

```bash
cd liubin_pipeline

python workflow.py \
  --filter sirnacm \
  --g AGT \
  --m 6 \
  --sirnacm_path ../examples/AGT_user_naked_sirnas.csv \
  --base0_target_fasta ../examples/AGT.fasta \
  --design_patterns_per_sirna 20 \
  --design_final_top_n 100 \
  --design_ss_length 21
```

Method 6 can also be combined with off-target methods, for example `--m 6,2,3`
with `--k 100`.

## Optional browser exports

The ONNX exporters use the locally trained checkpoints. For scripts under
`web_export/` that generate browser assets or parity fixtures, check out the
standalone browser project at the ignored path `webapp/` if you have access:

```bash
git clone git@github.com:leighv318/KDMSi-webapp.git webapp
```

See that project's README for the browser build and deployment instructions.

## Citation

Citation information will be added after manuscript acceptance.

## License

No project-wide license is granted for KDMSi code.

Third-party components retain their original licenses and copyright notices.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and license
copies, including the vendored RNA-FM/ESM implementation.
