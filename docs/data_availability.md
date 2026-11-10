# KDMSi reproduction data

Keep the processed training tables in their original row order.

| File | Use |
| --- | --- |
| `Base3model/data/normalized_sirna_with_mrna.csv` | KDMSi-Site training and fold-specific inference inputs |
| `base0model/data/CMsiRNAdb_Dsicm10_FINAL_trainready.csv` | KDMSi-Mod training and fold-specific inference inputs |
| `base0model/data/fingerprints.npy` and `base0model/data/gvae.npy` | Chemical feature lookup tables |
| [`base0model/agt_adaptation/`](../base0model/agt_adaptation/) | AGT inputs, fixed split, holdout predictions and metric summary |
| `liubin_pipeline/off_target/UTR_SEQ.csv` | miRNA-like off-target annotation |
| `liubin_pipeline/genomes/GRCh38_rna_ref/ref_dict.csv` | Transcript metadata for gene-name lookup |

In both ten-fold protocols, fold `k` evaluates the zero-based data rows whose
index modulo ten equals `k`. KDMSi-Site source-held-out runs select all rows
with `source_paper` equal to `T` or `R`. KDMSi-Mod leave-one-gene-out runs
select all rows with the corresponding `target_gene` value. The separate AGT
adaptation uses the supplied group-level split manifest.

Follow the [training instructions](../README.md#train-kdmsi-from-the-included-data)
to download upstream RNA-FM, generate its feature caches, and train the KDMSi
checkpoints locally. The [optional reference data instructions](../README.md#optional-reference-data)
cover the GRCh38 transcript FASTA and `NM_SEQ.csv`. Providing a target FASTA
directly avoids the need for the gene-name lookup assets.
