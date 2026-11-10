# KDMSi Model Card

## Model Purpose

KDMSi ranks candidate siRNA duplexes, including chemically modified siRNAs,
for downstream experimental prioritization.

## Inputs

- Target gene name or target FASTA
- Candidate naked siRNA sequences
- Candidate chemically modified siRNA sequences
- Optional local mRNA context

## Outputs

- Ranked siRNA candidates
- KDMSi score
- Optional off-target summary in downloadable CSV outputs

## Intended Use

Research-use prioritization of siRNA candidates before experimental validation.

## Limitations

- Predictions should not replace experimental validation.
- Off-target analysis depends on the supplied reference tables.
- Reproducibility depends on the training configuration, RNA-FM checkpoint and reference versions.

## Chemical Modification Notation

Internal model symbols are mapped to user-facing notation in documentation and
web output. Common mappings include:

```text
Y -> 2'-O-methyl: Am, Cm, Gm, Um
F -> 2'-fluoro: Af, Cf, Gf, Uf
Z -> 2'-deoxy: Ad, Cd, Gd, Ud
q -> phosphorothioate linkage: Ps
```
