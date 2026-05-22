# Clinical QA Benchmark Dataset

This directory contains the reproducible benchmark inputs for the broad
clinical QA version of the project.

## Default benchmark

Run:

```bash
python3 benchmark_dataset/build_agreement_dataset.py
```

The builder writes `benchmark_dataset/agreement_benchmark.csv` with:

- 25 default rows across five clinical domains: Cardiology, Pharmacology,
  Neurology, Pediatrics, Emergency
- short question/reference-answer pairs that fit small 7B-class model context
  windows
- at least five rows for each expected showcase class:
  `fully_agree`, `majority_agree`, `split`, `full_disagree`
- heuristic component scores used only for reproducible testing and demos

`agreement_benchmark.csv` is generated and ignored by git.

## Optional human/reference-answer source CSVs

Place CSVs copied from the companion repository into
`benchmark_dataset/source_datasets/`, then run:

```bash
USE_SOURCE_DATASETS=1 python3 benchmark_dataset/build_agreement_dataset.py
```

The script attempts to detect question and answer/reference columns, filters rows
to the token budget, classifies each row into one of the five domains, and
assigns transparent heuristic expected labels. The actual experiment still
records observed judge-panel labels separately.

## Legacy JSON for Exp3

Exp3 currently expects `benchmark_dataset/adrd_questions.json`. The filename is
kept for compatibility, but its contents are generated from the broad clinical
QA benchmark:

```bash
python3 benchmark_dataset/build_adrd_questions.py
```
