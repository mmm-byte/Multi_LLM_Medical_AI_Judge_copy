# Source Datasets

Place human/reference-answer CSV files here before running the optional
source-dataset benchmark mode.

The intended sources are the medical QA datasets used in the companion
repository, such as:

| File to create here | HuggingFace Dataset | Description |
|---|---|---|
| `MedQuAD_train.csv` | `Malikeh1375/medical-questions-and-answers` | NIH medical question-answer pairs |
| `MedDialog_validation.csv` | `bigbio/med_dialog` (en, validation split) | Patient-doctor dialogue validation rows |
| `medical_meadow_health_advice.csv` | `medalpaca/medical_meadow_health_advice` | Health-advice instruction/response pairs |

---

## Option A — Download from HuggingFace (recommended on HPC)

Run this once before `run_all.sh --use-source-datasets`:

```bash
pip install datasets pandas --quiet

python3 - <<'EOF'
from datasets import load_dataset
import pandas as pd

print("Downloading MedQuAD...")
ds = load_dataset("Malikeh1375/medical-questions-and-answers", split="train")
df = pd.DataFrame(ds)
# Rename to standard column names the builder expects
df = df.rename(columns={"input": "question", "output": "answer"}) if "input" in df.columns else df
df.to_csv("benchmark_dataset/source_datasets/MedQuAD_train.csv", index=False)
print(f"  Saved {len(df)} rows -> MedQuAD_train.csv")

print("Downloading MedDialog (validation)...")
ds2 = load_dataset("bigbio/med_dialog", "med_dialog_en_bigbio_qa", split="validation", trust_remote_code=True)
df2 = pd.DataFrame(ds2)
df2.to_csv("benchmark_dataset/source_datasets/MedDialog_validation.csv", index=False)
print(f"  Saved {len(df2)} rows -> MedDialog_validation.csv")

print("Downloading medical_meadow_health_advice...")
ds3 = load_dataset("medalpaca/medical_meadow_health_advice", split="train")
df3 = pd.DataFrame(ds3)
df3.to_csv("benchmark_dataset/source_datasets/medical_meadow_health_advice.csv", index=False)
print(f"  Saved {len(df3)} rows -> medical_meadow_health_advice.csv")

print("All source datasets saved.")
EOF
```

---

## Option B — Copy from companion repo on HPC

If you already have the old repo checked out on HPC:

```bash
OLD_REPO=/path/to/Multi-LLMs-as-Judge
NEW_REPO=/path/to/Multi_LLM_as_Judge_Medical_AI

cp "$OLD_REPO/dataset/MedQuAD/train.csv" \
   "$NEW_REPO/benchmark_dataset/source_datasets/MedQuAD_train.csv"

cp "$OLD_REPO/dataset/MedDialog/validation.csv" \
   "$NEW_REPO/benchmark_dataset/source_datasets/MedDialog_validation.csv"

cp "$OLD_REPO/dataset/medical_meadow_health_advice/medical_meadow_health_advice.csv" \
   "$NEW_REPO/benchmark_dataset/source_datasets/medical_meadow_health_advice.csv"
```

---

## Run the full pipeline with source datasets

```bash
# After CSVs are in place:
bash run_all.sh --use-source-datasets
```

Or step by step:

```bash
USE_SOURCE_DATASETS=1 python3 benchmark_dataset/build_agreement_dataset.py
USE_SOURCE_DATASETS=1 python3 benchmark_dataset/build_adrd_questions.py
python3 experiments/exp1_dataset_analysis.py
python3 experiments/exp2_agreement_analysis.py
python3 experiments/exp3_rubric_sensitivity.py
python3 experiments/exp4_boxplot_agreement.py
```

---

## What the builder does with source rows

1. Detects likely question and answer/reference columns automatically.
2. Keeps rows that fit the small-model Q+A token budget (`MAX_QA_TOKENS=200` estimated tokens).
3. Classifies rows into Cardiology, Pharmacology, Neurology, Pediatrics, or
   Emergency using transparent keyword matching.
4. Assigns heuristic expected agreement classes for reproducible comparison.
5. Merges source rows **before** the 25 curated synthetic rows so source data
   is the primary benchmark when available.
6. Preserves the actual observed LLM-judge class separately after Exp2 runs.

The heuristic expected class is **not** a physician gold label. For paper claims,
use the source reference answers as human-written answers and, if possible, add
a small expert-labeled agreement subset.
