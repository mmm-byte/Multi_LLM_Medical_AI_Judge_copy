# Multi-LLM-as-Judge for Clinical QA Evaluation

> **EMNLP 2026 research repository** - studying whether small, locally hosted
> medical LLMs can act as reliable judges for broad clinical question answering,
> and how judge agreement changes with rubric choice and scoring scale.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![EMNLP 2026](https://img.shields.io/badge/Venue-EMNLP%202026-green)]()

Working paper title:

> **"Small Medical LLMs as Local Judges: Rubric and Scoring Sensitivity in Clinical QA Evaluation"**

Companion general-purpose framework: [Multi-LLMs-as-Judge](https://github.com/m22oct2000/Multi-LLMs-as-Judge)

---

## Research Overview

Cloud-based frontier models are not always acceptable judges for medical AI:
clinical text can be sensitive, and hospitals/HPC groups often need local
evaluation workflows. This repository tests a privacy-preserving alternative:
use 3-4 small medical LLMs, each near or below 7B parameters, as a local judge
panel for clinical QA answers.

The project asks:

| # | Research Question |
|---|---|
| RQ1 | Do small local medical LLM judges agree with each other on clinical QA answer quality? |
| RQ2 | Which rubric produces the most stable inter-judge agreement? |
| RQ3 | Does scoring scale change agreement when rubric content is held constant? |
| RQ4 | Do agreement patterns differ across clinical domains such as emergency care, pharmacology, and pediatrics? |

Key design decisions:

- **Small local judges only**: models are intended for vLLM/FastAPI deployment on local GPU/HPC resources.
- **Context-window aware prompts**: prompts are intentionally short because 7B-class medical models and BioMedLM have limited context windows.
- **No mediator/correction loop**: this repo studies agreement, disagreement, and rubric sensitivity; it does not try to repair answers.
- **Human/reference-answer datasets**: source CSV support is provided for MedQuAD, MedDialog, and Medical Meadow style datasets copied from the companion repo.
- **Deterministic microbenchmark**: a built-in short benchmark keeps tests and examples reproducible without downloading large datasets.

---

## Benchmark Scope

The active paper direction is **broad clinical QA**, not ADRD-only.

The default benchmark uses five clinical domains:

| Domain | Example focus |
|---|---|
| Cardiology | STEMI, ACS chest pain, atrial fibrillation |
| Pharmacology | metformin, warfarin, opioid overdose |
| Neurology | stroke, seizures, Alzheimer disease |
| Pediatrics | febrile seizures, otitis media, dehydration |
| Emergency | anaphylaxis, sepsis, tension pneumothorax |

`benchmark_dataset/build_agreement_dataset.py` writes a small-context CSV with
balanced showcase rows for:

- `fully_agree`
- `majority_agree` (partial agreement with one likely outlier)
- `split` (neutral/borderline disagreement: two-vs-two or unclear majority)
- `full_disagree`

The script also has an optional source-data mode:

```bash
USE_SOURCE_DATASETS=1 python3 benchmark_dataset/build_agreement_dataset.py
```

That mode reads CSV files placed in `benchmark_dataset/source_datasets/`, keeps
rows that fit the small-model token budget, classifies them into the five
domains, and assigns transparent heuristic expected labels. This is intended for
human/reference-answer datasets copied from the companion repository.

---

## Rubrics

The experiments use **four published rubric families plus one controlled
scoring variant**:

| Rubric | Type | Purpose |
|---|---|---|
| `rubric1_pemat.json` | Binary | Patient-facing understandability/actionability |
| `rubric2_healthbench.json` | Binary | Clinical correctness, safety, escalation, uncertainty |
| `rubric3_clinical_eval.json` | Likert 1-5 | Expert-review dimensions: accuracy, safety, relevance, completeness, clarity |
| `rubric4_prometheus.json` | Likert 1-5 | General LLM-as-judge dimensions: instruction following, factuality, coherence, completeness |
| `rubric5_pemat_likert.json` | Likert 1-5 variant | Controlled scoring-scale test: same PEMAT criteria as rubric 1, but Likert instead of binary |

Rubric 5 is **not a fifth independent published instrument**. It is a controlled
variant used to isolate scoring-scale effects:

> same PEMAT criteria + different scoring scale = direct test of whether binary
> scoring yields higher agreement than Likert scoring.

---

## Judge Panel

The target judge panel is 3-4 small medical/domain LLMs:

| Judge ID | Model | Backend |
|---|---|---|
| `medgemma` | `google/medgemma-4b-it` | vLLM chat endpoint |
| `biomistral` | `BioMistral/BioMistral-7B` | vLLM completion endpoint |
| `meditron` | `epfl-llm/meditron-7b` | vLLM completion endpoint |
| `biomedlm` | `stanford-crfm/BioMedLM` | local FastAPI completion endpoint |

The adapters in `core/model_adapters.py` deliberately use short prompts and
pipe-format outputs so the smaller context windows remain usable.

---

## Agreement Classification

For every question-answer-rubric triple, the framework calls each judge,
parses item-level scores, computes pairwise agreement, and classifies the panel:

| Class | Meaning |
|---|---|
| `fully_agree` | All judges are above the agreement threshold with each other |
| `majority_agree` | Most judges agree and one judge is an outlier |
| `split` | No clear majority; useful as a neutral/borderline ambiguity signal |
| `full_disagree` | Judges broadly diverge |
| `skipped` | Too few judge endpoints responded |

Run a no-LLM walkthrough:

```bash
python3 experiments/demo_agreement_showcases.py
```

This prints examples of full agreement, partial/majority agreement, neutral
split, and full disagreement without requiring GPUs.

---

## Repository Structure

```text
core/
  wrapper.py                 # Judge-panel orchestration
  model_adapters.py          # Per-model prompt/parse adapters
  rubric_engine.py           # Score aggregation and pairwise agreement
  agreement.py               # Panel agreement taxonomy
  consensus_core/            # Dataclasses, event log, in-memory store

config/
  endpoint_config.py         # Default local judge endpoints
  llm_client.py              # Lightweight OpenAI-compatible HTTP client
  configs/                   # Experiment configs

benchmark_dataset/
  build_agreement_dataset.py # Reproducible benchmark + optional source CSV ingestion
  build_adrd_questions.py    # Legacy filename builder for Exp3 JSON input
  source_datasets/           # Place companion-repo/human-reference CSVs here

rubrics/
  rubric1_pemat.json
  rubric2_healthbench.json
  rubric3_clinical_eval.json
  rubric4_prometheus.json
  rubric5_pemat_likert.json

experiments/
  demo_agreement_showcases.py
  exp1_dataset_analysis.py
  exp2_agreement_analysis.py
  exp3_rubric_sensitivity.py
  exp4_boxplot_agreement.py

tests/
  test_core.py
```

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# No LLM required
python3 tests/test_core.py
python3 benchmark_dataset/build_agreement_dataset.py
python3 benchmark_dataset/build_adrd_questions.py
python3 experiments/demo_agreement_showcases.py

# Full local/HPC pipeline, after judge endpoints are running
bash run_all.sh
```

LLM experiments require local judge endpoints on ports 8001-8004 as configured
in `config/configs/config_exp2_agreement.json`.

---

## Experiments

| Experiment | Script | Requires LLMs? | Output |
|---|---|---|---|
| Exp1 | `experiments/exp1_dataset_analysis.py` | No | dataset table and JSON |
| Exp2 | `experiments/exp2_agreement_analysis.py` | Yes | per-rubric judge results and rationales |
| Exp3 | `experiments/exp3_rubric_sensitivity.py` | Yes | scoring-scale sensitivity results |
| Exp4 | `experiments/exp4_boxplot_agreement.py` | No, reads Exp2 | PNG box plots and plot data |
| Demo | `experiments/demo_agreement_showcases.py` | No | console walkthrough of agreement classes |

---

## Paper Positioning

Recommended main claim:

> Small local medical LLM judges are feasible for privacy-preserving clinical QA
> evaluation, but their apparent reliability depends strongly on rubric wording
> and scoring scale.

Important limitations to report clearly:

- LLM-judge agreement is not the same as clinical correctness.
- The deterministic benchmark labels are heuristic priors, not physician gold labels.
- Human/reference-answer datasets strengthen the setting, but an expert-labeled subset is still recommended for final paper claims.
- Small-model context windows force shorter rubrics/prompts, which is part of the practical deployment trade-off studied here.

---

## Ethical Considerations

This repository is research infrastructure for evaluating clinical NLP systems.
It does not provide, endorse, or validate medical advice. Any clinical use would
require expert review, prospective validation, and appropriate regulatory review.
