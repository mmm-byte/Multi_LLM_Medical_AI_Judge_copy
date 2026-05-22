# Real-LLM run results (CPU-only)

This folder contains the artefacts produced by running the four
clinical-QA judge-panel experiments with **real LLM inference** on a
GPU-less environment (4 CPUs, 15 GB RAM).

## What was actually run

The original judge panel needs GPUs (~28 GB VRAM) for
`google/medgemma-4b-it`, `BioMistral/BioMistral-7B`,
`epfl-llm/meditron-7b`, and `stanford-crfm/BioMedLM`. Since this host
has no GPU, the four judge **slots** were preserved (so
`core/model_adapters.py` keeps applying its per-judge prompts and
parsers) but each slot's **underlying model** was remapped to a small
CPU-runnable open instruction-tuned LM:

| Judge id   | Original model              | CPU substitute used here          | Port |
|------------|------------------------------|------------------------------------|------|
| medgemma   | google/medgemma-4b-it        | Qwen/Qwen2.5-1.5B-Instruct         | 8001 |
| biomistral | BioMistral/BioMistral-7B     | Qwen/Qwen2.5-0.5B-Instruct         | 8002 |
| meditron   | epfl-llm/meditron-7b         | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 8003 |
| biomedlm   | stanford-crfm/BioMedLM       | HuggingFaceTB/SmolLM2-360M-Instruct| 8004 |

All four models were served via `serve_hf_model.py`, a generic
OpenAI-compatible FastAPI server (chat + completion endpoints,
chat-template auto-applied for instruction-tuned models).

## Sample size

To keep the CPU runtime tractable, the per-row pipeline runs on a
balanced 8-row sample of the deterministic 25-row showcase benchmark
(`benchmark_dataset/agreement_benchmark_sample.csv`, 2 rows per
expected agreement class).

Inference counts actually executed:

* Exp2 — 8 questions × 5 rubrics × 4 judges = 160 judge calls
* Exp3 — 8 questions × 2 rubrics × 3 scoring variants × 4 judges = 192 judge calls
* Exp4 — no LLM (reads Exp2 JSON)

## Headline numbers

### Exp2: per-rubric agreement (mean pairwise % across 8 questions)

| Rubric                              | Mean PW | fully | majority | split | full_dis |
|-------------------------------------|---------|-------|----------|-------|----------|
| PEMAT (binary)                      | 50.0%   | 25%   | 37.5%    | 37.5% | 0%       |
| HealthBench (binary)                | 37.1%   | 0%    | 50.0%    | 50.0% | 0%       |
| Clinical LLM Eval (Likert 1-5)      | 30.9%   | 0%    | 12.5%    | 75.0% | 12.5%    |
| Prometheus (Likert 1-5)             | 31.3%   | 0%    | 12.5%    | 62.5% | 25.0%    |
| PEMAT-Likert (Likert 1-5)           | 24.0%   | 0%    | 0%       | 87.5% | 12.5%    |

### Exp3: rubric × scoring-variant sensitivity (mean pairwise % ± std)

| Rubric         | BINARY        | LIKERT_1_5     | SCALED_0_10    |
|----------------|---------------|----------------|----------------|
| PEMAT          | 50.0 ± 34.5%  | 28.7 ± 14.0%   | 42.1 ± 20.0%   |
| PEMAT-Likert   | 50.0 ± 34.5%  | 24.0 ± 2.9%    | 36.5 ± 10.0%   |

Within the **controlled PEMAT pair** (criteria held constant, only
the scoring scale varies) the binary scoring produces a substantially
higher inter-judge agreement than the Likert-1-5 scoring (50.0% vs
28.7% / 24.0%), reproducing the qualitative pattern the paper hypothesises.

The `LIKERT_1_5` variant of `rubric5_pemat_likert` was flagged by the
built-in ablation step (std = 2.95% < 5%) as a low-spread row.

## Files

* `exp1_dataset_table.json` — domain × source breakdown (no LLM)
* `exp2_agreement_results_realllm.json` — full per-row judge scores, rationales, agreement
* `exp3_sensitivity_results_realllm.json` — per-rubric × per-variant mean / std / per-question PW
* `exp4_boxplot_data_realllm.json` — flat domain × rubric scores used to build the boxplots
* `figures/realllm/fig_exp2_agreement_rubric{1..5}_*.png` — per-rubric box plot by domain
* `figures/realllm/fig_exp4_boxplot_by_domain.png` — combined 5-panel figure

## Reproduce

```bash
# (one-time) install deps and pre-download weights into hf_cache/
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate fastapi uvicorn
HF_HOME=./hf_cache python -c "from huggingface_hub import snapshot_download as s; \
  [s(m, cache_dir='./hf_cache') for m in [\
    'Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-0.5B-Instruct',\
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0','HuggingFaceTB/SmolLM2-360M-Instruct']]"

# start the 4 judge servers (in separate terminals or tmux panes)
HF_HOME=./hf_cache DTYPE=bfloat16 MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct          PORT=8001 python serve_hf_model.py
HF_HOME=./hf_cache DTYPE=bfloat16 MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct          PORT=8002 python serve_hf_model.py
HF_HOME=./hf_cache DTYPE=bfloat16 MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0  PORT=8003 python serve_hf_model.py
HF_HOME=./hf_cache DTYPE=bfloat16 MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct PORT=8004 python serve_hf_model.py

# run the experiments
python experiments/exp1_dataset_analysis.py
EXP_CONFIG=config/configs/config_exp2_realllm.json \
  BENCHMARK_CSV=benchmark_dataset/agreement_benchmark_sample.csv \
  python experiments/exp2_agreement_analysis.py
EXP_CONFIG=config/configs/config_exp3_realllm.json \
  QUESTIONS_PATH=benchmark_dataset/adrd_questions_sample.json \
  python experiments/exp3_rubric_sensitivity.py
EXP_CONFIG=config/configs/config_exp4_realllm.json \
  EXP2_RESULTS_PATH=results/exp2_agreement_results_realllm.json \
  python experiments/exp4_boxplot_agreement.py
```

## Caveats

* The CPU substitutes are general-purpose small instruct LLMs, **not**
  medical fine-tunes. Pipeline behaviour is real LLM behaviour, but
  clinical-domain calibration of the original 4B-7B medical judges is
  not replicated.
* Sample size is small (8 questions); per-domain rows in box plots
  are single-point series. The figures are infrastructure validation,
  not statistical evidence.
* The "verification rate" line printed by `exp2_agreement_analysis.py`
  still treats matches across 5 rubrics as if they were drawn from a
  single 8-question pool (so the printed line can exceed 100%); this
  is pre-existing behaviour and not a real-LLM-specific artefact.
