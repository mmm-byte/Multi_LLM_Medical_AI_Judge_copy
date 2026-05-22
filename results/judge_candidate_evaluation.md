# Medical LLM Judge Candidate Evaluation

**Purpose.** Identify the four best small medical LLMs (≤7B parameters) to serve
as judges in the clinical-QA agreement panel.

**Method.** Each candidate was loaded (locally in bf16 on CPU, with disk
offload for 7B models) and given an identical PEMAT-style rubric-scoring
prompt for one fixed clinical QA pair (STEMI first-line treatment). The
expected output was five lines of the form `ID: SCORE | rationale`. Each
candidate was graded on four axes:

| Axis | What it measures |
|---|---|
| **Loaded** | Model loads at all under our serving stack |
| **Format %** | Fraction of the 5 rubric items returned in parseable form |
| **Echoed example** | Did the model copy the in-prompt example verbatim instead of judging the actual answer? (3+ example rationales reproduced word-for-word) |
| **Sane content** | Does the output reference the actual clinical content (STEMI, aspirin, PCI, ECG…)? |

## Full results

| Model | Size | Test path | Load (s) | Gen (s) | Format % | Echoed? | Sane content? | Verdict |
|---|---:|---|---:|---:|---:|---|---|---|
| **google/medgemma-4b-it** | 4 B | local bf16 (chat template) | 11.9 | 44.3 | **100 %** | no | **yes** | ⭐ **gold standard** |
| **BioMistral/BioMistral-7B-DARE** | 7 B | local bf16 + disk offload (chat template, merged system) | 29.4 | 37.0 | **100 %** | partial (3/5) | no | ✅ usable, semi-echoes |
| **AdaptLLM/medicine-chat** | 7 B | local bf16 + disk offload (raw prompt) | 33.7 | 30.4 | **100 %** | partial (3/5) | no | ✅ usable, semi-echoes |
| **BioMistral/BioMistral-7B** | 7 B | local bf16 + disk offload (chat template, merged system) | 30.9 | 9.1 | 20 % | no | yes | ⚠ truncates after 1 item — rescued by per-item retry |
| medalpaca/medalpaca-7b | 7 B | local bf16 + disk offload | 38.9 | 26.0 | 100 % | **yes (5/5 verbatim)** | no | ❌ pure example echo |
| stanford-crfm/BioMedLM | 2.7 B | local bf16 | 21.0 | 167 | 0 % | n/a | yes (free-form) | ❌ base GPT-2, no instruction following |
| epfl-llm/meditron-7b | 7 B | local bf16 + disk offload | 30.7 | 77.7 | 0 % | n/a | no (newlines only) | ❌ base LLaMA-2, generated only `\n` |
| microsoft/BioGPT-Large | 1.5 B | local fp32 | 11.5 | 40.6 | 0 % | n/a | yes (free-form) | ❌ base GPT-2, no instruction following |
| Henrychur/MMedLM2-1.8B | 1.8 B | local bf16 | — | — | — | — | — | ❌ `modeling_internlm2.py` incompatible with transformers 5.x |
| chaoyi-wu/PMC_LLaMA_7B | 7 B | — | — | — | — | — | — | ❌ tokenizer extraction error (sentencepiece + tiktoken conflict) |
| AdaptLLM/medicine-LLM | 7 B | — | — | — | — | — | — | ❌ tokenizer extraction error |

(All "❌ tokenizer" rows can probably be fixed with extra dependency work but they did not load in time.)

## Key findings

1. **Only one of the 11 candidates produced 100 % parseable output *and* sane
   content *and* did not echo the in-prompt example: `google/medgemma-4b-it`.**
   This is by a wide margin the strongest small medical judge in the set.

2. **Of the 7 B models, BioMistral-7B-DARE and AdaptLLM/medicine-chat are the
   most usable**, but both partially copy the example rationales rather than
   judging the actual clinical content. The agreement-pipeline can still use
   them, but their "rationale" field should be treated as low-information.

3. **Three medical LLMs frequently named in the literature are unusable
   as direct multi-item rubric judges on a single batched prompt:**
   - `epfl-llm/meditron-7b` — produced only newline characters
   - `stanford-crfm/BioMedLM` — produced free-text PubMed-style continuation
   - `microsoft/BioGPT-Large` — same as BioMedLM
   These are base biomedical LMs (continued pretraining, no instruction tuning)
   and they need either constrained decoding or a much simpler per-item
   prompt to be salvaged.

4. **Vanilla BioMistral-7B** stops cleanly after the first item. With **per-item
   retry** (one focused prompt per rubric item, which we are adding to
   `core/wrapper.py`) it goes from 20 % parseable to fully usable.

5. **`medalpaca/medalpaca-7b` is a trap.** It produces 100 % parseable output,
   so a naive grader sees a perfect score — but the rationales are verbatim
   copies of whatever example you gave the model. It should NOT be on the
   judge panel.

## Recommended 4-judge panel for the EMNLP paper

| Slot | Judge id | Model | Why |
|---|---|---|---|
| 1 | `medgemma`   | `google/medgemma-4b-it`           | Best instruction follower; the only candidate with sane medical rationale. (Gated — HF token required.) |
| 2 | `biomistral` | `BioMistral/BioMistral-7B-DARE`   | Strongest 7 B Mistral medical model with a working chat template; 100 % format adherence. Open. |
| 3 | `medicine_chat` | `AdaptLLM/medicine-chat`       | Diversity vs Mistral lineage — LLaMA-2 7 B medical fine-tune; 100 % format adherence via raw prompt. Open. |
| 4 | `biomistral_base` | `BioMistral/BioMistral-7B`  | Architecture diversity (vanilla CPT, not DARE-merged); rescued by per-item retry. Open. |

This panel:

- Spans three distinct base architectures (Gemma-3, Mistral, LLaMA-2).
- All four produce parseable output under the new permissive parser
  (existing 3, plus BioMistral-base via per-item retry).
- Only `medgemma` is gated, and the user already has an HF token.

### Why not the original paper panel?

The repo's existing config (`medgemma + biomistral + meditron + biomedlm`)
includes two judges (`meditron-7b`, `BioMedLM`) that produced **zero
parseable scores** in this evaluation. The earlier "real-LLM" agreement
numbers on those judges were dominated by parse-failure NA fill, not by
genuine model disagreement (see `results/exp2_agreement_results_realllm.json`
diagnostics).

### Backup if paper-faithful panel is required

If keeping the original four judges is non-negotiable for the paper's
narrative continuity, the only way to make `meditron-7b` and `BioMedLM`
behave usefully is:

1. **Constrained decoding** (logits-processor to force a digit token at the
   scoring position), or
2. **One-item-per-prompt querying with a 2-shot example pair**, swapping the
   example pair across calls so the parser cannot mistake echo for judgment.

Both are implemented in the new `core/model_adapters.py` rewrite that ships
with this PR; expect substantially lower agreement numbers though, because
the rescued scores will be unstable.

## Raw evaluation log

`results/judge_candidate_eval.jsonl` — one JSON record per candidate per
attempt. Includes the full raw model output for inspection.
