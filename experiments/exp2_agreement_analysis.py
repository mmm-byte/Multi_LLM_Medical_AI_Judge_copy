"""Experiment 2: Per-Rubric Agreement Analysis (Core Experiment).

Loads benchmark_dataset/source_datasets/benchmark_dataset_500.csv
(525 rows, 5 domains) and runs all judges on every row under all 4 rubrics.

For each (question, rubric) pair:
  - Sends question + reference_answer to all judges via vLLM
  - Computes pairwise agreement across all judge pairs
  - Classifies panel agreement: fully_agree / majority_agree / split / full_disagree / skipped
  - Prints all judge rationales to stdout

Partial results are written after each rubric so exp4 can run even
if the pipeline is interrupted.

Env vars:
  MAX_QUESTIONS  -- cap total questions, sampled evenly across domains
                    Default: 100 (= 20 per domain across 5 domains).
                    Set to 0 to run all rows.
  DATASET_PATH   -- override dataset CSV path
  EXP_CONFIG     -- override config JSON path

Run on HPC:
    # Default: 100 questions, 20 per domain (balanced)
    python experiments/exp2_agreement_analysis.py

    # Full dataset run:
    MAX_QUESTIONS=0 python experiments/exp2_agreement_analysis.py

    # Custom size:
    MAX_QUESTIONS=50 python experiments/exp2_agreement_analysis.py

Outputs:
    results/exp2_agreement_results.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.consensus_core.models import Answer, Question, Rubric, RubricItem, new_id
from core.wrapper import ADRDJudgeRunner, PanelResult

import os as _os

CONFIG_PATH   = Path(_os.environ.get(
    'EXP_CONFIG',
    str(ROOT / 'config' / 'configs' / 'config_exp2_agreement.json'),
))
DATASET_PATH  = Path(_os.environ.get(
    'DATASET_PATH',
    str(ROOT / 'benchmark_dataset' / 'source_datasets' / 'benchmark_dataset_500.csv'),
))
# Default 100 = 20 questions per domain across 5 domains (balanced).
# Set MAX_QUESTIONS=0 in env to run the full dataset.
MAX_QUESTIONS = int(_os.environ.get('MAX_QUESTIONS', '100'))


def load_rubric(path: str) -> Rubric:
    with open(ROOT / path) as f:
        data = json.load(f)
    items = [
        RubricItem(
            id=it['id'], name=it['name'],
            description=it['description'],
            scale=it['scale'], weight=float(it['weight']),
            source_paper=data.get('source_paper', ''),
        )
        for it in data['items']
    ]
    return Rubric(
        id=data['id'], name=data['name'],
        source_paper=data['source_paper'],
        source_url=data.get('source_url'),
        items=items,
    )


def load_dataset(path: Path, max_questions: int) -> List[Dict]:
    """Load benchmark_dataset_500.csv. If max_questions > 0, sample evenly
    across domains so every domain is equally represented."""
    if not path.exists():
        print(f'ERROR: Dataset not found at {path}')
        print('Expected: benchmark_dataset/source_datasets/benchmark_dataset_500.csv')
        sys.exit(1)
    df = pd.read_csv(path)

    # Normalise column names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ('question', 'text', 'input', 'prompt'):
            col_map[col] = 'question'
        elif cl in ('reference_answer', 'answer', 'output', 'response'):
            col_map[col] = 'reference_answer'
        elif cl == 'domain':
            col_map[col] = 'domain'
        elif cl in ('id', 'question_id'):
            col_map[col] = 'id'
        elif cl == 'source':
            col_map[col] = 'source'
        elif cl in ('expected_class', 'expected_agreement'):
            col_map[col] = 'expected_class'
    df = df.rename(columns=col_map)
    if 'id' not in df.columns:
        df['id'] = [f'q_{i:04d}' for i in range(len(df))]
    if 'expected_class' not in df.columns:
        df['expected_class'] = ''

    # Even sampling across domains
    if max_questions > 0 and max_questions < len(df):
        domains = df['domain'].unique()
        per_domain = max(1, max_questions // len(domains))
        sampled = (
            df.groupby('domain', group_keys=False)
              .apply(lambda g: g.sample(n=min(per_domain, len(g)), random_state=42))
        )
        df = sampled.reset_index(drop=True)
        print(f'Sampled {len(df)} questions ({per_domain} per domain) from {path.name}')
    else:
        print(f'Loaded {len(df)} questions from {path.name} (full dataset)')

    domain_counts = Counter(df['domain'].tolist())
    for domain, n in sorted(domain_counts.items()):
        print(f'  {domain}: {n}')

    return df.to_dict('records')


def _print_rubric_summary(rubric_block: Dict) -> None:
    results = rubric_block['results']
    live    = [r for r in results if r.get('agreement_class') != 'skipped']
    skipped = len(results) - len(live)
    total_q = len(live)
    counts  = Counter(r['agreement_class'] for r in live)
    mean_pw = (sum(r['mean_pairwise_agreement'] for r in live) / total_q
               if total_q else 0)
    print(f"\n{rubric_block['rubric_name']}:")
    for cls in ['fully_agree', 'majority_agree', 'split', 'full_disagree']:
        n   = counts.get(cls, 0)
        pct = n / total_q * 100 if total_q else 0
        print(f'  {cls:<20}: {n:3d}/{total_q} ({pct:.1f}%)')
    if skipped:
        print(f'  {"skipped":<20}: {skipped:3d} rows')
    print(f'  Mean pairwise : {mean_pw:.1f}%')


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    benchmark_rows = load_dataset(DATASET_PATH, MAX_QUESTIONS)
    rubrics  = [load_rubric(r) for r in config['rubrics']]
    runner   = ADRDJudgeRunner(config_path=str(CONFIG_PATH))

    out_path = ROOT / config['output_files']['results_json']
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    already_done: set = set()
    if out_path.exists():
        try:
            with open(out_path) as f:
                all_results = json.load(f)
            already_done = {b['rubric_id'] for b in all_results}
            print(f'Resuming: {len(already_done)} rubric(s) done: {already_done}')
        except Exception:
            all_results = []

    total   = len(benchmark_rows) * len(rubrics)
    done    = len(already_done) * len(benchmark_rows)
    t_start = time.time()

    print(f'\nRunning {len(benchmark_rows)} questions x {len(rubrics)} rubrics = {total} rows total')
    if MAX_QUESTIONS > 0:
        print(f'(MAX_QUESTIONS={MAX_QUESTIONS} — sampled evenly across domains)')
    else:
        print('(MAX_QUESTIONS=0 — full dataset)')

    for rubric in rubrics:
        if rubric.id in already_done:
            print(f'\nSkipping already-done rubric: {rubric.id}')
            continue

        print(f"\n{'#'*70}")
        print(f'RUBRIC: {rubric.name}')
        print(f'SOURCE: {rubric.source_paper}')
        print(f"{'#'*70}")
        rubric_results = []

        for row in benchmark_rows:
            question = Question(
                id=str(row['id']),
                text=str(row.get('question', row.get('text', ''))),
                category=str(row.get('domain', '')),
                source=str(row.get('source', '')),
            )
            answer = Answer(
                id=new_id('ans'),
                text=str(row.get('reference_answer', '')),
                provider='reference',
            )
            expected = str(row.get('expected_class', ''))
            print(f"\n  Q [{question.category}] expected={expected}: {question.text[:70]}...")

            panel: PanelResult = runner.run(question, answer, rubric)
            observed = panel.agreement_class
            n_judges_ran = len([jr for jr in panel.judge_results
                                if jr.raw_response.strip() or any(
                                    str(s['score']).upper() not in ('', 'NA', 'N/A', 'NONE')
                                    for s in jr.scores
                                )])
            if n_judges_ran < len(runner.judges):
                print(f'  \u26a0\ufe0f  Only {n_judges_ran}/{len(runner.judges)} judges produced output')

            rubric_results.append(panel.to_dict())
            done += 1
            elapsed = time.time() - t_start
            match_flag = '\u2713' if observed == expected and expected else '-'
            print(f'  [{done}/{total}] {match_flag} expected={expected} '
                  f'observed={observed} | '
                  f'MeanPW={panel.mean_pairwise_agreement:.1f}% | '
                  f'judges={n_judges_ran} | {elapsed:.0f}s')
            if panel.outlier_judge:
                print(f'  \u26a0\ufe0f  Outlier: {panel.outlier_judge}')

        rubric_block = {
            'rubric_id':    rubric.id,
            'rubric_name':  rubric.name,
            'source_paper': rubric.source_paper,
            'results':      rubric_results,
        }
        all_results.append(rubric_block)

        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  \u2713 Partial results saved -> {out_path}')
        _print_rubric_summary(rubric_block)

    print('\n' + '='*70)
    print('AGREEMENT SUMMARY')
    print('='*70)
    for rubric_block in all_results:
        _print_rubric_summary(rubric_block)

    print(f'\nResults saved -> {out_path}')


if __name__ == '__main__':
    main()
