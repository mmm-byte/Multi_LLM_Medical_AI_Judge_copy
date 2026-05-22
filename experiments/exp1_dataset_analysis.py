"""Experiment 1: Dataset Analysis.

Reads the pre-built benchmark_dataset_500.csv (525 rows, 5 domains,
105 questions per domain) and generates the paper's Table 1.

No raw source CSV copying required — benchmark_dataset_500.csv is
already in the repo at benchmark_dataset/source_datasets/.

Run:
    python experiments/exp1_dataset_analysis.py

Outputs:
    benchmark_dataset/dataset_table.md
    results/exp1_dataset_table.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os as _os

CONFIG_PATH  = Path(_os.environ.get(
    'EXP_CONFIG',
    str(ROOT / 'config' / 'configs' / 'config_exp1_dataset.json'),
))
DATASET_PATH = Path(_os.environ.get(
    'DATASET_PATH',
    str(ROOT / 'benchmark_dataset' / 'source_datasets' / 'benchmark_dataset_500.csv'),
))

DOMAIN_DEFINITIONS = {
    'Cardiology':   {'summary': 'STEMI management, 65-year-old, inferior ST elevation',
                     'example_q': 'A 65-year-old presents with inferior ST elevation. What is the management?'},
    'Pharmacology': {'summary': 'Metformin contraindications and renal safety criteria',
                     'example_q': 'What are the contraindications for metformin use in patients with renal impairment?'},
    'Neurology':    {'summary': 'Thunderclap headache workup, SAH rule-out protocol',
                     'example_q': 'A patient presents with a thunderclap headache. How do you rule out SAH?'},
    'Pediatrics':   {'summary': '2-month vaccination schedule, US CDC ACIP guidelines',
                     'example_q': 'What vaccines are recommended at the 2-month well-child visit per CDC ACIP?'},
    'Emergency':    {'summary': 'BLS protocol, unresponsive non-breathing patient',
                     'example_q': 'What are the steps for BLS in an unresponsive, non-breathing adult?'},
}


def generate_table_md(df: pd.DataFrame) -> str:
    counts  = df['domain'].value_counts().sort_index()
    sources = df['source'].value_counts()
    lines = [
        '# Benchmark Dataset — Domain Question Summary',
        '',
        '## Domain Overview',
        '',
        '| Domain | Question Summary | # Questions |',
        '|---|---|---|',
    ]
    for domain, defn in DOMAIN_DEFINITIONS.items():
        n = counts.get(domain, 0)
        lines.append(f'| {domain} | {defn["summary"]} | {n} |')
    lines += [
        f'| **Total** | | **{len(df)}** |',
        '',
        '## Representative Questions per Domain',
        '',
        '| Domain | Representative Question |',
        '|---|---|',
    ]
    for domain, defn in DOMAIN_DEFINITIONS.items():
        lines.append(f'| {domain} | {defn["example_q"]} |')
    lines += [
        '',
        '## Source Dataset Breakdown',
        '',
        '| Source | # Questions |',
        '|---|---|',
    ]
    for src, n in sources.items():
        lines.append(f'| {src} | {n} |')
    return '\n'.join(lines)


def main():
    if not DATASET_PATH.exists():
        print(f'ERROR: Dataset not found: {DATASET_PATH}')
        print('Expected: benchmark_dataset/source_datasets/benchmark_dataset_500.csv')
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    df = pd.read_csv(DATASET_PATH)
    print('=' * 70)
    print('Experiment 1: Dataset Analysis')
    print(f'Dataset: {DATASET_PATH}')
    print('=' * 70)
    print(f'Total rows  : {len(df)}')
    print(f'Columns     : {list(df.columns)}')

    counts  = df['domain'].value_counts().sort_index()
    sources = df['source'].value_counts()

    print('\nDomain breakdown:')
    for domain in DOMAIN_DEFINITIONS:
        print(f'  {domain:15s}: {counts.get(domain, 0):4d}')
    print(f'  {"TOTAL":15s}: {len(df):4d}')

    print('\nSource breakdown:')
    for src, n in sources.items():
        print(f'  {src:25s}: {n:4d}')

    print('\n' + '=' * 70)
    print('BENCHMARK TABLE (paper Table 1):')
    print('=' * 70)
    print(f'{"Domain":<15} {"Question Summary":<55} {"#":>4}')
    print('-' * 76)
    for domain, defn in DOMAIN_DEFINITIONS.items():
        print(f'{domain:<15} {defn["summary"]:<55} {counts.get(domain, 0):>4}')
    print('-' * 76)
    print(f'{"Total":<15} {"": <55} {len(df):>4}')

    # Save markdown table
    table_md = generate_table_md(df)
    table_path = ROOT / 'benchmark_dataset' / 'dataset_table.md'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, 'w') as f:
        f.write(table_md)
    print(f'\nSaved table -> {table_path}')

    # Save results JSON
    results_path = ROOT / config['output_files']['results_json']
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'dataset_path':   str(DATASET_PATH),
            'total':          len(df),
            'domain_counts':  counts.to_dict(),
            'source_counts':  sources.to_dict(),
        }, f, indent=2)
    print(f'Saved results -> {results_path}')


if __name__ == '__main__':
    main()
