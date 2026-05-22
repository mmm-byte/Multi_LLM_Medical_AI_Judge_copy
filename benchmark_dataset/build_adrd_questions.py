"""Build benchmark_dataset/adrd_questions.json for exp3.

Exp3 needs a standalone question list (not a CSV) with fields:
  id, text, category, reference_answer

The filename is kept for backward compatibility with earlier ADRD-focused
experiments. The contents now come from the broad clinical QA agreement
benchmark and use the five clinical domains as categories.

Outputs:
    benchmark_dataset/adrd_questions.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / 'benchmark_dataset' / 'agreement_benchmark.csv'
OUT  = ROOT / 'benchmark_dataset' / 'adrd_questions.json'


def main() -> None:
    if not SRC.exists():
        print(f'ERROR: {SRC} not found. Run build_agreement_dataset.py first.')
        raise SystemExit(1)

    questions = []
    with open(SRC, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            questions.append({
                'id':               row['id'],
                'text':             row['question'],
                'category':         row['domain'],
                'reference_answer': row['reference_answer'],
                'source':           row.get('source', ''),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'metadata': {
            'source': 'agreement_benchmark.csv',
            'note': 'Legacy filename retained; categories are broad clinical QA domains.',
            'categories': sorted({q['category'] for q in questions}),
            'total_questions': len(questions),
        },
        'questions': questions,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'Written: {OUT}')
    print(f'  Questions: {len(questions)}')


if __name__ == '__main__':
    main()
