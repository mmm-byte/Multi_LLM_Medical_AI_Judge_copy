"""No-LLM showcase of the clinical QA agreement framework.

This script is intentionally lightweight: it demonstrates the agreement
taxonomy and benchmark examples without requiring vLLM servers. Use it in the
paper/repo walkthrough before running expensive judge experiments.

Run:
    python3 experiments/demo_agreement_showcases.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agreement import classify_panel_agreement, summarize_agreement

BENCHMARK_CSV = ROOT / "benchmark_dataset" / "agreement_benchmark.csv"
BUILDER = ROOT / "benchmark_dataset" / "build_agreement_dataset.py"


PAIRWISE_CASES: Dict[str, Dict[Tuple[str, str], float]] = {
    "fully_agree": {
        ("medgemma", "biomistral"): 94.0,
        ("medgemma", "meditron"): 91.0,
        ("medgemma", "biomedlm"): 89.0,
        ("biomistral", "meditron"): 92.0,
        ("biomistral", "biomedlm"): 90.0,
        ("meditron", "biomedlm"): 88.0,
    },
    "majority_agree": {
        ("medgemma", "biomistral"): 90.0,
        ("medgemma", "meditron"): 87.0,
        ("medgemma", "biomedlm"): 35.0,
        ("biomistral", "meditron"): 91.0,
        ("biomistral", "biomedlm"): 32.0,
        ("meditron", "biomedlm"): 28.0,
    },
    # "Neutral" in the walkthrough means no clear majority: two judges cluster
    # together and two judges cluster together, so the framework reports split.
    "neutral_split": {
        ("medgemma", "biomistral"): 88.0,
        ("medgemma", "meditron"): 52.0,
        ("medgemma", "biomedlm"): 48.0,
        ("biomistral", "meditron"): 50.0,
        ("biomistral", "biomedlm"): 46.0,
        ("meditron", "biomedlm"): 86.0,
    },
    "full_disagree": {
        ("medgemma", "biomistral"): 42.0,
        ("medgemma", "meditron"): 35.0,
        ("medgemma", "biomedlm"): 20.0,
        ("biomistral", "meditron"): 38.0,
        ("biomistral", "biomedlm"): 22.0,
        ("meditron", "biomedlm"): 30.0,
    },
}


def _symmetric(scores: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    out = {}
    for (a, b), score in scores.items():
        out[(a, b)] = score
        out[(b, a)] = score
    return out


def _ensure_benchmark() -> None:
    if BENCHMARK_CSV.exists():
        return
    subprocess.run([sys.executable, str(BUILDER)], check=True)


def _load_examples() -> Dict[str, Dict[str, str]]:
    _ensure_benchmark()
    examples: Dict[str, Dict[str, str]] = {}
    with open(BENCHMARK_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row["expected_class"]
            examples.setdefault(cls, row)
    return examples


def main() -> None:
    judges = ["medgemma", "biomistral", "meditron", "biomedlm"]
    examples = _load_examples()

    print("=" * 72)
    print("Clinical QA Multi-LLM-as-Judge Showcase (no LLM calls)")
    print("=" * 72)
    print("Agreement threshold: 80% pairwise agreement\n")

    for display_name, pairwise in PAIRWISE_CASES.items():
        matrix = _symmetric(pairwise)
        agreement_class, outlier = classify_panel_agreement(matrix, judges, 80.0)
        summary = summarize_agreement(matrix, judges, 80.0)
        example_key = "split" if display_name == "neutral_split" else agreement_class
        example = examples.get(example_key, {})

        print("-" * 72)
        print(f"SHOWCASE: {display_name}")
        print(f"Framework class: {agreement_class}")
        print(f"Outlier judge: {outlier or 'none'}")
        print(f"Mean pairwise agreement: {summary['mean_pairwise_agreement']}%")
        if example:
            print(f"Example row: {example['id']} ({example['domain']})")
            print(f"Question: {example['question']}")
            print(f"Answer: {example['reference_answer']}")
            print(f"Why included: {example['rationale']}")
        print("Pairwise scores:")
        for pair in summary["pairwise_scores"]:
            print(f"  {pair['judge_a']} vs {pair['judge_b']}: {pair['score']}%")

    print("\nUse Exp2 for real LLM outputs; this script only explains the mechanics.")


if __name__ == "__main__":
    main()
