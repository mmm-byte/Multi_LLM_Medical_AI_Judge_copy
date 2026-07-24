from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "benchmark_dataset" / "source_datasets" / "benchmark_dataset_500.csv"
MERGED_CSV_PATH = Path("/content/drive/MyDrive/judge_outputs/all_judges_item_scores_merged.csv")
OUTPUT_PATH = ROOT / "results" / "exp2_agreement_results.json"

JUDGES = ["medgemma", "biomistral", "meditron", "medalpaca"]

def normalize_score(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"NA", "N/A", "NONE", ""}:
        return "NA"
    try:
        v = float(s)
        if v.is_integer():
            return int(v)
        return v
    except Exception:
        return s

def agreement_ratio(scores_a: Dict[str, Any], scores_b: Dict[str, Any]) -> float | None:
    common_items = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not common_items:
        return None

    matched = 0
    total = 0
    for item_id in common_items:
        a = normalize_score(scores_a[item_id])
        b = normalize_score(scores_b[item_id])

        if a is None or b is None:
            continue

        total += 1
        if a == b:
            matched += 1

    if total == 0:
        return None
    return matched / total

def classify_panel(pairwise_scores: List[float | None], threshold: float = 0.8) -> str:
    valid = [x for x in pairwise_scores if x is not None]
    if len(valid) < 3:
        return "skipped"

    n_good = sum(x >= threshold for x in valid)

    if n_good == len(valid):
        return "fully_agree"
    if n_good >= 3:
        return "majority_agree"
    if n_good == 0:
        return "full_disagree"
    return "split"

def build_item_scores(panel_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    judge_item_scores = {}
    for judge_id, gdf in panel_df.groupby("judge_id"):
        item_scores = {}
        for _, row in gdf.iterrows():
            item_id = row["item_id"]
            if pd.isna(item_id):
                continue
            if str(row.get("parse_status", "ok")).strip().lower() == "failed":
                continue
            item_scores[str(item_id)] = row.get("score")
        judge_item_scores[judge_id] = item_scores
    return judge_item_scores

def build_rationales(panel_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out = defaultdict(dict)
    for _, row in panel_df.iterrows():
        judge_id = row["judge_id"]
        item_id = row["item_id"]
        rationale = row.get("rationale", "")
        if pd.notna(item_id):
            out[judge_id][str(item_id)] = "" if pd.isna(rationale) else str(rationale)
    return dict(out)

def main():
    print("=" * 70)
    print("Experiment 2: Per-Rubric Agreement Analysis FROM EXISTING CSV")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Merged judge CSV: {MERGED_CSV_PATH}")
    print("=" * 70)

    dataset_df = pd.read_csv(DATASET_PATH, low_memory=False)
    all_df = pd.read_csv(MERGED_CSV_PATH, low_memory=False)

    all_df = all_df.copy()
    all_df["question_id"] = pd.to_numeric(all_df["question_id"], errors="coerce")
    all_df["rubric_id"] = all_df["rubric_id"].astype(str)
    all_df["judge_id"] = all_df["judge_id"].astype(str)
    all_df["item_id"] = all_df["item_id"].astype(str)

    all_df = all_df[all_df["judge_id"].isin(JUDGES)]

    results: List[Dict[str, Any]] = []

    grouped = all_df.groupby(["question_id", "rubric_id"], dropna=False)
    total_groups = len(grouped)

    for idx, ((question_id, rubric_id), panel_df) in enumerate(grouped, start=1):
        question_rows = dataset_df[dataset_df["id"] == question_id]
        if question_rows.empty:
            continue

        qrow = question_rows.iloc[0]

        judge_item_scores = build_item_scores(panel_df)
        rationales = build_rationales(panel_df)

        available_judges = sorted(judge_item_scores.keys())
        pairwise = []

        for i in range(len(available_judges)):
            for j in range(i + 1, len(available_judges)):
                ja = available_judges[i]
                jb = available_judges[j]
                ar = agreement_ratio(judge_item_scores.get(ja, {}), judge_item_scores.get(jb, {}))
                pairwise.append({
                    "judge_a": ja,
                    "judge_b": jb,
                    "agreement": ar,
                })

        pairwise_vals = [x["agreement"] for x in pairwise]
        panel_class = classify_panel(pairwise_vals, threshold=0.8)

        result = {
            "question_id": int(question_id) if not pd.isna(question_id) else None,
            "domain": qrow.get("domain"),
            "source_dataset": qrow.get("source"),
            "question": qrow.get("question"),
            "reference_answer": qrow.get("answer"),
            "rubric_id": rubric_id,
            "judges_available": available_judges,
            "n_judges": len(available_judges),
            "pairwise_agreement": pairwise,
            "panel_agreement_class": panel_class,
            "judge_item_scores": judge_item_scores,
            "judge_rationales": rationales,
        }
        results.append(result)

        if idx % 50 == 0 or idx == total_groups:
            print(f"[{idx}/{total_groups}] processed")

    summary = Counter(r["panel_agreement_class"] for r in results)

    output = {
        "metadata": {
            "dataset_path": str(DATASET_PATH),
            "merged_csv_path": str(MERGED_CSV_PATH),
            "n_results": len(results),
            "judges": JUDGES,
            "agreement_threshold": 0.8,
            "note": "Derived from existing merged judge CSV; no live judge calls.",
        },
        "summary": dict(summary),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved results ->", OUTPUT_PATH)
    print("Agreement summary:")
    for k, v in summary.items():
        print(f"  {k:16s}: {v}")

if __name__ == "__main__":
    main()
