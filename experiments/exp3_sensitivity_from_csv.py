from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "benchmark_dataset" / "1000_questions_dataset.csv"
MERGED_CSV_PATH = Path("/content/drive/MyDrive/judge_outputs/all_judges_item_scores_merged.csv")
OUTPUT_PATH = ROOT / "results" / "exp3_sensitivity_results.json"
SUMMARY_CSV_PATH = ROOT / "results" / "exp3_summary_table.csv"

JUDGES = ["medgemma", "biomistral", "meditron", "medalpaca"]

RUBRIC_PAIR = ("rubric1_pemat", "rubric5_pemat_likert")


def normalize_score(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"NA", "N/A", "NONE", ""}:
        return "NA"
    try:
        v = float(s)
        return int(v) if v.is_integer() else v
    except Exception:
        return s


def agreement_ratio(scores_a: Dict[str, Any], scores_b: Dict[str, Any]):
    common_items = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not common_items:
        return None
    matched, total = 0, 0
    for item_id in common_items:
        a = normalize_score(scores_a[item_id])
        b = normalize_score(scores_b[item_id])
        if a is None or b is None:
            continue
        total += 1
        if a == b:
            matched += 1
    return matched / total if total else None


def classify_panel(pairwise_scores: List[Any], threshold: float = 0.8) -> str:
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


def compute_panel_result(panel_df: pd.DataFrame) -> Dict[str, Any]:
    judge_item_scores = build_item_scores(panel_df)
    available_judges = sorted(judge_item_scores.keys())
    pairwise = []
    for i in range(len(available_judges)):
        for j in range(i + 1, len(available_judges)):
            ja, jb = available_judges[i], available_judges[j]
            ar = agreement_ratio(judge_item_scores.get(ja, {}), judge_item_scores.get(jb, {}))
            pairwise.append(ar)
    panel_class = classify_panel(pairwise, threshold=0.8)
    mean_pw = (sum(x for x in pairwise if x is not None) / len([x for x in pairwise if x is not None])
               if any(x is not None for x in pairwise) else None)
    return {
        "n_judges": len(available_judges),
        "pairwise_agreement": pairwise,
        "panel_agreement_class": panel_class,
        "mean_pairwise_agreement": mean_pw,
    }


def main():
    print("=" * 70)
    print("Experiment 3: Rubric Sensitivity (Scoring-Scale Comparison)")
    print(f"Comparing: {RUBRIC_PAIR[0]}  vs  {RUBRIC_PAIR[1]}")
    print("=" * 70)

    dataset_df = pd.read_csv(DATASET_PATH, low_memory=False)
    all_df = pd.read_csv(MERGED_CSV_PATH, low_memory=False)

    all_df["question_id"] = pd.to_numeric(all_df["question_id"], errors="coerce")
    all_df["rubric_id"] = all_df["rubric_id"].astype(str)
    all_df["judge_id"] = all_df["judge_id"].astype(str)
    all_df["item_id"] = all_df["item_id"].astype(str)
    all_df = all_df[all_df["judge_id"].isin(JUDGES)]

    results: List[Dict[str, Any]] = []
    rubric_summaries: Dict[str, List[Dict[str, Any]]] = {r: [] for r in RUBRIC_PAIR}

    for rubric_id in RUBRIC_PAIR:
        sub_df = all_df[all_df["rubric_id"] == rubric_id]
        grouped = sub_df.groupby("question_id", dropna=False)

        for question_id, panel_df in grouped:
            question_rows = dataset_df[dataset_df["id"] == question_id]
            if question_rows.empty:
                continue
            qrow = question_rows.iloc[0]

            panel_result = compute_panel_result(panel_df)
            entry = {
                "question_id": int(question_id) if not pd.isna(question_id) else None,
                "domain": qrow.get("domain"),
                "rubric_id": rubric_id,
                **panel_result,
            }
            results.append(entry)
            rubric_summaries[rubric_id].append(entry)

        print(f"[{rubric_id}] processed {len(rubric_summaries[rubric_id])} questions")

    summary_rows = []
    for rubric_id, entries in rubric_summaries.items():
        n = len(entries)
        if n == 0:
            continue
        mean_pw_vals = [e["mean_pairwise_agreement"] for e in entries if e["mean_pairwise_agreement"] is not None]
        mean_pw = sum(mean_pw_vals) / len(mean_pw_vals) * 100 if mean_pw_vals else None
        class_counts = Counter(e["panel_agreement_class"] for e in entries)
        summary_rows.append({
            "rubric_id": rubric_id,
            "n": n,
            "mean_pw_%": round(mean_pw, 1) if mean_pw is not None else None,
            "fully_%": round(100 * class_counts.get("fully_agree", 0) / n, 1),
            "majority_%": round(100 * class_counts.get("majority_agree", 0) / n, 1),
            "split_%": round(100 * class_counts.get("split", 0) / n, 1),
            "full_disagree_%": round(100 * class_counts.get("full_disagree", 0) / n, 1),
            "skipped_%": round(100 * class_counts.get("skipped", 0) / n, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nRubric Sensitivity Summary:")
    print(summary_df.to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print(f"\nSaved detailed results -> {OUTPUT_PATH}")
    print(f"Saved summary table -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
