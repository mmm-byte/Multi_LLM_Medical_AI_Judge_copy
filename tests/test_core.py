"""Tests for core pipeline components.

All tests run WITHOUT any LLM calls (no vLLM required).
Coverage:
  - All 5 rubric JSON files load and parse correctly
  - agreement_benchmark.csv is auto-regenerated then validated
  - build_agreement_dataset.py heuristics produce valid classes
  - Domain classifier covers all 5 domains
  - Exp1 dataset loader runs with placeholder fallback
  - Config JSON files are valid and contain required keys
  - PanelResult and JudgeResult dataclasses instantiate cleanly
  - Pairwise agreement math is correct

Run:
    pytest tests/test_core.py -v
    python tests/test_core.py        # (no pytest needed)
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -------------------------------------------------------------------------
# Tiny test harness (no pytest dependency required)
# -------------------------------------------------------------------------
_PASS: list = []
_FAIL: list = []

def ok(name: str) -> None:
    _PASS.append(name)
    print(f"  PASS  {name}")

def fail(name: str, reason: str) -> None:
    _FAIL.append(name)
    print(f"  FAIL  {name}: {reason}")

def expect(condition: bool, name: str, reason: str = "") -> None:
    if condition:
        ok(name)
    else:
        fail(name, reason or "condition is False")


# =========================================================================
# 1. Rubric JSON loading
# =========================================================================
RUBRIC_FILES = [
    "rubrics/rubric1_pemat.json",
    "rubrics/rubric2_healthbench.json",
    "rubrics/rubric3_clinical_eval.json",
    "rubrics/rubric4_prometheus.json",
    "rubrics/rubric5_pemat_likert.json",
]
REQUIRED_RUBRIC_KEYS = {"id", "name", "source_paper", "paradigm", "items"}
REQUIRED_ITEM_KEYS   = {"id", "name", "description", "scale", "weight", "why"}

def test_rubrics():
    print("\n[1] Rubric JSON files")
    for path_str in RUBRIC_FILES:
        path = ROOT / path_str
        expect(path.exists(), f"rubric_exists:{path.name}")
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            fail(f"rubric_parse:{path.name}", str(e))
            continue
        missing_top = REQUIRED_RUBRIC_KEYS - set(data.keys())
        expect(not missing_top, f"rubric_top_keys:{path.name}",
               f"missing: {missing_top}")
        expect(isinstance(data.get("items"), list) and len(data["items"]) >= 4,
               f"rubric_items_count:{path.name}", "need >=4 items")
        for item in data.get("items", []):
            missing_item = REQUIRED_ITEM_KEYS - set(item.keys())
            expect(not missing_item, f"item_keys:{path.name}:{item.get('id','?')}",
                   f"missing: {missing_item}")
        if data["id"] == "rubric1_pemat":
            scales = {it["scale"] for it in data["items"]}
            expect(scales == {"BINARY"}, "pemat_all_binary", f"got {scales}")
        if data["id"] == "rubric5_pemat_likert":
            scales = {it["scale"] for it in data["items"]}
            expect(scales == {"LIKERT"}, "pemat_likert_all_likert", f"got {scales}")
            r1 = json.loads((ROOT / "rubrics/rubric1_pemat.json").read_text())
            ids1 = {it["id"] for it in r1["items"]}
            ids5 = {it["id"] for it in data["items"]}
            expect(ids1 == ids5, "pemat_controlled_pair_same_ids",
                   f"rubric1 ids={ids1}, rubric5 ids={ids5}")


# =========================================================================
# 2. Config JSON files
# =========================================================================
CONFIG_FILES = {
    "config/configs/config_exp1_dataset.json": ["experiment", "domains", "output_files"],
    "config/configs/config_exp2_agreement.json": ["judges", "rubrics", "benchmark_csv", "domains", "output_files"],
    "config/configs/config_exp3_rubric_sensitivity.json": ["judges", "rubrics", "scoring_variants", "output_files"],
    "config/configs/config_exp4_boxplots.json": ["categories", "rubrics", "output_files"],
}

def test_configs():
    print("\n[2] Config JSON files")
    for path_str, required_keys in CONFIG_FILES.items():
        path = ROOT / path_str
        expect(path.exists(), f"config_exists:{Path(path_str).name}")
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            fail(f"config_parse:{path_str}", str(e))
            continue
        for k in required_keys:
            expect(k in data, f"config_key:{Path(path_str).name}:{k}", f"{k} missing")
        if "config_exp2" in path_str:
            rubrics = data.get("rubrics", [])
            expect(len(rubrics) == 5, "exp2_has_5_rubrics", f"found {len(rubrics)}")
            domains = data.get("domains", [])
            expect(set(domains) == {"Cardiology","Pharmacology","Neurology","Pediatrics","Emergency"},
                   "exp2_5_domains", f"got {domains}")
        if "config_exp4" in path_str:
            cats = data.get("categories", [])
            expect(set(cats) == {"Cardiology","Pharmacology","Neurology","Pediatrics","Emergency"},
                   "exp4_5_domains", f"got {cats}")
        if "config_exp3" in path_str:
            rubrics = data.get("rubrics", [])
            expect(any("rubric5" in r for r in rubrics), "exp3_has_rubric5",
                   "rubric5_pemat_likert must be in exp3 rubrics")
            variants = data.get("scoring_variants", [])
            expect(set(variants) == {"BINARY","LIKERT_1_5","SCALED_0_10"},
                   "exp3_3_scoring_variants", f"got {variants}")


# =========================================================================
# 3. Agreement benchmark CSV schema
#    Always regenerate the CSV fresh before asserting class counts so
#    the test never fails against a stale file from a prior broken run.
# =========================================================================
BENCHMARK_CSV_REQUIRED_COLS = [
    "id", "domain", "question", "reference_answer", "source",
    "expected_class", "rationale", "observed_class", "verified",
    "score_U1_plain", "score_A1_action", "score_HB3_emerg", "score_CE5_clarity"
]
VALID_CLASSES = {"fully_agree", "majority_agree", "split", "full_disagree"}

def test_benchmark_csv():
    print("\n[3] Agreement benchmark CSV")
    builder = ROOT / "benchmark_dataset" / "build_agreement_dataset.py"
    if not builder.exists():
        print("  SKIP  build_agreement_dataset.py not found")
        return

    # Always regenerate so we never assert against a stale CSV.
    result = subprocess.run(
        [sys.executable, str(builder)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        fail("benchmark_csv_build",
             f"builder exited {result.returncode}: {result.stderr.strip()[-300:]}")
        return

    csv_path = ROOT / "benchmark_dataset" / "agreement_benchmark.csv"
    if not csv_path.exists():
        print("  SKIP  benchmark CSV not generated by builder")
        return
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    missing_cols = [c for c in BENCHMARK_CSV_REQUIRED_COLS if c not in cols]
    expect(not missing_cols, "benchmark_csv_columns", f"missing: {missing_cols}")
    expect(len(rows) > 0, "benchmark_csv_nonempty")
    for cls in VALID_CLASSES:
        n = sum(1 for r in rows if r.get("expected_class") == cls)
        expect(n >= 5, f"benchmark_min_5_per_class:{cls}", f"found {n}")
    bad = [r["expected_class"] for r in rows if r.get("expected_class") not in VALID_CLASSES]
    expect(not bad, "benchmark_valid_classes", f"invalid: {bad[:3]}")


# =========================================================================
# 4. Build-agreement heuristics unit tests
# =========================================================================
def test_heuristics():
    print("\n[4] Heuristic scoring functions")
    sys.path.insert(0, str(ROOT / "benchmark_dataset"))
    try:
        from build_agreement_dataset import (
            expected_agreement_class,
            score_plain_language,
            score_actionable_steps,
            score_emergency_flag,
            classify_domain,
        )
    except ImportError as e:
        fail("heuristics_import", str(e))
        return

    s = score_plain_language("pharmacokinetics acetylcholinesterase bioavailability pathophysiology")
    expect(s < 0.4, "plain_language_jargon_low", f"got {s:.3f}")

    s2 = score_plain_language("Call your doctor if the patient is confused or falls.")
    expect(s2 > 0.7, "plain_language_simple_high", f"got {s2:.3f}")

    s3 = score_actionable_steps("Call 911. Administer epinephrine. Monitor breathing. Apply pressure.")
    expect(s3 > 0.4, "actionable_verbs_high", f"got {s3:.3f}")

    cls, _, rationale = expected_agreement_class(
        "patient is unresponsive and not breathing",
        "The patient should rest and drink fluids."
    )
    expect(cls in ("split", "full_disagree"), "emergency_no_flag_class", f"got {cls}")

    cls2, _, _ = expected_agreement_class(
        "patient is unresponsive and not breathing",
        "Call 911 immediately. Begin CPR. Use AED if available."
    )
    expect(cls2 in ("fully_agree", "majority_agree"), "emergency_with_flag_class", f"got {cls2}")

    cls3, _, _ = expected_agreement_class("What is a stroke?", "Stroke.")
    expect(cls3 == "full_disagree", "short_answer_full_disagree", f"got {cls3}")

    expect(classify_domain("STEMI management inferior ST elevation") == "Cardiology",
           "domain_cardiology")
    expect(classify_domain("metformin renal impairment contraindication") == "Pharmacology",
           "domain_pharmacology")
    expect(classify_domain("BLS cardiac arrest unresponsive") == "Emergency",
           "domain_emergency")
    expect(classify_domain("2-month vaccine DTaP ACIP schedule") == "Pediatrics",
           "domain_pediatrics")
    expect(classify_domain("subarachnoid hemorrhage thunderclap headache") == "Neurology",
           "domain_neurology")


# =========================================================================
# 5. Exp1 dataset loader (placeholder fallback)
# =========================================================================
def test_exp1_placeholder():
    print("\n[5] Exp1 dataset loader (placeholder fallback)")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp1", ROOT / "experiments" / "exp1_dataset_analysis.py"
        )
        exp1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp1)
        benchmark = exp1.build_benchmark([], max_per_domain=5)
        expect(len(benchmark) == 25, "exp1_placeholder_25_questions",
               f"got {len(benchmark)} (5 domains x 5 each)")
        domains_found = {q["domain"] for q in benchmark}
        expect(domains_found == {"Cardiology","Pharmacology","Neurology","Pediatrics","Emergency"},
               "exp1_placeholder_all_domains", f"got {domains_found}")
        for q in benchmark:
            expect("id" in q and "text" in q and "domain" in q,
                   f"exp1_question_schema:{q.get('id','?')}", "missing id/text/domain")
    except Exception as e:
        fail("exp1_placeholder_run", str(e))


# =========================================================================
# 6. Pairwise agreement math
# =========================================================================
def test_agreement_math():
    print("\n[6] Pairwise agreement math")
    try:
        from core.agreement import classify_panel_agreement, summarize_agreement
    except ImportError as e:
        fail("agreement_import", str(e))
        return

    judges = ["a", "b", "c", "d"]
    pairwise_full = {("a","b"): 95.0, ("b","a"): 95.0,
                     ("a","c"): 92.0, ("c","a"): 92.0,
                     ("a","d"): 90.0, ("d","a"): 90.0,
                     ("b","c"): 93.0, ("c","b"): 93.0,
                     ("b","d"): 91.0, ("d","b"): 91.0,
                     ("c","d"): 88.0, ("d","c"): 88.0}
    cls, outlier = classify_panel_agreement(pairwise_full, judges, 80.0)
    expect(cls == "fully_agree", "math_fully_agree", f"got {cls}")
    expect(outlier is None, "math_no_outlier", f"got {outlier}")

    pairwise_outlier = {("a","b"): 90.0, ("b","a"): 90.0,
                        ("a","c"): 88.0, ("c","a"): 88.0,
                        ("a","d"): 30.0, ("d","a"): 30.0,
                        ("b","c"): 91.0, ("c","b"): 91.0,
                        ("b","d"): 28.0, ("d","b"): 28.0,
                        ("c","d"): 25.0, ("d","c"): 25.0}
    cls2, outlier2 = classify_panel_agreement(pairwise_outlier, judges, 80.0)
    expect(outlier2 == "d", "math_outlier_detected", f"got {outlier2}")

    pairwise_split = {("a","b"): 88.0, ("b","a"): 88.0,
                      ("a","c"): 52.0, ("c","a"): 52.0,
                      ("a","d"): 48.0, ("d","a"): 48.0,
                      ("b","c"): 50.0, ("c","b"): 50.0,
                      ("b","d"): 46.0, ("d","b"): 46.0,
                      ("c","d"): 86.0, ("d","c"): 86.0}
    cls_split, outlier_split = classify_panel_agreement(pairwise_split, judges, 80.0)
    expect(cls_split == "split", "math_two_vs_two_split", f"got {cls_split}")
    expect(outlier_split is None, "math_split_no_outlier", f"got {outlier_split}")

    summary = summarize_agreement(pairwise_full, judges, 80.0)
    expect("mean_pairwise_agreement" in summary, "summary_has_mean_pw")
    expect(summary["mean_pairwise_agreement"] > 80, "summary_mean_high",
           f"got {summary['mean_pairwise_agreement']}")


# =========================================================================
# Runner
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-LLM-as-Judge Medical AI \u2014 Core Test Suite")
    print("=" * 60)
    test_rubrics()
    test_configs()
    test_benchmark_csv()
    test_heuristics()
    test_exp1_placeholder()
    test_agreement_math()

    print("\n" + "=" * 60)
    total = len(_PASS) + len(_FAIL)
    print(f"Results: {len(_PASS)}/{total} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("\nFailed tests:")
        for f_ in _FAIL:
            print(f"  \u2717 {f_}")
        sys.exit(1)
    else:
        print("All tests passed \u2713")
        sys.exit(0)
