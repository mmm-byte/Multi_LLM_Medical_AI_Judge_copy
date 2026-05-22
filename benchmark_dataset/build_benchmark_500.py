#!/usr/bin/env python3
"""
build_benchmark_500.py

Builds the final 525-row balanced benchmark dataset from files already on HPC:
  - train.csv        (MedQuAD  — 16k rows)
  - validation.csv   (MedDialog — 25k rows)
  - curated_seed.csv (25 expert rows, already in source_datasets/)

Output:
  benchmark_dataset/source_datasets/benchmark_dataset_500.csv
  Columns: id, question, answer, domain, source, difficulty
  Rows:    525 total (100 per domain x 5 domains + 25 curated)

Usage (from repo root):
  python benchmark_dataset/build_benchmark_500.py \\
    --medquad   /path/to/train.csv \\
    --meddialog /path/to/validation.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DEST_DIR = ROOT / "benchmark_dataset" / "source_datasets"
OUT_FILE = DEST_DIR / "benchmark_dataset_500.csv"

TARGET_PER_DOMAIN = 100

DOMAINS = ["Cardiology", "Pharmacology", "Neurology", "Pediatrics", "Emergency"]

DOMAIN_KEYWORDS = {
    "Cardiology":    ["heart","cardiac","cardiology","hypertension","blood pressure",
                      "chest pain","stemi","acs","ecg","ekg","atrial fibrillation",
                      "arrhythmia","coronary","angina","myocardial infarction",
                      "palpitation","tachycardia","bradycardia","heart failure",
                      "ejection fraction","aortic","mitral","valve","pericarditis",
                      "endocarditis","troponin","cholesterol","lipid","statin",
                      "cardiovascular","cardiomyopathy","ventricular"],
    "Pharmacology":  ["drug","medication","dose","dosing","antibiotic","warfarin",
                      "metformin","prescription","side effect","adverse effect",
                      "drug interaction","pharmacology","opioid","naloxone","steroid",
                      "insulin","aspirin","ibuprofen","acetaminophen","antihistamine",
                      "antidepressant","antifungal","antiviral","chemotherapy",
                      "toxicity","overdose","withdrawal","contraindication"],
    "Neurology":     ["headache","migraine","seizure","epilepsy","stroke","tpa",
                      "neurology","brain","nerve","neuropathy","multiple sclerosis",
                      "parkinson","alzheimer","dementia","tremor","vertigo",
                      "dizziness","syncope","meningitis","encephalitis","spinal cord",
                      "myelopathy","radiculopathy","cognitive","memory loss",
                      "confusion","paralysis","numbness","subarachnoid"],
    "Pediatrics":    ["child","children","pediatric","infant","baby","newborn",
                      "neonate","toddler","adolescent","teenager","puberty",
                      "vaccine","vaccination","immunization","febrile seizure",
                      "otitis media","croup","rsv","bronchiolitis","childhood",
                      "growth chart","developmental","neonatal","congenital",
                      "pediatric asthma","preterm"],
    "Emergency":     ["emergency","urgent","immediately","acute","life-threatening",
                      "anaphylaxis","sepsis","shock","trauma","bleeding","hemorrhage",
                      "cardiac arrest","cpr","resuscitation","intubation",
                      "airway management","poisoning","burn","drowning","choking",
                      "unconscious","unresponsive","not breathing","critical care",
                      "icu","intensive care","triage","overdose emergency",
                      "tension pneumothorax","anaphylactic"],
}


def classify_domain(q: str, a: str) -> str | None:
    text = (str(q) + " " + str(a)).lower()
    scores = {
        d: sum(1 for kw in kws if kw in text)
        for d, kws in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else None


def token_est(text: str) -> int:
    return max(1, len(str(text)) // 4)


def load_medquad(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df[["Question", "Answer"]].copy()
    df.columns = ["question", "answer"]
    df["source"] = "MedQuAD"
    return df


def load_meddialog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df[["patient_message", "doctor_response"]].copy()
    df.columns = ["question", "answer"]
    df["source"] = "MedDialog"
    return df


def build(medquad_path: str, meddialog_path: str) -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("build_benchmark_500.py")
    print("=" * 60)

    # ── Load curated seed ────────────────────────────────────────
    curated_path = DEST_DIR / "curated_seed.csv"
    if not curated_path.exists():
        print(f"[ERROR] curated_seed.csv not found at {curated_path}")
        sys.exit(1)
    curated = pd.read_csv(curated_path)
    print(f"Loaded curated_seed.csv: {len(curated)} rows")

    # ── Load source datasets ─────────────────────────────────────
    medquad   = load_medquad(medquad_path)
    meddialog = load_meddialog(meddialog_path)
    pool = pd.concat([medquad, meddialog], ignore_index=True)

    pool.dropna(subset=["question", "answer"], inplace=True)
    pool["question"] = pool["question"].astype(str).str.strip()
    pool["answer"]   = pool["answer"].astype(str).str.strip()
    pool = pool[
        (pool["question"].str.len() > 20) &
        (pool["answer"].str.len() > 60)
    ]
    pool.drop_duplicates(subset="question", inplace=True)
    pool.reset_index(drop=True, inplace=True)
    print(f"Cleaned pool: {len(pool)} rows")

    # ── Classify domains ─────────────────────────────────────────
    pool["domain"] = pool.apply(
        lambda r: classify_domain(r["question"], r["answer"]), axis=1
    )
    pool["tok"] = pool.apply(
        lambda r: token_est(r["question"]) + token_est(r["answer"]), axis=1
    )
    classified = pool[
        pool["domain"].notna() & (pool["tok"] <= 700)
    ].copy()
    print(f"Classifiable rows: {len(classified)}")
    print("Domain counts:", classified["domain"].value_counts().to_dict())

    # ── Stratified sample: TARGET_PER_DOMAIN per domain ──────────
    sampled_parts = []
    print("\nStratified sampling:")
    for domain in DOMAINS:
        subset = classified[classified["domain"] == domain].copy()
        # Prefer richer answers (sort by token count desc)
        subset = subset.sort_values("tok", ascending=False).head(TARGET_PER_DOMAIN * 3)
        taken  = subset.head(TARGET_PER_DOMAIN)
        taken  = taken[["question", "answer", "source", "domain"]].copy()
        taken["difficulty"] = "medium"
        taken["id"] = [
            f"SRC_{domain[:3].upper()}_{str(i).zfill(4)}"
            for i in range(len(taken))
        ]
        sampled_parts.append(taken)
        print(f"  {domain:<15}: {len(taken):>3} rows  (pool: {len(subset)})")

    source_df = pd.concat(sampled_parts, ignore_index=True)

    # ── Combine curated + source ──────────────────────────────────
    curated_clean = curated[["id", "domain", "difficulty", "question", "answer", "source"]].copy() \
        if "id" in curated.columns else curated
    source_clean  = source_df[["id", "domain", "difficulty", "question", "answer", "source"]].copy()

    final = pd.concat([curated_clean, source_clean], ignore_index=True)
    final.reset_index(drop=True, inplace=True)

    # ── Save ─────────────────────────────────────────────────────
    final.to_csv(OUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("=== Final dataset summary ===")
    print(f"  Total rows  : {len(final)}")
    print("  Per domain  :")
    for d, n in final["domain"].value_counts().sort_index().items():
        print(f"    {d:<15}: {n}")
    print(f"  Output file : {OUT_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build balanced 525-row benchmark dataset"
    )
    parser.add_argument(
        "--medquad",
        required=True,
        help="Path to MedQuAD train.csv (columns: qtype, Question, Answer)",
    )
    parser.add_argument(
        "--meddialog",
        required=True,
        help="Path to MedDialog validation.csv (columns: patient_message, doctor_response)",
    )
    args = parser.parse_args()
    build(args.medquad, args.meddialog)
