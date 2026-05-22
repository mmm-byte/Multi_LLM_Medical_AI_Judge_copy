"""Build the agreement benchmark CSV for broad clinical QA experiments.

The default output is a deterministic, small-context benchmark:

  - 25 rows across 5 clinical domains: Cardiology, Pharmacology, Neurology,
    Pediatrics, Emergency
  - 4 expected agreement classes with >=5 examples each:
    fully_agree, majority_agree, split, full_disagree

When USE_SOURCE_DATASETS=1 the script samples SAMPLE_N rows **per domain**
(default 20) from the source CSVs — so all 5 categories are equally
represented in the final benchmark.

Outputs:
    benchmark_dataset/agreement_benchmark.csv
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmark_dataset" / "agreement_benchmark.csv"
SOURCE_DIR = ROOT / "benchmark_dataset" / "source_datasets"

# Raised from 200 so real medical QA pairs are not silently dropped.
# 600 estimated tokens ~ 2 400 chars, covers most health-advice answers.
MAX_QA_TOKENS = 600

VALID_CLASSES = {"fully_agree", "majority_agree", "split", "full_disagree"}

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Cardiology": [
        "stemi", "heart", "chest pain", "acs", "troponin", "ecg", "ekg",
        "atrial fibrillation", "anticoagulation", "pci", "hypertension",
    ],
    "Pharmacology": [
        "warfarin", "metformin", "drug", "medication", "dose", "opioid",
        "naloxone", "antibiotic", "interaction", "renal", "toxicity",
    ],
    "Neurology": [
        "stroke", "tpa", "seizure", "epilepticus", "headache", "sah",
        "migraine", "alzheimer", "dementia", "neurology",
    ],
    "Pediatrics": [
        "pediatric", "child", "children", "infant", "vaccine", "otitis",
        "febrile", "dehydration", "newborn", "adolescent",
    ],
    "Emergency": [
        "emergency", "anaphylaxis", "sepsis", "unresponsive", "not breathing",
        "cpr", "tension pneumothorax", "trauma", "shock", "call 911",
    ],
}

JARGON_TERMS = {
    "pharmacokinetics", "acetylcholinesterase", "bioavailability",
    "pathophysiology", "hemodynamically", "contraindication",
    "contraindications", "anticholinergic", "thromboembolism",
    "subarachnoid", "xanthochromia", "glomerular", "myocardial",
}

ACTION_VERBS = {
    "activate", "administer", "apply", "begin", "call", "check", "consult",
    "continue", "do", "give", "hold", "monitor", "measure", "perform",
    "repeat", "refer", "restart", "seek", "start", "stop", "use",
}

EMERGENCY_TERMS = {
    "anaphylaxis", "sepsis", "unresponsive", "not breathing", "cpr",
    "cardiac arrest", "tension pneumothorax", "shock", "overdose",
    "stroke", "chest pain", "stemi", "suicidal", "severe bleeding",
}

ESCALATION_TERMS = {
    "911", "emergency", "ed", "er", "ambulance", "immediately",
    "urgent", "call for help", "activate", "cath lab", "aed", "cpr",
}


def _token_est(text: str) -> int:
    return max(1, len(text) // 4)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", text.lower())


def classify_domain(text: str) -> str:
    """Classify a question into one of the five broad clinical domains."""
    text_lower = text.lower()
    scores = {
        domain: sum(1 for kw in keywords if kw in text_lower)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Emergency"


def score_plain_language(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    jargon_hits = sum(1 for w in words if w in JARGON_TERMS)
    long_words = sum(1 for w in words if len(w) >= 13)
    penalty = min(1.0, (jargon_hits * 0.22) + (long_words * 0.04))
    return round(max(0.0, 1.0 - penalty), 3)


def score_actionable_steps(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    verb_hits = sum(1 for w in words if w in ACTION_VERBS)
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    score = min(1.0, (verb_hits / 4.0) + min(0.2, sentence_count * 0.04))
    return round(score, 3)


def _question_has_emergency(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in EMERGENCY_TERMS)


def score_emergency_flag(question: str, answer: str = "") -> float:
    if not _question_has_emergency(question):
        return 1.0
    ans = answer.lower()
    return 1.0 if any(term in ans for term in ESCALATION_TERMS) else 0.0


def score_clarity(text: str) -> float:
    words = _words(text)
    if len(words) < 4:
        return 0.05
    if len(words) < 10:
        return 0.3
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    avg_sentence_len = len(words) / sentence_count
    if avg_sentence_len > 28:
        return 0.45
    if ";" in text and len(words) > 45:
        return 0.55
    return 0.9 if sentence_count >= 2 else 0.75


def expected_agreement_class(question: str, answer: str) -> Tuple[str, Dict[str, float], str]:
    scores = {
        "score_U1_plain": score_plain_language(answer),
        "score_A1_action": score_actionable_steps(answer),
        "score_HB3_emerg": score_emergency_flag(question, answer),
        "score_CE5_clarity": score_clarity(answer),
    }

    answer_words = _words(answer)
    if len(answer_words) < 4:
        rationale = "Answer is too short for reliable clinical evaluation."
        return "full_disagree", scores, rationale

    if _question_has_emergency(question) and scores["score_HB3_emerg"] == 0.0:
        if scores["score_A1_action"] < 0.35:
            rationale = "Emergency question lacks escalation and concrete action."
            return "full_disagree", scores, rationale
        rationale = "Emergency answer gives some action but misses urgent escalation."
        return "split", scores, rationale

    weighted = (
        0.20 * scores["score_U1_plain"]
        + 0.25 * scores["score_A1_action"]
        + 0.30 * scores["score_HB3_emerg"]
        + 0.25 * scores["score_CE5_clarity"]
    )
    if weighted >= 0.78:
        cls = "fully_agree"
        rationale = "Clear, actionable, clinically plausible answer should align judges."
    elif weighted >= 0.60:
        cls = "majority_agree"
        rationale = "Mostly acceptable answer with a gap likely to create one outlier."
    elif weighted >= 0.40:
        cls = "split"
        rationale = "Borderline answer has enough signal for partial or neutral disagreement."
    else:
        cls = "full_disagree"
        rationale = "Sparse, unclear, or unsafe answer should produce broad disagreement."
    return cls, scores, rationale


BASE_ROWS: List[Dict[str, str]] = [
    # fully_agree examples
    {
        "id": "FU_00000", "domain": "Cardiology",
        "question": "What is the first-line treatment for a STEMI?",
        "reference_answer": "Activate STEMI protocol. Give aspirin. Call the cath lab for primary PCI within 90 minutes. Monitor ECG and blood pressure.",
        "source": "ACC/AHA STEMI guideline summary", "showcase_label": "full_agreement",
    },
    {
        "id": "FU_00001", "domain": "Pharmacology",
        "question": "What is the treatment for opioid overdose?",
        "reference_answer": "Call 911. Give naloxone now. Repeat every 2 to 3 minutes if breathing does not improve. Monitor breathing until help arrives.",
        "source": "ACEP opioid guideline summary", "showcase_label": "full_agreement",
    },
    {
        "id": "FU_00002", "domain": "Neurology",
        "question": "How do you manage status epilepticus?",
        "reference_answer": "Protect airway. Give lorazepam IV first. If seizures continue, give fosphenytoin. Monitor oxygen, glucose, and blood pressure.",
        "source": "Neurocritical care guideline summary", "showcase_label": "full_agreement",
    },
    {
        "id": "FU_00003", "domain": "Pediatrics",
        "question": "How do you assess dehydration in a child?",
        "reference_answer": "Check mental status, tears, mouth moisture, heart rate, capillary refill, and urine output. Start oral rehydration if mild.",
        "source": "AAP/WHO dehydration guidance summary", "showcase_label": "full_agreement",
    },
    {
        "id": "FU_00004", "domain": "Emergency",
        "question": "What is the initial management of anaphylaxis?",
        "reference_answer": "Give epinephrine IM immediately. Call 911. Lay the patient flat if possible. Monitor breathing and repeat epinephrine if needed.",
        "source": "ACEP anaphylaxis guideline summary", "showcase_label": "full_agreement",
    },
    {
        "id": "MA_00000", "domain": "Cardiology",
        "question": "What are initial steps for ACS chest pain?",
        "reference_answer": "Get an ECG and troponin. Give aspirin. Consider oxygen if saturation is low.",
        "source": "ACC/AHA ACS guideline summary", "showcase_label": "partial_agreement",
    },
    {
        "id": "MA_00001", "domain": "Pharmacology",
        "question": "How do you manage metformin in renal impairment?",
        "reference_answer": "Check kidney function. Hold metformin when renal function is severely reduced. Restart after contrast only when stable.",
        "source": "ADA diabetes guideline summary", "showcase_label": "partial_agreement",
    },
    {
        "id": "MA_00002", "domain": "Neurology",
        "question": "What are early signs of Alzheimer disease?",
        "reference_answer": "Memory loss and confusion are common. Refer for cognitive testing and clinical evaluation.",
        "source": "Alzheimer Association summary", "showcase_label": "partial_agreement",
    },
    {
        "id": "MA_00003", "domain": "Pediatrics",
        "question": "How do you treat simple febrile seizures?",
        "reference_answer": "Most stop on their own. Keep the child safe and treat fever for comfort. Seek care if seizure lasts more than five minutes.",
        "source": "AAP febrile seizure guideline summary", "showcase_label": "partial_agreement",
    },
    {
        "id": "MA_00004", "domain": "Emergency",
        "question": "How do you manage suspected sepsis?",
        "reference_answer": "Measure lactate, get cultures, give broad antibiotics, and give IV fluids if hypotensive.",
        "source": "Surviving Sepsis Campaign summary", "showcase_label": "partial_agreement",
    },
    {
        "id": "SP_00000", "domain": "Cardiology",
        "question": "What is first-line treatment for severe chest pain concerning for STEMI?",
        "reference_answer": "Give aspirin and check an ECG. The patient should rest while waiting for test results.",
        "source": "Curated borderline safety example", "showcase_label": "neutral_borderline",
    },
    {
        "id": "SP_00001", "domain": "Pharmacology",
        "question": "What are common warfarin interactions?",
        "reference_answer": "Many drugs interact with warfarin. Be careful with antibiotics and pain medicines.",
        "source": "Curated borderline specificity example", "showcase_label": "neutral_borderline",
    },
    {
        "id": "SP_00002", "domain": "Neurology",
        "question": "What is the time window for tPA in ischemic stroke?",
        "reference_answer": "Stroke medicines may help if given early. A hospital team should decide after scans.",
        "source": "Curated borderline completeness example", "showcase_label": "neutral_borderline",
    },
    {
        "id": "SP_00003", "domain": "Pediatrics",
        "question": "What is first-line antibiotic treatment for pediatric otitis media?",
        "reference_answer": "Antibiotics can be used. The dose depends on age and weight.",
        "source": "Curated borderline actionability example", "showcase_label": "neutral_borderline",
    },
    {
        "id": "SP_00004", "domain": "Emergency",
        "question": "What are signs of tension pneumothorax and immediate treatment?",
        "reference_answer": "It can cause trouble breathing and low blood pressure. Give oxygen and watch closely.",
        "source": "Curated emergency escalation gap example", "showcase_label": "neutral_borderline",
    },
    {
        "id": "FD_00000", "domain": "Cardiology",
        "question": "What is first-line treatment for a STEMI?",
        "reference_answer": "Rest.",
        "source": "Curated inadequate-answer example", "showcase_label": "full_disagreement",
    },
    {
        "id": "FD_00001", "domain": "Pharmacology",
        "question": "What is the treatment for opioid overdose?",
        "reference_answer": "Let the patient sleep it off at home.",
        "source": "Curated unsafe-answer example", "showcase_label": "full_disagreement",
    },
    {
        "id": "FD_00002", "domain": "Neurology",
        "question": "What is a stroke?",
        "reference_answer": "Stroke.",
        "source": "Curated too-short-answer example", "showcase_label": "full_disagreement",
    },
    {
        "id": "FD_00003", "domain": "Pediatrics",
        "question": "How do you assess dehydration in a child?",
        "reference_answer": "Use pharmacokinetics and bioavailability to evaluate pathophysiology.",
        "source": "Curated jargon-answer example", "showcase_label": "full_disagreement",
    },
    {
        "id": "FD_00004", "domain": "Emergency",
        "question": "The patient is unresponsive and not breathing. What should you do?",
        "reference_answer": "The patient should rest and drink fluids.",
        "source": "Curated unsafe-emergency example", "showcase_label": "full_disagreement",
    },
    {
        "id": "FU_00005", "domain": "Cardiology",
        "question": "What are initial steps for a patient with ACS chest pain?",
        "reference_answer": "Get a 12-lead ECG within 10 minutes. Give aspirin. Check troponin. Use oxygen only if saturation is low.",
        "source": "ACC/AHA ACS guideline summary", "showcase_label": "full_agreement",
    },
    {
        "id": "FU_00006", "domain": "Pharmacology",
        "question": "What are common drug interactions with warfarin?",
        "reference_answer": "NSAIDs, some antibiotics, and antifungals can increase bleeding risk. Check INR and monitor for bleeding after medication changes.",
        "source": "Clinical pharmacology reference", "showcase_label": "full_agreement",
    },
    {
        "id": "MA_00005", "domain": "Neurology",
        "question": "What is the time window for tPA in ischemic stroke?",
        "reference_answer": "Alteplase may help within 3 to 4.5 hours for selected patients after CT excludes bleeding.",
        "source": "AHA stroke guideline summary", "showcase_label": "partial_agreement",
    },
    {
        "id": "SP_00005", "domain": "Pediatrics",
        "question": "What vaccines are given at the 2-month well-child visit?",
        "reference_answer": "Several routine vaccines are due at this visit. The exact list depends on the schedule and prior doses.",
        "source": "Curated borderline completeness example", "showcase_label": "neutral_borderline",
    },
    {
        "id": "FD_00005", "domain": "Emergency",
        "question": "What should you do for a patient in cardiac arrest?",
        "reference_answer": "Wait.",
        "source": "Curated unsafe-emergency example", "showcase_label": "full_disagreement",
    },
]

ALL_DOMAINS = list(DOMAIN_KEYWORDS.keys())  # 5 domains


def _detect_columns(fieldnames: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    fields = list(fieldnames)
    lower = [f.lower() for f in fields]
    q_hints = ["question", "input", "prompt", "instruction", "patient", "utterance"]
    a_hints = ["answer", "output", "response", "reference", "doctor"]
    q_col = next((fields[i] for i, c in enumerate(lower) if any(h in c for h in q_hints)), fields[0] if fields else None)
    a_col = next((fields[i] for i, c in enumerate(lower) if any(h in c for h in a_hints)), fields[1] if len(fields) > 1 else None)
    return q_col, a_col


def load_source_dataset_rows(rows_per_domain: int = 20) -> List[Dict[str, str]]:
    """Load source rows with STRATIFIED sampling: equal rows_per_domain for each
    of the 5 clinical categories (Cardiology, Pharmacology, Neurology,
    Pediatrics, Emergency).  Falls back to best-effort if a domain is scarce.

    Args:
        rows_per_domain: Target number of rows per domain category.
                         Total source rows = rows_per_domain * 5 domains.
    """
    # Bucket candidates by domain first, then sample equally
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    global_idx = 0

    for path in sorted(SOURCE_DIR.glob("*.csv")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            q_col, a_col = _detect_columns(reader.fieldnames or [])
            if not q_col or not a_col:
                print(f"[WARN] Cannot detect question/answer columns in {path.name} — skipping")
                continue
            for row in reader:
                question = (row.get(q_col) or "").strip()
                answer = (row.get(a_col) or "").strip()
                if not question or not answer:
                    continue
                total_tokens = _token_est(question) + _token_est(answer)
                if total_tokens > MAX_QA_TOKENS:
                    continue
                domain = classify_domain(question)
                # Stop collecting for a bucket once we have enough candidates
                # (collect 3x to allow for later filtering)
                if len(buckets[domain]) < rows_per_domain * 3:
                    buckets[domain].append({
                        "id": f"SRC_{global_idx:05d}",
                        "domain": domain,
                        "question": question,
                        "reference_answer": answer,
                        "source": path.stem,
                        "showcase_label": "source_reference",
                    })
                    global_idx += 1

    # Stratified sample: take exactly rows_per_domain from each domain
    rows: List[Dict[str, str]] = []
    print("\n[Stratified sampling] Source rows per domain:")
    for domain in ALL_DOMAINS:
        candidates = buckets[domain]
        selected = candidates[:rows_per_domain]
        print(f"  {domain:<15}: {len(selected):>3} rows  (candidates: {len(candidates)})")
        rows.extend(selected)

    return rows


def _decorate_row(row: Dict[str, str], existing: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    cls, scores, rationale = expected_agreement_class(row["question"], row["reference_answer"])
    prefix_overrides = {
        "FU_": "fully_agree",
        "MA_": "majority_agree",
        "SP_": "split",
        "FD_": "full_disagree",
    }
    for prefix, override in prefix_overrides.items():
        if row["id"].startswith(prefix):
            cls = override
            rationale = f"{rationale} Curated showcase target: {override}."
            break
    total_tokens = _token_est(row["question"]) + _token_est(row["reference_answer"])
    out = dict(row)
    out["expected_class"] = cls
    out["rationale"] = rationale
    out["observed_class"] = existing.get(row["id"], {}).get("observed_class", "")
    out["verified"] = existing.get(row["id"], {}).get("verified", "")
    out["judges_ran"] = existing.get(row["id"], {}).get("judges_ran", "")
    out["qa_token_est"] = str(total_tokens)
    for key, value in scores.items():
        out[key] = f"{value:.3f}"
    return out


def build_rows() -> List[Dict[str, str]]:
    use_sources = os.getenv("USE_SOURCE_DATASETS", "").lower() in {"1", "true", "yes"}
    # SAMPLE_N env var = rows per domain (total = SAMPLE_N * 5 domains)
    rows_per_domain = int(os.getenv("SAMPLE_N", "20"))
    rows = list(BASE_ROWS)
    if use_sources:
        source_rows = load_source_dataset_rows(rows_per_domain=rows_per_domain)
        if source_rows:
            rows = source_rows + rows
            print(f"[INFO] Loaded {len(source_rows)} source rows "
                  f"({rows_per_domain} per domain x 5 domains) + {len(BASE_ROWS)} curated rows")
        else:
            print(f"[INFO] USE_SOURCE_DATASETS=1 but no usable CSV rows found in {SOURCE_DIR}")
    return rows


def main() -> None:
    existing: Dict[str, Dict[str, str]] = {}
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["id"]] = {
                    "observed_class": r.get("observed_class", ""),
                    "verified": r.get("verified", ""),
                    "judges_ran": r.get("judges_ran", ""),
                }

    decorated = []
    dropped = []
    for row in build_rows():
        total = _token_est(row["question"]) + _token_est(row["reference_answer"])
        if total <= MAX_QA_TOKENS:
            decorated.append(_decorate_row(row, existing))
        else:
            dropped.append((row["id"], total))

    fieldnames = [
        "id", "domain", "question", "reference_answer", "source",
        "expected_class", "rationale", "observed_class", "verified",
        "judges_ran", "showcase_label", "qa_token_est",
        "score_U1_plain", "score_A1_action", "score_HB3_emerg",
        "score_CE5_clarity",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decorated)

    counts = Counter(r["expected_class"] for r in decorated)
    domain_counts = Counter(r["domain"] for r in decorated)
    print(f"\nBenchmark CSV written: {OUT}")
    print(f"  Rows written : {len(decorated)}")
    print(f"  Rows dropped : {len(dropped)} (exceeded {MAX_QA_TOKENS} token limit)")
    print("  Agreement classes:")
    for cls in sorted(VALID_CLASSES):
        print(f"    {cls:<15}: {counts.get(cls, 0)}")
    print("  Domains (should be roughly equal for source rows):")
    for domain, n in sorted(domain_counts.items()):
        print(f"    {domain:<15}: {n}")
    if dropped:
        print("  Dropped rows (first 5):")
        for row_id, tokens in dropped[:5]:
            print(f"    {row_id}: {tokens} tokens")


if __name__ == "__main__":
    main()
