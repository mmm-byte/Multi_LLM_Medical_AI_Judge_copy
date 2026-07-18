#!/usr/bin/env python3
"""Build strict real-only datasets from MedQuAD, MedDialog, and Medical Meadow.

Outputs:
  - benchmark_dataset/strict_real_candidates_1500.csv
  - benchmark_dataset/1000_questions_dataset.csv

Design:
  - Uses only real rows from the three requested source families.
  - Builds a 1,500-row candidate pool with 500 rows per family.
  - Filters down to a final 1,000-row dataset.
  - Keeps source-family counts closer than the current skewed build.
  - No synthetic rows and no curated seed rows.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
OUT_CANDIDATES = ROOT / "benchmark_dataset" / "strict_real_candidates_1500.csv"
OUT_FINAL = ROOT / "benchmark_dataset" / "1000_questions_dataset.csv"

MAX_QA_TOKENS = 600
FAMILIES = ("medquad", "meddialog", "medical_meadow")
CANDIDATE_TARGETS = {"medquad": 500, "meddialog": 500, "medical_meadow": 500}
FINAL_TARGETS = {"medquad": 334, "meddialog": 333, "medical_meadow": 333}
DOMAIN_TARGETS = {
    "Cardiology": 200,
    "Pharmacology": 200,
    "Neurology": 200,
    "Pediatrics": 200,
    "Emergency": 200,
}
FAMILY_CANDIDATE_DOMAIN_TARGETS = {
    "medquad": {
        "Cardiology": 140,
        "Pharmacology": 100,
        "Neurology": 140,
        "Pediatrics": 100,
        "Emergency": 20,
    },
}

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
ALL_DOMAINS = list(DOMAIN_KEYWORDS.keys())


def _token_est(text: str) -> int:
    return max(1, len(text) // 4)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", text.lower())


def classify_domain(question: str, answer: str = "") -> str:
    text_lower = f"{question} {answer}".lower()
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
    jargon_terms = {
        "pharmacokinetics", "acetylcholinesterase", "bioavailability",
        "pathophysiology", "hemodynamically", "contraindication",
        "contraindications", "anticholinergic", "thromboembolism",
        "subarachnoid", "xanthochromia", "glomerular", "myocardial",
    }
    jargon_hits = sum(1 for w in words if w in jargon_terms)
    long_words = sum(1 for w in words if len(w) >= 13)
    penalty = min(1.0, (jargon_hits * 0.22) + (long_words * 0.04))
    return round(max(0.0, 1.0 - penalty), 3)


def score_actionable_steps(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    action_verbs = {
        "activate", "administer", "apply", "begin", "call", "check", "consult",
        "continue", "do", "give", "hold", "monitor", "measure", "perform",
        "repeat", "refer", "restart", "seek", "start", "stop", "use",
    }
    verb_hits = sum(1 for w in words if w in action_verbs)
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    score = min(1.0, (verb_hits / 4.0) + min(0.2, sentence_count * 0.04))
    return round(score, 3)


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


def expected_agreement_class(question: str, answer: str) -> str:
    emergency_terms = {
        "anaphylaxis", "sepsis", "unresponsive", "not breathing", "cpr",
        "cardiac arrest", "tension pneumothorax", "shock", "overdose",
        "stroke", "chest pain", "stemi", "suicidal", "severe bleeding",
    }
    escalation_terms = {
        "911", "emergency", "ed", "er", "ambulance", "immediately",
        "urgent", "call for help", "activate", "cath lab", "aed", "cpr",
    }
    scores = {
        "plain": score_plain_language(answer),
        "action": score_actionable_steps(answer),
        "clarity": score_clarity(answer),
    }
    has_emergency = any(term in question.lower() for term in emergency_terms)
    emerg_score = 1.0 if (not has_emergency or any(term in answer.lower() for term in escalation_terms)) else 0.0
    if len(_words(answer)) < 4:
        return "full_disagree"
    if has_emergency and emerg_score == 0.0:
        return "full_disagree" if scores["action"] < 0.35 else "split"
    weighted = 0.20 * scores["plain"] + 0.25 * scores["action"] + 0.30 * emerg_score + 0.25 * scores["clarity"]
    if weighted >= 0.78:
        return "fully_agree"
    if weighted >= 0.60:
        return "majority_agree"
    if weighted >= 0.40:
        return "split"
    return "full_disagree"


def is_relevant_medquad(question: str, answer: str) -> bool:
    q = question.strip().lower()
    a = answer.strip().lower()
    if len(q) < 15 or len(a) < 20:
        return False
    if "more detailed information" in a or "for more information" in a:
        return False
    if a.startswith("to find out more"):
        return False
    return True


def is_relevant_meddialog(question: str, answer: str) -> bool:
    q = question.strip().lower()
    a = answer.strip().lower()
    if len(q) < 20 or len(a) < 20:
        return False
    if a in {"doctor:", "doctor: yes.", "doctor: no."}:
        return False
    return True


def is_relevant_medical_meadow(question: str, answer: str) -> bool:
    q = question.strip().lower()
    a = answer.strip().lower()
    if len(q.split()) < 8:
        return False
    action_cues = [
        "should", "recommend", "advice", "need to", "must", "avoid", "use ",
        "monitor", "consult", "follow up", "administer", "consider", "treat",
        "screen", "vaccin", "refer", "therapy", "management",
    ]
    if not any(cue in q for cue in action_cues):
        return False
    bad_cues = [
        "study", "studies", "research", "meta-analysis", "systematic review",
        "cross-sectional", "prevalence", "association", "findings",
        "investigated", "our findings", "this investigation", "cohort",
        "trial", "citation", "[", "]",
    ]
    if any(cue in q for cue in bad_cues):
        return False
    if q.startswith("further ") or q.startswith("here, we "):
        return False
    if a not in {"this is no advice", "this is a weak advice", "this is a strong advice"}:
        return False
    return True


def _normalize(question: str, answer: str) -> tuple[str, str]:
    return question.strip(), answer.strip()


def _signature(question: str, answer: str) -> str:
    return f"{question.strip().lower()}||{answer.strip().lower()}"


def _fetch_github_csv(owner: str, repo: str, path: str, ref: str = "main") -> List[Dict[str, str]]:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    with urllib.request.urlopen(api_url, timeout=30) as resp:
        meta = json.loads(resp.read())
    if meta.get("download_url"):
        content = urllib.request.urlopen(meta["download_url"], timeout=120).read().decode("utf-8", errors="ignore")
    else:
        content = base64.b64decode(meta["content"]).decode("utf-8", errors="ignore")
    return list(csv.DictReader(io.StringIO(content)))


def load_medquad_rows() -> List[Dict[str, str]]:
    local = ROOT / "benchmark_dataset" / "MedQuAD" / "train.csv"
    rows: List[Dict[str, str]] = []
    local_rows = []
    if local.exists() and local.stat().st_size > 10:
        with open(local, newline="", encoding="utf-8", errors="ignore") as f:
            local_rows = list(csv.DictReader(f))
    source_rows = local_rows or _fetch_github_csv("m22oct2000", "Multi-LLMs-as-Judge", "dataset/MedQuAD/train.csv")
    for idx, row in enumerate(source_rows):
        q = (row.get("Question") or row.get("question") or row.get("input") or "").strip()
        a = (row.get("Answer") or row.get("answer") or row.get("output") or "").strip()
        if not q or not a:
            continue
        if _token_est(q) + _token_est(a) > MAX_QA_TOKENS:
            continue
        if not is_relevant_medquad(q, a):
            continue
        q, a = _normalize(q, a)
        rows.append({
            "id": f"MQ_{idx:05d}",
            "dataset": "MedQuAD",
            "source_family": "medquad",
            "source": "MedQuAD",
            "question": q,
            "reference_answer": a,
            "domain": classify_domain(q, a),
        })
    return rows


def load_meddialog_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    csv_paths = [
        ROOT / "benchmark_dataset" / "MedDialog" / "en-train.csv",
        ROOT / "benchmark_dataset" / "MedDialog" / "en-dev.csv",
        ROOT / "benchmark_dataset" / "MedDialog" / "en-test.csv",
    ]
    for path in csv_paths:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                q = (row.get("description") or row.get("patient") or "").strip()
                a = (row.get("doctor") or "").strip()
                if not q or not a:
                    continue
                if _token_est(q) + _token_est(a) > MAX_QA_TOKENS:
                    continue
                if not is_relevant_meddialog(q, a):
                    continue
                q, a = _normalize(q, a)
                sig = _signature(q, a)
                if sig in seen:
                    continue
                seen.add(sig)
                rows.append({
                    "id": f"MD_{path.stem}_{idx:05d}",
                    "dataset": "MedDialog",
                    "source_family": "meddialog",
                    "source": path.stem,
                    "question": q,
                    "reference_answer": a,
                    "domain": classify_domain(q, a),
                })
    return rows


def load_medical_meadow_rows() -> List[Dict[str, str]]:
    path = ROOT / "benchmark_dataset" / "Medical Meadow" / "medical_meadow_health_advice.csv"
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            instruction = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            q = f"{instruction} {inp}".strip()
            a = (row.get("output") or "").strip()
            if not q or not a:
                continue
            if _token_est(q) + _token_est(a) > MAX_QA_TOKENS:
                continue
            if not is_relevant_medical_meadow(q, a):
                continue
            q, a = _normalize(q, a)
            rows.append({
                "id": f"MM_{idx:06d}",
                "dataset": "Medical Meadow",
                "source_family": "medical_meadow",
                "source": "medical_meadow_health_advice",
                "question": q,
                "reference_answer": a,
                "domain": classify_domain(q, a),
            })
    return rows


def dedupe_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        sig = _signature(row["question"], row["reference_answer"])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def sample_family_candidates(rows: List[Dict[str, str]], family: str, target: int) -> List[Dict[str, str]]:
    by_domain: Dict[str, Deque[Dict[str, str]]] = {}
    for domain in ALL_DOMAINS:
        domain_rows = sorted(
            [r for r in rows if r["domain"] == domain],
            key=lambda r: (
                0 if r["reference_answer"].strip().lower() == "this is a strong advice" else
                1 if r["reference_answer"].strip().lower() == "this is a weak advice" else
                2 if r["reference_answer"].strip().lower() == "this is no advice" else 0,
                _token_est(r["question"]) + _token_est(r["reference_answer"]),
                r["id"],
            ),
        )
        by_domain[domain] = deque(domain_rows)

    selected: List[Dict[str, str]] = []
    selected_ids = set()

    # Preserve rare agreement-class cases before filling bulk quotas.
    rare_priority = [
        (domain, cls)
        for domain in ALL_DOMAINS
        for cls in ("full_disagree", "split", "majority_agree", "fully_agree")
    ]
    for domain, cls in rare_priority:
        domain_rows = list(by_domain[domain])
        candidates = [
            row for row in domain_rows
            if expected_agreement_class(row["question"], row["reference_answer"]) == cls
            and row["id"] not in selected_ids
        ]
        if not candidates:
            continue
        row = candidates[0]
        selected.append(row)
        selected_ids.add(row["id"])
        by_domain[domain] = deque([r for r in by_domain[domain] if r["id"] != row["id"]])

    requested = FAMILY_CANDIDATE_DOMAIN_TARGETS.get(family)
    if requested:
        for domain in ALL_DOMAINS:
            need = requested.get(domain, 0)
            already = sum(1 for r in selected if r["domain"] == domain)
            need = max(0, need - already)
            if need > len(by_domain[domain]):
                raise RuntimeError(
                    f"{family} candidate target for {domain} exceeds available rows: need {need}, have {len(by_domain[domain])}"
                )
            for _ in range(need):
                selected.append(by_domain[domain].popleft())
    else:
        per_domain_seed = target // len(ALL_DOMAINS)
        for domain in ALL_DOMAINS:
            already = sum(1 for r in selected if r["domain"] == domain)
            need = max(0, per_domain_seed - already)
            for _ in range(min(need, len(by_domain[domain]))):
                selected.append(by_domain[domain].popleft())

    leftovers = []
    for domain in ALL_DOMAINS:
        leftovers.extend(list(by_domain[domain]))
    leftovers.sort(key=lambda r: (
        ALL_DOMAINS.index(r["domain"]),
        expected_agreement_class(r["question"], r["reference_answer"]),
        _token_est(r["question"]) + _token_est(r["reference_answer"]),
        r["id"],
    ))
    for row in leftovers:
        if len(selected) >= target:
            break
        selected.append(row)
    if len(selected) < target:
        raise RuntimeError(f"{family} has only {len(selected)} usable rows; target was {target}")
    return selected[:target]


def build_candidate_pool() -> List[Dict[str, str]]:
    families = {
        "medquad": dedupe_rows(load_medquad_rows()),
        "meddialog": dedupe_rows(load_meddialog_rows()),
        "medical_meadow": dedupe_rows(load_medical_meadow_rows()),
    }
    candidate_rows: List[Dict[str, str]] = []
    for family, rows in families.items():
        candidate_rows.extend(sample_family_candidates(rows, family, CANDIDATE_TARGETS[family]))
    return candidate_rows


def plan_family_domain_quotas(candidate_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    available: Dict[str, Dict[str, int]] = {f: {d: 0 for d in ALL_DOMAINS} for f in FAMILIES}
    for row in candidate_rows:
        available[row["source_family"]][row["domain"]] += 1

    quotas: Dict[str, Dict[str, int]] = {f: {d: 0 for d in ALL_DOMAINS} for f in FAMILIES}
    remaining_domain = DOMAIN_TARGETS.copy()
    family_order = ["meddialog", "medquad", "medical_meadow"]

    for idx, family in enumerate(family_order):
        family_target = FINAL_TARGETS[family]
        if idx == len(family_order) - 1:
            for domain in ALL_DOMAINS:
                need = remaining_domain[domain]
                if need > available[family][domain]:
                    raise RuntimeError(
                        f"Insufficient {family} rows for {domain}: need {need}, have {available[family][domain]}"
                    )
                quotas[family][domain] = need
            continue

        later_families = family_order[idx + 1 :]
        lower = {}
        upper = {}
        for domain in ALL_DOMAINS:
            later_capacity = sum(available[later][domain] for later in later_families)
            lower[domain] = max(0, remaining_domain[domain] - later_capacity)
            upper[domain] = min(available[family][domain], remaining_domain[domain])
            quotas[family][domain] = lower[domain]

        allocated = sum(quotas[family].values())
        if allocated > family_target:
            raise RuntimeError(f"Infeasible lower bounds for {family}: {allocated} > {family_target}")

        remaining = family_target - allocated
        # Seed with an even split where possible, then use remaining slack.
        desired_base = family_target // len(ALL_DOMAINS)
        for domain in ALL_DOMAINS:
            if remaining <= 0:
                break
            want = max(0, min(desired_base, upper[domain]) - quotas[family][domain])
            take = min(remaining, want)
            quotas[family][domain] += take
            remaining -= take

        while remaining > 0:
            choices = [
                domain for domain in ALL_DOMAINS
                if quotas[family][domain] < upper[domain]
            ]
            if not choices:
                raise RuntimeError(f"Cannot satisfy target for {family}; remaining={remaining}")
            domain = max(
                choices,
                key=lambda d: (upper[d] - quotas[family][d], remaining_domain[d], -ALL_DOMAINS.index(d)),
            )
            quotas[family][domain] += 1
            remaining -= 1

        for domain in ALL_DOMAINS:
            remaining_domain[domain] -= quotas[family][domain]

    return quotas


def select_final_rows(candidate_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    pools: Dict[str, Dict[str, Deque[Dict[str, str]]]] = {f: {d: deque() for d in ALL_DOMAINS} for f in FAMILIES}
    for row in sorted(candidate_rows, key=lambda r: (
        r["source_family"],
        ALL_DOMAINS.index(r["domain"]),
        expected_agreement_class(r["question"], r["reference_answer"]),
        _token_est(r["question"]) + _token_est(r["reference_answer"]),
        r["id"],
        )):
        pools[row["source_family"]][row["domain"]].append(row)
    quotas = plan_family_domain_quotas(candidate_rows)
    picked: List[Dict[str, str]] = []

    # Reserve at least one row per domain/agreement class when available.
    reserved = set()
    for domain in ALL_DOMAINS:
        for cls in ("fully_agree", "majority_agree", "split", "full_disagree"):
            choices = []
            for family in FAMILIES:
                if quotas[family][domain] <= 0:
                    continue
                for row in pools[family][domain]:
                    if expected_agreement_class(row["question"], row["reference_answer"]) == cls:
                        choices.append((family, row))
                        break
            if not choices:
                continue
            family, row = choices[0]
            # Remove the chosen row from its pool while preserving order.
            kept = deque()
            found = False
            while pools[family][domain]:
                cur = pools[family][domain].popleft()
                if not found and cur["id"] == row["id"]:
                    found = True
                    continue
                kept.append(cur)
            pools[family][domain] = kept
            picked.append(row)
            quotas[family][domain] -= 1
            reserved.add(row["id"])

    for family in FAMILIES:
        for domain in ALL_DOMAINS:
            need = quotas[family][domain]
            if need > len(pools[family][domain]):
                raise RuntimeError(
                    f"Quota exceeds pool for {family}/{domain}: need {need}, have {len(pools[family][domain])}"
                )
            for _ in range(need):
                picked.append(pools[family][domain].popleft())

    if len(picked) != 1000:
        raise RuntimeError(f"Failed to build 1000-row final dataset. Picked={len(picked)}")
    return picked


def decorate_row(row: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)
    out["expected_class"] = expected_agreement_class(row["question"], row["reference_answer"])
    out["qa_token_est"] = str(_token_est(row["question"]) + _token_est(row["reference_answer"]))
    out["answer"] = out.pop("reference_answer")
    out["source_dataset"] = out.pop("source")
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "id", "dataset", "source_family", "source_dataset", "domain",
        "question", "answer", "expected_class", "qa_token_est",
    ]
    output_rows = []
    for idx, row in enumerate(rows, start=1):
        out = decorate_row(row)
        out["id"] = str(idx)
        output_rows.append(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    candidates = build_candidate_pool()
    final_rows = select_final_rows(candidates)
    write_csv(OUT_CANDIDATES, candidates)
    write_csv(OUT_FINAL, final_rows)

    print(f"Candidates written: {OUT_CANDIDATES}")
    print(f"Final dataset written: {OUT_FINAL}")
    print("Candidate family counts:", Counter(r["source_family"] for r in candidates))
    print("Final family counts:", Counter(r["source_family"] for r in final_rows))
    print("Final domain counts:", Counter(r["domain"] for r in final_rows))


if __name__ == "__main__":
    main()
