"""Agreement classification logic for the clinical QA multi-judge panel.

For each question evaluated by the panel under a given rubric, classifies
the panel's agreement as one of:
  - fully_agree     : all judges agree (pairwise agreement >= threshold)
  - majority_agree  : 3 of 4 judges agree; 1 is an outlier
  - split           : 2 agree, 2 disagree
  - full_disagree   : all judges diverge

The outlier judge (if any) is identified and its rationale is flagged for
special attention in the paper output.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

AGREEMENT_THRESHOLD = 80.0  # pairwise agreement >= this is considered "agree"


def classify_panel_agreement(
    pairwise_scores: Dict[Tuple[str, str], float],
    judge_ids: List[str],
    threshold: float = AGREEMENT_THRESHOLD,
) -> Tuple[str, Optional[str]]:
    """
    Args:
        pairwise_scores: dict mapping (judge_a, judge_b) -> agreement_score (0-100).
                         May contain both (a,b) and (b,a) — only canonical pairs
                         (lower index first) are counted to avoid double-counting.
        judge_ids: list of judge identifiers
        threshold: minimum agreement score to count as "agree"

    Returns:
        (agreement_class, outlier_judge_id)
        agreement_class: 'fully_agree' | 'majority_agree' | 'split' | 'full_disagree'
        outlier_judge_id: judge ID of the outlier (None if fully_agree or full_disagree)
    """
    n = len(judge_ids)
    if n < 2:
        return "fully_agree", None

    idx = {j: i for i, j in enumerate(judge_ids)}

    # Count agreements using only canonical (lower-index-first) pairs
    # to avoid double-counting when pairwise_scores has both (a,b) and (b,a).
    agree_counts: Dict[str, int] = {j: 0 for j in judge_ids}
    agreeing_pairs = []
    for (ja, jb), score in pairwise_scores.items():
        # Skip reverse duplicates
        if ja not in idx or jb not in idx:
            continue
        if idx[ja] > idx[jb]:
            continue
        if score >= threshold:
            agree_counts[ja] += 1
            agree_counts[jb] += 1
            agreeing_pairs.append((ja, jb))

    max_possible = n - 1  # max other judges any one judge can agree with

    # All agree with everyone
    if all(c == max_possible for c in agree_counts.values()):
        return "fully_agree", None

    if all(c == 0 for c in agree_counts.values()):
        return "full_disagree", None

    if n == 4 and len(agreeing_pairs) == 2:
        covered = {j for pair in agreeing_pairs for j in pair}
        if len(covered) == 4:
            return "split", None

    # Find the outlier: the judge with lowest agreement count
    min_agrees = min(agree_counts.values())
    outliers = [j for j, c in agree_counts.items() if c == min_agrees]

    if len(outliers) == 1 and min_agrees == 0:
        # One judge disagrees with all others
        return "majority_agree", outliers[0]

    if len(outliers) == 2 and n == 4:
        # Two groups of two
        return "split", None

    # Fallback
    return "majority_agree", outliers[0]


def build_pairwise_matrix(
    judge_ids: List[str],
    score_getter,  # callable(judge_a, judge_b) -> float
) -> Dict[Tuple[str, str], float]:
    """Build full pairwise agreement matrix for a panel."""
    matrix = {}
    for ja, jb in combinations(judge_ids, 2):
        score = score_getter(ja, jb)
        matrix[(ja, jb)] = score
        matrix[(jb, ja)] = score
    return matrix


def summarize_agreement(
    pairwise_scores: Dict[Tuple[str, str], float],
    judge_ids: List[str],
    threshold: float = AGREEMENT_THRESHOLD,
) -> Dict:
    """Return a full summary dict suitable for JSON results output."""
    agreement_class, outlier = classify_panel_agreement(pairwise_scores, judge_ids, threshold)
    pairs = [
        {"judge_a": ja, "judge_b": jb, "score": round(score, 2)}
        for (ja, jb), score in pairwise_scores.items()
        if judge_ids.index(ja) < judge_ids.index(jb)
    ]
    return {
        "agreement_class": agreement_class,
        "outlier_judge": outlier,
        "pairwise_scores": pairs,
        "mean_pairwise_agreement": round(
            sum(p["score"] for p in pairs) / len(pairs) if pairs else 0.0, 2
        ),
    }
