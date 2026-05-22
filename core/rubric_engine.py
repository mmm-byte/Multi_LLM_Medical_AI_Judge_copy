"""DynamicRubricParser: aggregates judge scores and computes pairwise agreement.

Supports:
  BINARY:      scores 0 or 1; aggregation = (count of 1s) / non-NA * 100
  LIKERT 1-5:  weighted average; normalisation range (1, 5)
  LIKERT 0-10: detected from item description suffix 'Score 0-10';
               normalisation range (0, 10)

Scale-range detection order:
  1. item.description ends with 'Score X-Y.' where X,Y are ints
  2. item.scale == 'BINARY'
  3. Default LIKERT range (1, 5)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.consensus_core.models import Rubric, RubricItem, JudgeScore


def _detect_scale_range(item: RubricItem) -> Tuple[float, float]:
    """Return (lo, hi) normalisation range for this item."""
    import re
    if (item.scale or '').upper() == 'BINARY':
        return 0.0, 1.0
    # Check description suffix: 'Score 0-10.' or 'Score 1-5.'
    desc = (item.description or '').strip()
    m = re.search(r'Score\s+(\d+)-(\d+)\.?\s*$', desc, re.IGNORECASE)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi > lo:
            return lo, hi
    return 1.0, 5.0   # default LIKERT


class DynamicRubricParser:
    """Parses a published rubric and computes pairwise agreement.

    Supports BINARY (0/1/NA) and LIKERT (any integer range).
    Each rubric is taken whole from a single published paper.
    """

    def __init__(self, rubric: Rubric) -> None:
        self.rubric     = rubric
        self.paradigm   = self._detect_paradigm(rubric)
        self.item_by_id = {it.id: it for it in rubric.items}

    @staticmethod
    def _detect_paradigm(rubric: Rubric) -> str:
        if rubric.items and all((it.scale or '').upper() == 'BINARY'
                                for it in rubric.items):
            return 'BINARY'
        return 'LIKERT'

    def generate_judge_instructions(self, rubric: Optional[Rubric] = None) -> str:
        rb       = rubric or self.rubric
        paradigm = self._detect_paradigm(rb)
        header   = [
            'You are a strict medical domain judge evaluating a clinical QA answer.',
            'Score ONLY based on the rubric items below.',
            f'This rubric is taken whole from: {rb.name}',
        ]
        if paradigm == 'BINARY':
            header.append('Score each item: 1=Present/Meets, 0=Absent/Does not meet, NA=Not Applicable.')
        else:
            lo, hi = _detect_scale_range(rb.items[0]) if rb.items else (1.0, 5.0)
            header.append(f'Score each item: integer from {int(lo)} (poor) to {int(hi)} (excellent).')
        header.append('For EACH item provide a one-line rationale.')

        lines = ['\n'.join(header), '\nRubric Items:']
        for idx, it in enumerate(rb.items, start=1):
            lo, hi   = _detect_scale_range(it)
            scale_str = '1/0/NA' if (it.scale or '').upper() == 'BINARY' else f'{int(lo)}-{int(hi)}'
            lines.append(f'{idx}. [{it.id}] {it.name} (scale: {scale_str}, weight: {it.weight})')
            lines.append(f'   {it.description}')
        return '\n'.join(lines)

    def aggregate_score(self, scores: List[JudgeScore]) -> float:
        """BINARY: percent of 1s over non-NA. LIKERT: weighted average."""
        if self.paradigm == 'BINARY':
            present = total = 0
            for sc in scores:
                if sc.rubric_item_id not in self.item_by_id:
                    continue
                raw = str(sc.score).strip().upper()
                if raw in ('', 'NA', 'N/A', 'NONE', 'NULL'):
                    continue
                try:
                    v = int(float(raw))
                except (ValueError, TypeError):
                    continue
                total += 1
                if v == 1:
                    present += 1
            return 0.0 if total == 0 else (present / total) * 100.0

        # LIKERT
        num = den = 0.0
        for sc in scores:
            it = self.item_by_id.get(sc.rubric_item_id)
            if not it:
                continue
            raw = str(sc.score).strip().upper()
            if raw in ('', 'NA', 'N/A', 'NONE', 'NULL'):
                continue
            try:
                val = float(raw)
            except (ValueError, TypeError):
                continue
            w    = float(it.weight or 1.0)
            num += w * val
            den += w
        return 0.0 if den == 0.0 else num / den

    def calculate_pairwise_agreement(self,
                                     scores_a: List[JudgeScore],
                                     scores_b: List[JudgeScore]) -> float:
        """Weighted pairwise agreement [0, 100] using per-item scale range."""
        a_map = {s.rubric_item_id: s for s in scores_a
                 if s.rubric_item_id in self.item_by_id}
        b_map = {s.rubric_item_id: s for s in scores_b
                 if s.rubric_item_id in self.item_by_id}

        num = den = 0.0
        for item_id, it in self.item_by_id.items():
            sa, sb = a_map.get(item_id), b_map.get(item_id)
            if not sa or not sb:
                continue
            if (str(sa.score).strip().upper() in ('', 'NA', 'N/A') or
                    str(sb.score).strip().upper() in ('', 'NA', 'N/A')):
                continue
            a_norm, ok_a = self._normalize_score(it, sa.score)
            b_norm, ok_b = self._normalize_score(it, sb.score)
            if not (ok_a and ok_b):
                continue
            w    = float(it.weight or 1.0)
            num += w * (1.0 - abs(a_norm - b_norm))
            den += w

        return 0.0 if den == 0.0 else (num / den) * 100.0

    @staticmethod
    def _normalize_score(item: RubricItem, score) -> Tuple[float, bool]:
        """Normalise score to [0, 1] using item-specific scale range."""
        raw = str(score).strip().upper()
        if raw in ('', 'NA', 'N/A', 'NONE', 'NULL'):
            return 0.0, False
        try:
            v = float(raw)
        except (ValueError, TypeError):
            return 0.0, False

        if (item.scale or '').upper() == 'BINARY':
            return (1.0 if v >= 0.5 else 0.0), True

        lo, hi = _detect_scale_range(item)
        if hi == lo:
            return 0.0, False
        v = max(lo, min(hi, v))
        return (v - lo) / (hi - lo), True
