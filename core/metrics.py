"""Structured logging and metrics collection for clinical QA judge runs."""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class EvalRecord:
    """One record per (question, rubric, judge) evaluation."""
    timestamp: str
    question_id: str
    rubric_id: str
    rubric_name: str
    judge_id: str
    aggregate_score: float
    rationales: Dict[str, str]   # {rubric_item_id: rationale text}
    latency_ms: float
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgreementRecord:
    """Pairwise agreement between two judges on one question under one rubric."""
    timestamp: str
    question_id: str
    rubric_id: str
    judge_a: str
    judge_b: str
    agreement_score: float    # 0.0 - 100.0
    agreement_class: str      # fully_agree / majority_agree / split / full_disagree

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """In-memory collector for eval and agreement records."""

    def __init__(self) -> None:
        self.eval_records: List[EvalRecord] = []
        self.agreement_records: List[AgreementRecord] = []
        self.logger = logging.getLogger("clinical_judge_metrics")

    def record_eval(
        self,
        question_id: str,
        rubric_id: str,
        rubric_name: str,
        judge_id: str,
        aggregate_score: float,
        rationales: Dict[str, str],
        latency_ms: float,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        rec = EvalRecord(
            timestamp=datetime.utcnow().isoformat(),
            question_id=question_id,
            rubric_id=rubric_id,
            rubric_name=rubric_name,
            judge_id=judge_id,
            aggregate_score=aggregate_score,
            rationales=rationales,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )
        self.eval_records.append(rec)
        log_level = logging.ERROR if error else logging.INFO
        self.logger.log(
            log_level,
            f"[{rubric_name}] Q={question_id} J={judge_id}: score={aggregate_score:.2f} "
            f"latency={latency_ms:.0f}ms status={status}"
            + (f" error={error}" if error else ""),
        )

    def record_agreement(
        self,
        question_id: str,
        rubric_id: str,
        judge_a: str,
        judge_b: str,
        agreement_score: float,
        agreement_class: str,
    ) -> None:
        rec = AgreementRecord(
            timestamp=datetime.utcnow().isoformat(),
            question_id=question_id,
            rubric_id=rubric_id,
            judge_a=judge_a,
            judge_b=judge_b,
            agreement_score=agreement_score,
            agreement_class=agreement_class,
        )
        self.agreement_records.append(rec)
        self.logger.info(
            f"[Agreement] Q={question_id} Rubric={rubric_id} "
            f"{judge_a}<>{judge_b}: {agreement_score:.1f}% ({agreement_class})"
        )

    def get_eval_records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.eval_records]

    def get_agreement_records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.agreement_records]


_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
