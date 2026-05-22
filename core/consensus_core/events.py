from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime, timezone
import json
import uuid


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class Event:
    id: str
    type: str
    data: Dict[str, Any]
    created_at: str = field(default_factory=_utc_now_iso)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class EventLog:
    def __init__(self, events: Optional[Iterable[Event]] = None) -> None:
        self._events: List[Event] = list(events or [])

    def append(self, event_type: str, data: Dict[str, Any]) -> Event:
        evt = Event(id=new_event_id(), type=event_type, data=data)
        self._events.append(evt)
        return evt

    def __iter__(self):
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def to_jsonl(self) -> str:
        return "\n".join(evt.to_json() for evt in self._events)

    @staticmethod
    def from_jsonl(blob: str) -> "EventLog":
        if not blob.strip():
            return EventLog()
        return EventLog([Event(**json.loads(line)) for line in blob.splitlines()])


def append_judgment_recorded(
    log: EventLog,
    question_id: str,
    rubric_id: str,
    judge_id: str,
    score_ids: List[str],
) -> Event:
    return log.append("JudgmentRecorded", {
        "question_id": question_id,
        "rubric_id": rubric_id,
        "judge_id": judge_id,
        "score_ids": list(score_ids),
    })


def append_agreement_classified(
    log: EventLog,
    question_id: str,
    rubric_id: str,
    agreement_class: str,
    outlier_judge: Optional[str],
    mean_pairwise: float,
) -> Event:
    return log.append("AgreementClassified", {
        "question_id": question_id,
        "rubric_id": rubric_id,
        "agreement_class": agreement_class,
        "outlier_judge": outlier_judge,
        "mean_pairwise_agreement": mean_pairwise,
    })
