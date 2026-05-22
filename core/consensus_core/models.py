from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Union
import uuid


ScaleType = Literal["LIKERT", "BINARY"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    category: Optional[str] = None   # e.g. Cardiology | Pharmacology | Neurology | Pediatrics | Emergency
    source: Optional[str] = None     # e.g. MedQuAD, MedDialog, Medical Meadow, guideline summary
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass(frozen=True)
class Answer:
    id: str
    text: str
    provider: Optional[str] = None   # which LLM generated this answer
    meta: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass(frozen=True)
class RubricItem:
    id: str
    name: str
    description: str
    scale: ScaleType
    weight: float = 1.0
    source_paper: Optional[str] = None   # paper this item comes from


@dataclass(frozen=True)
class Rubric:
    id: str
    name: str
    source_paper: str   # e.g. "PEMAT (Shoemaker et al. 2014)"
    source_url: Optional[str] = None
    items: List[RubricItem] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)


ScoreValue = Union[int, float, Literal["NA"]]


@dataclass(frozen=True)
class JudgeScore:
    id: str
    judge_id: str
    rubric_item_id: str
    score: ScoreValue
    rationale: Optional[str] = None   # always printed for paper output
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass(frozen=True)
class Critique:
    id: str
    judge_id: str
    text: str
    created_at: str = field(default_factory=_utc_now_iso)


def to_dict(obj) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Unsupported type for to_dict: {type(obj)}")
