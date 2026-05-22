from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class RubricItemSchema(BaseModel):
    """Schema for a single rubric item taken from a published paper."""
    id: str
    name: str
    description: str = ""
    scale: str = "LIKERT"  # LIKERT or BINARY
    weight: float = 1.0
    source_paper: Optional[str] = None  # e.g. "PEMAT (Shoemaker et al. 2014)"

    @validator("scale")
    def validate_scale(cls, v: str) -> str:
        allowed = {"LIKERT", "BINARY"}
        if v.upper() not in allowed:
            raise ValueError(f"scale must be one of {allowed}")
        return v.upper()

    @validator("weight")
    def validate_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("weight must be > 0")
        return v


class RubricSchema(BaseModel):
    """Schema for a complete published rubric instrument."""
    id: Optional[str] = None
    name: str
    source_paper: str  # Full citation of the paper this rubric comes from
    source_url: Optional[str] = None
    items: List[RubricItemSchema] = Field(default_factory=list)

    @validator("items")
    def at_least_one_item(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("rubric must have at least one item")
        return v
