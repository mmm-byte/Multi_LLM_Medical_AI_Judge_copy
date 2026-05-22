"""Endpoint configuration for local vLLM judges on HPC."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointConfig:
    """Configuration for one vLLM judge endpoint."""
    id: str                  # e.g. "medgemma"
    model: str               # e.g. "google/medgemma-4b-it"
    url: str                 # e.g. "http://localhost:8001"
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    description: Optional[str] = None


DEFAULT_JUDGE_ENDPOINTS = [
    EndpointConfig(
        id="medgemma",
        model="google/medgemma-4b-it",
        url=os.getenv("LOCAL_VLLM_JUDGE1_URL", "http://localhost:8001"),
        description="MedGemma 4B — Google DeepMind medical fine-tune",
    ),
    EndpointConfig(
        id="biomistral",
        model="BioMistral/BioMistral-7B",
        url=os.getenv("LOCAL_VLLM_JUDGE2_URL", "http://localhost:8002"),
        description="BioMistral 7B — biomedical corpus fine-tune",
    ),
    EndpointConfig(
        id="meditron",
        model="epfl-llm/meditron-7b",
        url=os.getenv("LOCAL_VLLM_JUDGE3_URL", "http://localhost:8003"),
        description="Meditron 7B — EPFL medical LLM",
    ),
    EndpointConfig(
        id="biomedlm",
        model="stanford-crfm/BioMedLM",
        url=os.getenv("LOCAL_VLLM_JUDGE4_URL", "http://localhost:8004"),
        description="BioMedLM — Stanford CRFM biomedical LM",
    ),
]


def get_judge_configs_as_dicts():
    """Return judge list as plain dicts for JSON config files."""
    return [
        {"id": e.id, "model": e.model, "url": e.url,
         "max_tokens": e.max_tokens, "temperature": e.temperature,
         "timeout_seconds": e.timeout_seconds}
        for e in DEFAULT_JUDGE_ENDPOINTS
    ]
