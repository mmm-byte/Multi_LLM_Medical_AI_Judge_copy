"""Lightweight LLM client for vLLM OpenAI-compatible endpoints.
Used for HPC-local judge calls.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("llm_client")


class LLMClient:
    """Simple synchronous client for vLLM OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        api_key: str = "EMPTY",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        t0 = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self.headers,
            )
        resp.raise_for_status()
        latency_ms = (time.time() - t0) * 1000
        result = resp.json()
        result["_latency_ms"] = latency_ms
        return result

    def complete(self, prompt: str, **kwargs) -> str:
        """Convenience wrapper — returns just the assistant text."""
        resp = self.chat([{"role": "user", "content": prompt}], **kwargs)
        return resp["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        """Returns True if the vLLM endpoint is reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed for {self.base_url}: {e}")
            return False
