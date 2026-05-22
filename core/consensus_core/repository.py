from __future__ import annotations

from typing import Dict, Optional
from .events import EventLog


class InMemoryStore:
    def __init__(self) -> None:
        self._logs: Dict[str, EventLog] = {}

    def get(self, question_id: str) -> Optional[EventLog]:
        return self._logs.get(question_id)

    def put(self, question_id: str, log: EventLog) -> None:
        self._logs[question_id] = log

    def all_ids(self):
        return list(self._logs.keys())
