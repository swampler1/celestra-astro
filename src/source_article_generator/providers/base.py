from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, *, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

