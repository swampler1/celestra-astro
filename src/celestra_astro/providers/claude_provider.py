from __future__ import annotations

import os
from typing import Any

from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    def __init__(self) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:
            raise ImportError("Install `anthropic` to use provider=claude.") from exc

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

        self._client = Anthropic(api_key=api_key)

    @staticmethod
    def _format_error(model: str, exc: Exception) -> RuntimeError:
        text = str(exc)
        lowered = text.lower()

        if "credit balance is too low" in lowered or "plans & billing" in lowered:
            return RuntimeError(
                f"Claude request failed for model '{model}': Anthropic account has insufficient API credits. "
                "Please add credits or switch providers."
            )

        request_id = getattr(exc, "request_id", None)
        if request_id:
            return RuntimeError(
                f"Claude request failed for model '{model}': {text} (request_id={request_id})"
            )

        response: Any = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            header_request_id = headers.get("request-id") or headers.get("x-request-id")
            if header_request_id:
                return RuntimeError(
                    f"Claude request failed for model '{model}': {text} (request_id={header_request_id})"
                )

        return RuntimeError(f"Claude request failed for model '{model}': {text}")

    def generate(self, *, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        user_parts = [m.get("content", "") for m in messages if m.get("role") != "system"]
        system_prompt = "\n\n".join(system_parts).strip()
        user_prompt = "\n\n".join(user_parts).strip()

        try:
            response = self._client.messages.create(
                model=model,
                system=system_prompt if system_prompt else None,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise self._format_error(model, exc) from exc

        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        final = "".join(parts).strip()
        if not final:
            raise RuntimeError(f"Claude returned an empty response for model '{model}'.")
        return final
