from __future__ import annotations

import os

from openai import OpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = OpenAI()

    def generate(self, *, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        try:
            if model.startswith("gpt-5"):
                # GPT-5.5/5.6 are reasoning-capable models. Use Responses API
                # parameters for them; temperature is intentionally not passed.
                effort = os.getenv("OPENAI_REASONING_EFFORT", "none")
                response = self._client.responses.create(
                    model=model,
                    input=messages,
                    reasoning={"effort": effort},
                    max_output_tokens=max_tokens,
                )
                text = response.output_text
            else:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed for model '{model}': {exc}") from exc

        if not text:
            raise RuntimeError(f"OpenAI returned an empty response for model '{model}'.")
        return text.strip()
