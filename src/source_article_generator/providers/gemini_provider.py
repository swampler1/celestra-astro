from __future__ import annotations

import os
import random
import re
import threading
import time
import warnings
from collections import defaultdict, deque
from typing import Any

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    _rate_lock = threading.Lock()
    _request_history: dict[str, deque[float]] = defaultdict(deque)

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self._request_limit = max(1, int(os.getenv("SOURCE_ARTICLE_GENERATOR_GEMINI_REQUEST_LIMIT", "5")))
        self._window_seconds = max(1.0, float(os.getenv("SOURCE_ARTICLE_GENERATOR_GEMINI_WINDOW_SECONDS", "65")))
        self._max_retries = max(0, int(os.getenv("SOURCE_ARTICLE_GENERATOR_GEMINI_MAX_RETRIES", "8")))
        self._default_retry_seconds = max(
            1.0,
            float(os.getenv("SOURCE_ARTICLE_GENERATOR_GEMINI_DEFAULT_RETRY_SECONDS", "35")),
        )
        self._max_retry_seconds = max(
            self._default_retry_seconds,
            float(os.getenv("SOURCE_ARTICLE_GENERATOR_GEMINI_MAX_RETRY_SECONDS", "90")),
        )
        self._backend = ""
        self._genai: Any | None = None
        self._client: Any | None = None
        self._genai_types: Any | None = None

        try:
            from google import genai as genai_client  # type: ignore
            from google.genai import types as genai_types  # type: ignore

            self._client = genai_client.Client(api_key=api_key)
            self._genai_types = genai_types
            self._backend = "google_genai"
            return
        except Exception:
            pass

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import google.generativeai as genai  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Could not import a Gemini SDK. Install `google-genai` (preferred) or "
                "upgrade `google-generativeai` in your Python 3.12 environment."
            ) from exc

        if not hasattr(genai, "GenerativeModel"):
            version = getattr(genai, "__version__", "unknown")
            raise ImportError(
                "Detected `google-generativeai`, but it is too old for Gemini model calls "
                f"(version: {version}). Upgrade `google-generativeai` or install `google-genai`."
            )

        genai.configure(api_key=api_key)
        self._genai = genai
        self._backend = "google_generativeai"

    def _wait_for_rate_limit_slot(self, model: str) -> None:
        while True:
            with self._rate_lock:
                history = self._request_history[model]
                now = time.monotonic()
                while history and now - history[0] >= self._window_seconds:
                    history.popleft()

                if len(history) < self._request_limit:
                    history.append(now)
                    return

                sleep_for = (history[0] + self._window_seconds) - now + 0.5

            sleep_for = max(1.0, sleep_for)
            print(
                f"[INFO] Gemini throttle active for model '{model}'; "
                f"sleeping {sleep_for:.1f}s to stay under {self._request_limit} requests/"
                f"{self._window_seconds:.0f}s."
            )
            time.sleep(sleep_for)

    def _generate_once(self, *, model: str, prompt: str, temperature: float, max_tokens: int):
        if self._backend == "google_genai":
            config: Any = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if self._genai_types is not None and hasattr(self._genai_types, "GenerateContentConfig"):
                config = self._genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            return self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

        model_obj = self._genai.GenerativeModel(model_name=model)
        return model_obj.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        try:
            parts = []
            candidates = getattr(response, "candidates", []) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    ptxt = getattr(part, "text", None)
                    if ptxt:
                        parts.append(ptxt)
            joined = "".join(parts).strip()
            if joined:
                return joined
        except Exception:
            pass

        raise RuntimeError("Gemini returned an empty response.")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "429",
                "resource_exhausted",
                "quota exceeded",
                "rate limit",
                "retry in",
            )
        )

    def _extract_retry_delay(self, exc: Exception, attempt: int) -> float:
        text = str(exc)
        delays = []

        retry_in_matches = re.findall(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
        delays.extend(float(match) for match in retry_in_matches)

        retry_delay_seconds = re.findall(r"retry_delay\s*\{[^}]*seconds:\s*(\d+)", text, flags=re.IGNORECASE | re.DOTALL)
        delays.extend(float(match) for match in retry_delay_seconds)

        if delays:
            delay = max(delays)
        else:
            delay = min(self._default_retry_seconds * (2 ** attempt), self._max_retry_seconds)

        delay = min(delay, self._max_retry_seconds)
        return delay + random.uniform(0.5, 1.5)

    def generate(self, *, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        user_parts = [m.get("content", "") for m in messages if m.get("role") != "system"]
        prompt = "\n\n".join([p for p in (["\n\n".join(system_parts), "\n\n".join(user_parts)]) if p]).strip()

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._wait_for_rate_limit_slot(model)
            try:
                response = self._generate_once(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._extract_text(response)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries or not self._is_retryable_error(exc):
                    break

                retry_delay = self._extract_retry_delay(exc, attempt)
                print(
                    f"[WARN] Gemini request hit a temporary quota/rate limit for model '{model}'. "
                    f"Retrying in {retry_delay:.1f}s "
                    f"(attempt {attempt + 1}/{self._max_retries + 1})."
                )
                time.sleep(retry_delay)

        raise RuntimeError(f"Gemini request failed for model '{model}': {last_exc}") from last_exc
