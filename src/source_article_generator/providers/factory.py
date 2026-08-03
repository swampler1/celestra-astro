from __future__ import annotations

from .base import LLMProvider
from .claude_provider import ClaudeProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider

DEFAULT_MODELS = {
    "openai": {"summary": "gpt-5.6-luna", "article": "gpt-5.6-terra"},
    "gemini": {"summary": "gemini-2.5-flash-lite", "article": "gemini-2.5-flash"},
    "claude": {"summary": "claude-3-5-sonnet-latest", "article": "claude-3-5-sonnet-latest"},
}


def get_provider(name: str) -> LLMProvider:
    normalized = (name or "openai").strip().lower()
    if normalized == "openai":
        return OpenAIProvider()
    if normalized == "gemini":
        return GeminiProvider()
    if normalized == "claude":
        return ClaudeProvider()
    raise ValueError(f"Unsupported provider '{name}'. Use openai, gemini, or claude.")


def default_model_for(provider: str, purpose: str) -> str:
    normalized_provider = (provider or "openai").strip().lower()
    try:
        return DEFAULT_MODELS[normalized_provider][purpose]
    except KeyError as exc:
        raise ValueError(f"No default model mapping for provider={provider!r}, purpose={purpose!r}") from exc


def validate_model_for_provider(provider: str, model: str) -> None:
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()
    if not m:
        raise ValueError("Model name cannot be empty.")

    if p == "openai" and not m.startswith("gpt-"):
        raise ValueError(f"Model '{model}' does not look like an OpenAI GPT model for provider=openai.")
    if p == "gemini" and not m.startswith("gemini"):
        raise ValueError(f"Model '{model}' does not look like a Gemini model for provider=gemini.")
    if p == "claude" and not m.startswith("claude-"):
        raise ValueError(f"Model '{model}' does not look like a Claude model for provider=claude.")
