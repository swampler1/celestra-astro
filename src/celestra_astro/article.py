from __future__ import annotations

import datetime
import logging
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)

from .providers.base import LLMProvider
from .providers.factory import get_provider


# This block generates the plain text article from the given bib codes.
def generate_article(
    prompt: str,
    output_basename: str,
    *,
    model: str = "gpt-5.6-terra",
    temperature: float = 0.2,
    max_tokens: int = 16000,
    out_dir: str = "wiki_articles",
    logs_dir: str = "logs",
    provider: LLMProvider | None = None,
) -> str:
    """
    Generate a PLAIN TEXT wiki-style article and write it to <out_dir>/<output_basename>.txt.
    Returns the file path to the written article.

    Notes:
    - No markdown is produced; the model is instructed to output plain text only.
    - A run log with the prompt and a short response preview is saved under ./logs.
    - Provider is resolved lazily at runtime (OpenAI by default).
    """
    if provider is None:
        provider = get_provider("openai")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(logs_dir, f"logs_for_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    log_path = os.path.join(run_log_dir, f"{output_basename}_plaintext_generation_{timestamp}.log")

    system_message = (
        "You are a scientific writer producing a detailed wiki article for experts in PLAIN TEXT only. "
        "Do not use markdown, bullets, or code blocks. Use simple section titles on their own lines and "
        "tight, well-structured paragraphs. Prioritize methods, measurements, uncertainties, and concrete "
        "results over generalities. Use inline numeric citation markers like [n] that map to a provided source list."
    )

    article_txt = provider.generate(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = None

    # Write the article
    out_path = os.path.join(out_dir, f"{output_basename}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(article_txt)

    # Minimal run log
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"=== Plaintext Article for: {output_basename} ===\n\n")
        log.write("[System Message]\n")
        log.write(system_message + "\n\n")
        log.write("[Prompt]\n")
        log.write(prompt + "\n\n")
        log.write("[Response Preview]\n")
        log.write((article_txt[:1000] if article_txt else "") + ("...\n\n" if article_txt and len(article_txt) > 1000 else "\n\n"))
        if usage:
            log.write("[Token Usage]\n")
            log.write(f"  prompt_tokens: {usage.prompt_tokens}\n")
            log.write(f"  completion_tokens: {usage.completion_tokens}\n")
            log.write(f"  total_tokens: {usage.total_tokens}\n")
        log.write("=" * 60 + "\n\n")

    logging.info("Plain-text article written to %s", out_path)
    return out_path
