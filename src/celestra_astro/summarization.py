from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)

CHEAP_MODEL = "gpt-5.6-luna"
DEFAULT_SAFETY_MAX_CHUNKS = 30

from .providers.base import LLMProvider
from .providers.factory import get_provider
from .papers import download_pdf_for_bibcode, extract_text_from_pdf
from .prompts import (
    DEFAULT_SUMMARY_CHUNK_TEMPLATE,
    DEFAULT_SUMMARY_MERGE_TEMPLATE,
    load_template,
    render_template,
)
from .utils import safe_slug


# Splits text of PDF,s into chunks to be read by chatGPT
def chunk_text(text: str, max_chars: int = 3500) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end
    return chunks


# Generates a technical summary of a paper's text using the selected LLM.
# Takes the raw extracted PDF text and produces a concise, domain-relevant summary focusing on observational methods, key measurements, and results.
def summarize_paper_text(
    bibcode: str,
    raw_text: str,
    model: str = CHEAP_MODEL,
    provider: LLMProvider | None = None,
    summary_chunk_template_path: str | None = None,
    summary_merge_template_path: str | None = None,
    max_chunks: int = 0,
    chunk_size_chars: int = 3500,
    safety_max_chunks: int = DEFAULT_SAFETY_MAX_CHUNKS,
    chunk_temperature: float = 0.2,
    chunk_max_tokens: int = 1500,
    merge_temperature: float = 0.1,
    merge_max_tokens: int = 1000,
) -> str:
    if provider is None:
        provider = get_provider("openai")
    summary_chunk_template = load_template(summary_chunk_template_path, DEFAULT_SUMMARY_CHUNK_TEMPLATE)
    summary_merge_template = load_template(summary_merge_template_path, DEFAULT_SUMMARY_MERGE_TEMPLATE)

    chunks = chunk_text(raw_text, max_chars=chunk_size_chars)
    if max_chunks > 0 and len(chunks) > max_chunks:
        print(f"[INFO] {bibcode}: {len(chunks)} chunks → truncating to {max_chunks}.")
        chunks = chunks[:max_chunks]

    if safety_max_chunks > 0 and len(chunks) > safety_max_chunks:
        print(
            f"[WARN] {bibcode}: safety limit reached; processing the first "
            f"{safety_max_chunks} of {len(chunks)} chunks."
        )
        chunks = chunks[:safety_max_chunks]

    chunk_summaries: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Summarizing {bibcode} chunk {i}/{len(chunks)}...")
        summary = provider.generate(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert astrophysicist. You write concise but highly technical summaries "
                        "of scientific texts. You do not invent facts or numbers; you only use information "
                        "explicitly present in the text you are given."
                    ),
                },
                {
                    "role": "user",
                    "content": render_template(
                        summary_chunk_template,
                        bibcode=bibcode,
                        chunk_index=i,
                        chunk_total=len(chunks),
                        chunk_text=chunk,
                    ),
                },
            ],
            temperature=chunk_temperature,
            max_tokens=chunk_max_tokens,
        )
        chunk_summaries.append(summary)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    joined = "\n\n".join(chunk_summaries)
    print(f"[INFO] Combining {len(chunk_summaries)} chunk summaries for {bibcode}...")
    return provider.generate(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert astrophysicist. You combine section-level summaries of a paper into a single, "
                    "coherent, technical summary. You do not invent facts; you only use what is present in the input summaries."
                ),
            },
            {
                "role": "user",
                "content": render_template(
                    summary_merge_template,
                    bibcode=bibcode,
                    joined_summaries=joined,
                ),
            },
        ],
        temperature=merge_temperature,
        max_tokens=merge_max_tokens,
    )


# Builds the filesystem path where the summary for a given bibcode should be saved
def summary_path_for_bibcode(bibcode: str, summary_dir: str) -> str:
    return os.path.join(summary_dir, safe_slug(bibcode) + "_summary.txt")


# Ensures every paper has a generated summary by downloading PDFs and creating summaries if missing
def ensure_summaries_for_papers(
    papers,
    pdf_dir,
    summary_dir,
    model: str = CHEAP_MODEL,
    provider: LLMProvider | None = None,
    summary_chunk_template_path: str | None = None,
    summary_merge_template_path: str | None = None,
    max_chunks_per_paper: int = 0,
    chunk_size_chars: int = 3500,
    safety_max_chunks_per_paper: int = DEFAULT_SAFETY_MAX_CHUNKS,
    summary_chunk_temperature: float = 0.2,
    summary_chunk_max_tokens: int = 1500,
    summary_merge_temperature: float = 0.1,
    summary_merge_max_tokens: int = 1000,
) -> None:
    if provider is None:
        provider = get_provider("openai")

    for p in papers:
        bib = (
            p.get("bibcode")
            or p.get("Bibcode")
            or p.get("Bibcode/DOI")
        )
        if not bib:
            print("[WARN] Paper entry missing 'bibcode', skipping:", p)
            continue

        spath = summary_path_for_bibcode(bib, summary_dir)
        if os.path.exists(spath):
            print(f"[INFO] Summary already exists for {bib}: {spath}")
            continue

        # Make sure we have a PDF locally; if not, try to download it
        pdf_filename = safe_slug(bib) + ".pdf"
        pdf_path = os.path.join(pdf_dir, pdf_filename)

        if not os.path.exists(pdf_path):
            print(f"[INFO] No local PDF found for {bib} in {pdf_dir}. Attempting download...")
            ok = download_pdf_for_bibcode(bib, pdf_dir)
            if not ok:
                print(f"[ERROR] Could not obtain PDF for {bib}; skipping summary.")
                continue
            # pdf_path should now exist if download succeeded

        try:
            print(f"[INFO] Creating summary for {bib}...")
            raw_text = extract_text_from_pdf(pdf_path)
            summary_text = summarize_paper_text(
                bib,
                raw_text,
                model=model,
                provider=provider,
                summary_chunk_template_path=summary_chunk_template_path,
                summary_merge_template_path=summary_merge_template_path,
                max_chunks=max_chunks_per_paper,
                chunk_size_chars=chunk_size_chars,
                safety_max_chunks=safety_max_chunks_per_paper,
                chunk_temperature=summary_chunk_temperature,
                chunk_max_tokens=summary_chunk_max_tokens,
                merge_temperature=summary_merge_temperature,
                merge_max_tokens=summary_merge_max_tokens,
            )
            with open(spath, "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"[INFO] Summary written: {spath}")
        except Exception as e:
            print(f"[ERROR] Failed to summarize {bib}: {e}")


# Builds the full text block of all paper summaries to inject into the article prompt
def build_summaries_block(papers: list[dict], summary_dir: str, start_index: int = 1) -> str:
    blocks = []
    for i, p in enumerate(papers, start=start_index):
        bib = (
            p.get("bibcode")
            or p.get("Bibcode")
            or p.get("Bibcode/DOI")
            or f"UNKNOWN_{i}"
        )
        spath = summary_path_for_bibcode(bib, summary_dir)
        if not os.path.exists(spath):
            blocks.append(f"[{i}] {bib}\n(No summary available.)")
            continue
        with open(spath, "r", encoding="utf-8") as f:
            stext = f.read().strip()
        blocks.append(f"[{i}] {bib}\n{stext}")
    return "\n\n".join(blocks)
