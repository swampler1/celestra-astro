from __future__ import annotations

import textwrap


DEFAULT_SUMMARY_CHUNK_TEMPLATE = textwrap.dedent(
    """
    Paper bibcode: {bibcode}
    Chunk {chunk_index} of {chunk_total}.

    Task:
    Read the following chunk and write a structured, technical summary of JUST THIS CHUNK.

    Focus on:
    - Scientific context and target(s) mentioned (especially DG Tau, its disk, jet, or environment, if present).
    - Observational / experimental setup: facilities (ALMA, HST, VLT, etc.), bands, frequencies, spectral setup,
      angular / spectral resolution, sensitivities, integration times if present.
    - Key measurements: radii, velocities, temperatures, column densities, masses, line IDs, S/N, fluxes,
      position angles, inclinations, etc. Include numeric values and units when given.
    - Main results and interpretations that are directly supported by these measurements.
    - Any explicit uncertainties, limits, caveats, or open questions stated in this chunk.

    Constraints:
    - Do NOT make a general high-level summary, your goal is to extract as many concrete, technical facts as possible per chunk.
    - Do NOT speculate or add information that is not explicitly in the text.
    - Try to not keep it breif, details from the papers matter, use most of your tokens if possible.
    - If the chunk is mostly figure/table captions, summarize what they quantify
      (e.g., relation between NT and radius, S/N vs velocity, etc.).
    - Write in prose (no bullet lists) and keep the tone appropriate for an expert astrophysicist.

    Chunk text:
    {chunk_text}
    """
).strip()


DEFAULT_SUMMARY_MERGE_TEMPLATE = textwrap.dedent(
    """
    Paper bibcode: {bibcode}

    Task:
    Combine the following partial summaries (each derived from different chunks of the paper) into ONE coherent,
    technical summary of the entire paper, suitable for another expert.

    Focus on:
    - The main scientific context and motivation.
    - Observational / experimental setup: instruments, facilities, bands, spectral and spatial resolution, sensitivities.
    - Key measurements and numerical results (radii, velocities, temperatures, column densities, masses, line fluxes, S/N, etc.).
    - DG Tau-specific results, if the paper includes DG Tau (e.g. jet properties, disk structure, outflows, line emission).
    - Main conclusions and how they follow from the data.
    - Important caveats, uncertainties, or limits explicitly mentioned.

    Constraints:
    - Do NOT speculate or add any results not supported by the partial summaries.
    - If DG Tau is not a target of the paper, make that clear and summarize what the paper *does* focus on.
    - Avoid redundancy: if multiple chunks say the same thing, state it once, clearly.
    - Write in coherent paragraphs (no bullet lists).

    Partial summaries:
    {joined_summaries}
    """
).strip()


DEFAULT_ARTICLE_TEMPLATE = textwrap.dedent(
    """
    SOURCE PARAMETERS TABLE FOR {target}:
    {source_parameters_block}

    USE OF SOURCE PARAMETERS:
    - The following global properties MUST be taken from the source-parameter table, not from the papers:
      (1) Distance
      (2) Stellar mass
      (3) Systemic velocity
      (4) Disk position angle (PA)
      (5) Disk inclination
    - These values override any conflicting measurements in the paper summaries.
    - These five quantities MUST be stated clearly in the article.
    - Whenever the article mentions any of these five values, it MUST cite [1].

    Write a wiki-style article in PLAIN TEXT summarizing the characteristics and interesting research
    around the protoplanetary disk {canonical}.

    This article is for an expert reader. Avoid general knowledge or broad statements. Prioritize
    concrete, technical details:
    - specific measurements (radii, velocities, temperatures, masses, inclinations, position angles,
      fluxes, line IDs),
    - observational methods (instruments, arrays/configurations, wavelengths, spectral setups,
      analysis techniques),
    - and key results, including uncertainties where available.

    You are given technical summaries of each paper. Use ONLY these summaries as your information
    sources; do not invent facts beyond them.

    PAPER SUMMARIES (indexed [n] matching the source list):
    {summaries_block}

    SOURCES (index → bibcode):
    {sources_str}

    Formatting rules for the OUTPUT:
    - PLAIN TEXT ONLY: no markdown, no bullet lists, no numbered lists, no code blocks, no **bold** or *italic*.
    - Organize with simple section titles written on their own lines (for example:
      Overview
      Disk Structure and Kinematics
      Physical Conditions
      Chemical Inventory
      Outflows and Jets
      Observational Methods
      Key Results
      Open Questions
      References)

    Length and depth requirements:
    - The overall article should be at least 2000 words; do not be brief or high-level when detailed information is available in the summaries.
    - The following sections should each be at least 200–400 words, with dense technical content: Disk Structure and Kinematics; Outflows and Jets; Physical Conditions; Observational Methods; Key Results and Interpretation.
    - If the summaries provide many specific measurements, methods, and physical interpretations, it is acceptable (and preferred) for the article to be longer than 3000 words.

    Use of sources:
    - Use information from all of the provided paper summaries; do not ignore a source unless it is clearly irrelevant.
    - When multiple papers address the same topic (for example, jet velocities, knot spacing, or radio shocks), compare and synthesize their results instead of repeating the same generic statements.
    - Prefer sentences that contain specific numbers, methods, and uncertainties over vague qualitative phrases.

    Object specificity:
    - Only attribute numerical values and detailed properties to {canonical} when the summaries clearly state they refer to this object. If a number or result is explicitly for a different system (e.g. another comparison disk), omit it.

    Content rules:
    - Each main section should contain specific numbers and methods wherever possible, not just
      qualitative descriptions.
    - Every time you use information from a paper summary, you MUST add an inline citation
      immediately after the relevant clause or sentence, in square brackets, using the
      index of the paper, e.g. [2], [3], [4].
    - If a sentence combines information from multiple papers, include all relevant indices,
      e.g. [2, 4, 6].
    - If a summary block says "(No summary available.)", ignore that source and do not cite it.
    - If the summaries do not provide a detail (e.g., a specific mass or radius), state that the
      value is not well constrained instead of inventing a number.
    - Avoid speculation or hallucination; stay within what can be reasonably inferred from the
      provided summaries.

    In the References section, list each source as:
    [n] FirstAuthor et al., Year, Title (or description if unknown), Journal (if known), Bibcode

    The first reference should be the Source-parameter Table in the style "[1] source-parameter data table"
    """
).strip()


class _SafeDict(dict):
    def __missing__(self, key):  # type: ignore[override]
        return "{" + key + "}"


def load_template(path: str | None, default_text: str) -> str:
    if not path:
        return default_text
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or default_text
    except Exception:
        return default_text


def render_template(template: str, **values) -> str:
    return template.format_map(_SafeDict(values))

