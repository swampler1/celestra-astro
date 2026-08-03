from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from .article import generate_article
from .config import RunConfig
from .prompts import DEFAULT_ARTICLE_TEMPLATE, load_template, render_template
from .providers.factory import get_provider, validate_model_for_provider
from .source_parameters import format_source_parameters_block, load_source_parameters_table
from .sources import get_disk_lit, get_simbad_aliases, select_papers
from .summarization import build_summaries_block, ensure_summaries_for_papers
from .utils import safe_slug


def _resolve_path(base: str | None, path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if not base:
        return str(p)
    return str((Path(base) / p).resolve())


def _write_resolved_config(run_root: str, config: RunConfig, targets_list) -> str:
    payload = {
        "run_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "targets": list(targets_list),
        "config": config.to_dict(),
    }
    out_dir = Path(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yaml  # type: ignore

        out_path = out_dir / "run_config.resolved.yaml"
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return str(out_path)
    except Exception:
        out_path = out_dir / "run_config.resolved.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(out_path)


def _build_extra_context_block(project_root: str | None, files: list[str]) -> str:
    if not files:
        return ""

    blocks = []
    for raw_path in files:
        resolved = _resolve_path(project_root, raw_path)
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as exc:
            print(f"[WARN] Could not load extra context file '{resolved}': {exc}")
            continue

        if not content:
            continue

        blocks.append(
            f"EXTRA CONTEXT FILE: {raw_path}\n"
            f"{content}"
        )

    if not blocks:
        return ""
    return "\n\n".join(blocks)


def run_targets(targets_list, config: RunConfig | None = None, project_root: str | None = None):
    cfg = config or RunConfig()
    validate_model_for_provider(cfg.provider, cfg.summary_model)
    validate_model_for_provider(cfg.provider, cfg.article_model)
    provider = get_provider(cfg.provider)

    source_parameters_csv_path = _resolve_path(project_root, cfg.source_parameters_csv_path)
    scraped_table_dir = _resolve_path(project_root, cfg.scraped_table_dir)
    paper_pdfs_root = _resolve_path(project_root, cfg.paper_pdfs_root)
    summaries_root = _resolve_path(project_root, cfg.summaries_root)
    articles_dir = _resolve_path(project_root, cfg.articles_dir)
    logs_dir = _resolve_path(project_root, cfg.logs_dir)
    summary_chunk_template_path = _resolve_path(project_root, cfg.summary_chunk_template)
    summary_merge_template_path = _resolve_path(project_root, cfg.summary_merge_template)
    article_template_path = _resolve_path(project_root, cfg.article_template)

    run_root = cfg.output_root.strip()
    if run_root:
        resolved_run_root = _resolve_path(project_root, run_root)
        os.makedirs(resolved_run_root, exist_ok=True)

        scraped_table_dir = os.path.join(resolved_run_root, cfg.scraped_table_dir)
        paper_pdfs_root = os.path.join(resolved_run_root, cfg.paper_pdfs_root)
        summaries_root = os.path.join(resolved_run_root, cfg.summaries_root)
        articles_dir = os.path.join(resolved_run_root, cfg.articles_dir)
        logs_dir = os.path.join(resolved_run_root, cfg.logs_dir)
        _write_resolved_config(resolved_run_root, cfg, targets_list)
    else:
        _write_resolved_config(logs_dir, cfg, targets_list)

    extra_context_block = _build_extra_context_block(project_root, cfg.extra_context_files)
    article_template = load_template(article_template_path, DEFAULT_ARTICLE_TEMPLATE)

    SOURCE_PARAMETERS = load_source_parameters_table(source_parameters_csv_path)
    if not SOURCE_PARAMETERS:
        print(f"[WARN] SOURCE_PARAMETERS is empty. Check SOURCE_PARAMETERS_CSV_PATH = {source_parameters_csv_path}")
    else:
        print(f"[INFO] SOURCE_PARAMETERS has {len(SOURCE_PARAMETERS)} entries.")

    for target in targets_list:
        # Accept filesystem-friendly target names such as DG_Tau while using
        # the normal astronomical spelling for SIMBAD and source-table lookup.
        lookup_target = target.replace("_", " ")
        canonical, aliases = get_simbad_aliases(lookup_target)

        print("\n" + "=" * 72)
        print(f"Target: {target}  (canonical: {canonical})")
        print("=" * 72 + "\n")

        print(f"Aliases ({len(aliases)} total):")
        for a in aliases[:10]:
            print(f"  • {a}")
        if len(aliases) > 10:
            print(f"  … ({len(aliases) - 10} more not shown)")
        print()

        alias_note = (
            f"TARGET (canonical): {canonical}\n"
            f"Treat the following as exact aliases for the same object: {', '.join(aliases)}\n\n"
            if aliases else
            f"TARGET (canonical): {canonical}\n\n"
        )

        canonical_path = safe_slug(canonical)

        source_parameters_block = format_source_parameters_block(lookup_target, SOURCE_PARAMETERS)
        print("source parameters:")
        print(source_parameters_block)
        print()

        print("Directory structure:")
        os.makedirs(paper_pdfs_root, exist_ok=True)
        os.makedirs(summaries_root, exist_ok=True)
        os.makedirs(scraped_table_dir, exist_ok=True)

        PDF_DIR = os.path.join(paper_pdfs_root, f"{canonical_path}_papers_pdf")
        SUMMARY_DIR = os.path.join(summaries_root, f"{canonical_path}_paper_summaries")
        os.makedirs(PDF_DIR, exist_ok=True)
        os.makedirs(SUMMARY_DIR, exist_ok=True)

        print(f"  • PDF Directory:      {PDF_DIR}")
        print(f"  • Summary Directory:  {SUMMARY_DIR}\n")

        csv_path = os.path.join(scraped_table_dir, f"{canonical_path}_scraped_table.csv")
        if not os.path.exists(csv_path):
            try:
                print(f"[INFO] No CSV found → scraping SIMBAD for '{canonical}' ...")
                get_disk_lit(canonical, save_dir=scraped_table_dir)
            except Exception as e:
                print(f"[ERROR] Error scraping SIMBAD for {target}: {e}")
                continue
        if not os.path.exists(csv_path):
            print(
                f"[ERROR] SIMBAD literature table was not created for {canonical}; "
                "skipping this target."
            )
            continue
        else:
            print(f"[INFO] Using existing CSV: {csv_path}")

        print("[STEP] Selecting top papers...")
        papers = select_papers(
            canonical,
            score_limit=cfg.score_limit,
            paper_limit=cfg.paper_limit,
            scraped_table_dir=scraped_table_dir,
        )

        if not papers:
            print("No valid papers found.\n")
            continue

        print(f"Selected {len(papers)} paper(s):")
        for p in papers:
            print(f"    - {p['bibcode']}")
        print()

        print("[STEP] Preparing PDF + Summary pipeline...")
        ensure_summaries_for_papers(
            papers,
            PDF_DIR,
            SUMMARY_DIR,
            model=cfg.summary_model,
            provider=provider,
            summary_chunk_template_path=summary_chunk_template_path,
            summary_merge_template_path=summary_merge_template_path,
            max_chunks_per_paper=cfg.max_chunks_per_paper,
            chunk_size_chars=cfg.chunk_size_chars,
            safety_max_chunks_per_paper=cfg.safety_max_chunks_per_paper,
            summary_chunk_temperature=cfg.summary_temperature,
            summary_chunk_max_tokens=cfg.summary_chunk_max_tokens,
            summary_merge_temperature=cfg.summary_merge_temperature,
            summary_merge_max_tokens=cfg.summary_merge_max_tokens,
        )
        print("PDFs + summaries ready\n")
        summaries_block = build_summaries_block(papers, SUMMARY_DIR, start_index=2)

        print("[STEP] Building article prompt...")
        sources_str = "\n".join(f"[{i+2}] {paper['bibcode']}" for i, paper in enumerate(papers))
        print("Prompt built\n")

        prompt = alias_note + render_template(
            article_template,
            target=target,
            canonical=canonical,
            source_parameters_block=source_parameters_block,
            summaries_block=summaries_block,
            sources_str=sources_str,
        )

        if extra_context_block:
            prompt = prompt + "\n\nADDITIONAL CONTEXT (optional, user-provided files):\n" + extra_context_block

        print("[STEP] Generating final article...")
        generate_article(
            prompt=prompt,
            output_basename=f"{canonical_path}_article",
            model=cfg.article_model,
            temperature=cfg.article_temperature,
            max_tokens=cfg.article_max_tokens,
            out_dir=articles_dir,
            logs_dir=logs_dir,
            provider=provider,
        )
        print(f"Article written → {canonical_path}_article.txt\n")
        print("-" * 72)
        print("Article(s) complete.\n")
