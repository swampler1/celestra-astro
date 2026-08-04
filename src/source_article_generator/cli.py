from __future__ import annotations

import argparse
from dataclasses import replace

from .config import RunConfig, load_config_file
from .providers.factory import default_model_for
from .pipeline import run_targets


def _optional_int(value):
    return None if value is None else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate source articles for one or more astronomical targets.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  PYTHONPATH=src python -m source_article_generator --config configs/default.yaml --targets-file targets.txt\n"
            "  PYTHONPATH=src python -m source_article_generator --provider gemini \"TW Hya\"\n"
            "  PYTHONPATH=src python -m source_article_generator --provider openai --model gpt-5.6-terra --paper-limit 12 \"DL Tau\"\n"
        ),
    )

    parser.add_argument("targets", nargs="*", help="Target names (e.g. \"DL Tau\" \"GM Aur\")")
    parser.add_argument("--targets-file", default="", help="Path to newline-delimited targets file")
    parser.add_argument("--config", default="", help="Path to YAML/JSON config file")
    parser.add_argument("--project-root", default="", help="Base directory for relative paths")

    path_group = parser.add_argument_group("Path Overrides")
    path_group.add_argument("--output-root", default=None, help="Override output root for this run")
    path_group.add_argument(
        "--source-parameters-csv-path",
        default=None,
        help="Override source-parameter CSV path",
    )
    path_group.add_argument(
        "--extra-context-file",
        action="append",
        default=None,
        help="Additional context file to inject into prompts (repeatable)",
    )

    provider_group = parser.add_argument_group("Provider and Model Overrides")
    provider_group.add_argument(
        "--provider",
        default=None,
        choices=["openai", "gemini", "claude"],
        help="LLM provider",
    )
    provider_group.add_argument("--model", default=None, help="Override both summary and article model")
    provider_group.add_argument("--summary-model", default=None, help="Override summary model")
    provider_group.add_argument("--article-model", default=None, help="Override article model")

    generation_group = parser.add_argument_group("Generation Overrides")
    generation_group.add_argument(
        "--paper-limit",
        default=None,
        help="Hard cap on papers per target; 0 means dynamic selection with no cap",
    )
    generation_group.add_argument(
        "--score-limit",
        default=None,
        help="Absolute SIMBAD score threshold; used with --score-threshold-mode absolute",
    )
    generation_group.add_argument(
        "--score-threshold-mode",
        default=None,
        choices=["relative", "stddev", "absolute"],
        help="Dynamic score rule: relative max fraction (default), standard deviation, or absolute",
    )
    generation_group.add_argument(
        "--score-threshold-fraction",
        default=None,
        help="Relative mode: minimum fraction of each target's maximum score (default 0.28)",
    )
    generation_group.add_argument(
        "--score-stddevs-below-max",
        default=None,
        help="Stddev mode: number of score standard deviations below the target maximum",
    )
    generation_group.add_argument(
        "--max-chunks-per-paper",
        default=None,
        help="Override max chunks per paper; 0 means process the whole paper",
    )
    generation_group.add_argument(
        "--chunk-size-chars",
        default=None,
        help="Override approximate characters per chunk",
    )
    generation_group.add_argument("--summary-temperature", default=None, help="Override summary temperature")
    generation_group.add_argument("--article-temperature", default=None, help="Override article temperature")
    generation_group.add_argument("--summary-max-tokens", default=None, help="Override summary chunk max tokens")
    generation_group.add_argument("--summary-merge-max-tokens", default=None, help="Override summary merge max tokens")
    generation_group.add_argument("--article-max-tokens", default=None, help="Override article max tokens")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config_file(args.config or None)
    cfg = replace(
        cfg,
        output_root=cfg.output_root if args.output_root is None else args.output_root,
        source_parameters_csv_path=cfg.source_parameters_csv_path if args.source_parameters_csv_path is None else args.source_parameters_csv_path,
        paper_limit=cfg.paper_limit if args.paper_limit is None else _optional_int(args.paper_limit),
        score_limit=cfg.score_limit if args.score_limit is None else _optional_int(args.score_limit),
        score_threshold_mode=(
            cfg.score_threshold_mode
            if args.score_threshold_mode is None
            else args.score_threshold_mode
        ),
        score_threshold_fraction=(
            cfg.score_threshold_fraction
            if args.score_threshold_fraction is None
            else float(args.score_threshold_fraction)
        ),
        score_stddevs_below_max=(
            cfg.score_stddevs_below_max
            if args.score_stddevs_below_max is None
            else float(args.score_stddevs_below_max)
        ),
        summary_model=cfg.summary_model if args.summary_model is None else args.summary_model,
        article_model=cfg.article_model if args.article_model is None else args.article_model,
        provider=cfg.provider if args.provider is None else args.provider,
        summary_temperature=cfg.summary_temperature if args.summary_temperature is None else float(args.summary_temperature),
        article_temperature=cfg.article_temperature if args.article_temperature is None else float(args.article_temperature),
        summary_chunk_max_tokens=cfg.summary_chunk_max_tokens if args.summary_max_tokens is None else _optional_int(args.summary_max_tokens),
        summary_merge_max_tokens=cfg.summary_merge_max_tokens if args.summary_merge_max_tokens is None else _optional_int(args.summary_merge_max_tokens),
        article_max_tokens=cfg.article_max_tokens if args.article_max_tokens is None else _optional_int(args.article_max_tokens),
        max_chunks_per_paper=cfg.max_chunks_per_paper if args.max_chunks_per_paper is None else _optional_int(args.max_chunks_per_paper),
        chunk_size_chars=cfg.chunk_size_chars if args.chunk_size_chars is None else _optional_int(args.chunk_size_chars),
        extra_context_files=(
            cfg.extra_context_files
            if args.extra_context_file is None
            else list(cfg.extra_context_files) + list(args.extra_context_file)
        ),
    )

    if args.provider is not None:
        if args.summary_model is None and args.model is None:
            cfg = replace(cfg, summary_model=default_model_for(cfg.provider, "summary"))
        if args.article_model is None and args.model is None:
            cfg = replace(cfg, article_model=default_model_for(cfg.provider, "article"))

    if args.model is not None:
        cfg = replace(cfg, summary_model=args.model, article_model=args.model)

    targets = list(args.targets)
    if args.targets_file:
        with open(args.targets_file, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t and not t.startswith("#"):
                    targets.append(t)

    if not targets:
        raise SystemExit("No targets provided. Use positional targets and/or --targets-file.")

    run_targets(targets, config=cfg, project_root=(args.project_root or None))


if __name__ == "__main__":
    main()
