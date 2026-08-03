#!/usr/bin/env python3
from pathlib import Path
import sys

ARTICLES_DIR = Path("wiki_articles")
SUMMARIES_DIR = Path("summaries")
OUT_DIR = Path("stitched")

ARTICLE_SUFFIX = "_article.txt"
SUMMARY_SUFFIX = "_summary.txt"


def summaries_for_source(source: str):
    summ_dir = SUMMARIES_DIR / f"{source}_paper_summaries"
    if not summ_dir.exists():
        return []

    summaries = []
    for p in sorted(summ_dir.glob(f"*{SUMMARY_SUFFIX}")):
        bibcode = p.name.replace(SUMMARY_SUFFIX, "")
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        summaries.append((bibcode, text))

    return summaries[:10]  # cap at 10 if desired


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python stitch_wiki.py <SOURCE_NAME>")

    source = sys.argv[1]

    article_path = ARTICLES_DIR / f"{source}{ARTICLE_SUFFIX}"
    if not article_path.exists():
        raise SystemExit(f"Article not found: {article_path}")

    article_text = article_path.read_text(encoding="utf-8", errors="replace").strip()
    bib_summaries = summaries_for_source(source)

    OUT_DIR.mkdir(exist_ok=True)

    out_path = OUT_DIR / f"{source}_stitched.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(source + "\n\n")
        f.write(article_text + "\n\n")

        for bibcode, summary in bib_summaries:
            f.write(bibcode + "\n\n")
            f.write(summary + "\n\n")

    print(f"Wrote {out_path} ({len(bib_summaries)} summaries)")


if __name__ == "__main__":
    main()

