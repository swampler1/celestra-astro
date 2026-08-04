#!/usr/bin/env python3
"""Thin runner for the source article generation pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from source_article_generator.cli import main


if __name__ == "__main__":
    main()
