from __future__ import annotations

from pathlib import Path

from celestra_astro.config import (
    RunConfig,
    packaged_default_config_path,
    packaged_default_prompt_path,
    packaged_default_targets_path,
)


def test_packaged_defaults_are_available():
    assert Path(packaged_default_config_path()).is_file()
    assert Path(packaged_default_targets_path()).is_file()

    config = RunConfig()
    assert Path(config.article_template).is_file()
    assert Path(config.summary_chunk_template).is_file()
    assert Path(config.summary_merge_template).is_file()
    assert Path(packaged_default_prompt_path("article.txt")).is_file()
