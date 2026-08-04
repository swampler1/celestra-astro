from __future__ import annotations

import pandas as pd

# Importing the package loads the optional .env file. Disable that filesystem
# search for fast, deterministic unit tests.
import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: None

from source_article_generator.sources import select_papers


AUTHORS_COLUMN = "First \n3\n Authors"


def write_literature_table(tmp_path, scores):
    table = pd.DataFrame(
        {
            "Score": scores,
            "Bibcode/DOI": [f"paper-{i}" for i in range(len(scores))],
            AUTHORS_COLUMN: ["Author et al."] * len(scores),
        }
    )
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    table.to_csv(table_dir / "Test_Target_scraped_table.csv", index=False)
    return table_dir


def test_relative_selection_scales_to_target_maximum(tmp_path):
    table_dir = write_literature_table(tmp_path, [100, 80, 49, 10])

    papers = select_papers(
        "Test Target",
        scraped_table_dir=table_dir,
        score_threshold_mode="relative",
        score_threshold_fraction=0.5,
    )

    assert [paper["bibcode"] for paper in papers] == ["paper-0", "paper-1"]


def test_relative_selection_can_use_a_hard_cap(tmp_path):
    table_dir = write_literature_table(tmp_path, [100, 80, 70])

    papers = select_papers(
        "Test Target",
        paper_limit=2,
        scraped_table_dir=table_dir,
        score_threshold_mode="relative",
        score_threshold_fraction=0.5,
    )

    assert len(papers) == 2


def test_standard_deviation_selection_is_supported(tmp_path):
    table_dir = write_literature_table(tmp_path, [100, 90, 10])

    papers = select_papers(
        "Test Target",
        scraped_table_dir=table_dir,
        score_threshold_mode="stddev",
        score_stddevs_below_max=1.0,
    )

    assert [paper["bibcode"] for paper in papers] == ["paper-0", "paper-1"]


def test_absolute_mode_remains_available_as_an_explicit_override(tmp_path):
    table_dir = write_literature_table(tmp_path, [100, 80, 10])

    papers = select_papers(
        "Test Target",
        score_limit=79,
        scraped_table_dir=table_dir,
        score_threshold_mode="absolute",
    )

    assert [paper["bibcode"] for paper in papers] == ["paper-0", "paper-1"]
