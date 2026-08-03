from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    source_parameters_csv_path: str = "Master_Sample_Source_Parameters.csv"
    scraped_table_dir: str = "Scraped_Table_Data"
    paper_pdfs_root: str = "paper_pdfs"
    summaries_root: str = "summaries"
    articles_dir: str = "wiki_articles"
    logs_dir: str = "logs"
    output_root: str = ""

    score_limit: int = 0
    paper_limit: int = 10

    summary_model: str = "gpt-5.6-luna"
    article_model: str = "gpt-5.6-terra"
    provider: str = "openai"
    summary_temperature: float = 0.2
    summary_chunk_max_tokens: int = 1500
    summary_merge_temperature: float = 0.1
    summary_merge_max_tokens: int = 1000
    article_temperature: float = 0.2
    article_max_tokens: int = 16000
    # 0 means summarize the whole extracted paper. The safety limit prevents
    # unexpectedly large PDFs from creating an unbounded number of API calls.
    max_chunks_per_paper: int = 0
    chunk_size_chars: int = 3500
    safety_max_chunks_per_paper: int = 30
    extra_context_files: list[str] = field(default_factory=list)
    summary_chunk_template: str = "prompts/summary_chunk.txt"
    summary_merge_template: str = "prompts/summary_merge.txt"
    article_template: str = "prompts/article.txt"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config_file(path: str | None) -> RunConfig:
    if not path:
        return RunConfig()

    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")

    data: dict[str, Any] | None = None
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw_text)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = None

    if data is None:
        try:
            loaded = json.loads(raw_text)
            if isinstance(loaded, dict):
                data = loaded
        except Exception as exc:
            raise ValueError(
                f"Could not parse config file {path}. Use YAML/JSON format."
            ) from exc

    return RunConfig.from_dict(data)
