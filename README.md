# CELESTRA

## What This Repo Does
This repository builds citation-backed wiki-style source summaries for astronomical targets by combining structured catalog values, literature sources, and LLM-generated paper summaries. User-provided catalog or observational values can be incorporated into the final article alongside literature-derived context. Given a source name, the pipeline queries SIMBAD to identify aliases and locate a user-defined number of relevant papers. It then uses Harvard ADS to retrieve PDFs, summarizes each paper in configurable chunks, and synthesizes those summaries and any user provided values into a final article with footnote-style citations. 

## Current Recommended Workflow

The following is the general recomended workflow to generate articles.

Step 1) Pick the targets.
- The user may either add targets into targets.txt or list them on the command line.

Step 2) Pick the configuation
- The config controls provider, models, path layout, paper limits, chunk limits, token budgets, and prompt template paths.

Step 3) Run the package CLI.
- The CLI loads the config, applies any overide values from the command line, and calls run_targets() in the pipeline module.

Step 4) The pipeline checks what can be reused in the local area.
- Existing files such as scrabed SIMBAD CSV's, PDF files, generated paper sumaries, are not overwriten if already existing locally to save tokens and time.
- Article files are overwriten if a new run writes to the same path.

Step 5) Review outputs.
- Sumaries will end up in summareis_root/<target>_paper_summaries/
- Articles end up under articles_dir/
- Logs end up under logs_dir/

## Installation
To isntall the repository the user should follow this workflow.

'''
git clone https://github.com/swampler1/Source-Article-Generator
cd Source-Article-Generator
python -m venv .<your_venv_name>
source .venv/bin/activate
pip install -r requirements.txt
'''

## API Keys Needed
A few API keys are required to run this tool they are:
Harvard ADS: https://ui.adsabs.harvard.edu/help/api/

and one of the three supported LLM API's:

Open AI ChatGPT: https://openai.com/api/
Google Gemini: https://ai.google.dev/gemini-api/docs
Athropic Claude: https://platform.claude.com/docs/en/api/overview


## Quickstart
A basic version of the command to run the tool:
```bash
PYTHONPATH=src python -m source_article_generator --config configs/default.yaml --targets-file targets.txt
```
Where: 
-PYTHONPATH=src tells Python to look in the src/ directory so it can import the source_article_generator package.
-python -m source_article_generator runs the source_article_generator module as a command-line program.
--config is the path to the config file
--targets-file is the path to the targets.txt file/


## Main Inputs You Can Change

he main user-facing inputs you can change are:

Target list
The objects the pipeline will generate articles for, either passed directly on the command line or through a targets.txt file.

Config file
The main control point for a run. This sets models, providers, limits, prompt file paths, and output locations.

CSV path
The path to the master table containing fixed global source parameters such as distance, stellar mass, and disk inclination.

Prompt templates
The text prompt files used for summary generation and final article generation. Changing these is the main way to alter article style and behavior.

Provider / model choice
Which LLM backend to use (openai, gemini, or claude) and which summary/article models to run for that provider.

Paper limit / chunk limit
How many papers to include per target, and how many PDF text chunks to summarize for each paper.

Output directories
Where the pipeline writes scraped tables, PDFs, summaries, articles, and logs.

Extra context files
Optional additional text files that can be injected into the article prompt to provide extra context for a run.


## Config File Example
Most user customization happens through the config file rather than by editing Python code. The config controls where inputs are read from, which models are used, how many papers and chunks are processed, which prompt templates are loaded, and where outputs are written.

A simple example looks like this:

#Code-----------------------------------------------------
source_parameters_csv_path: Master_Sample_Source_Parameters.csv
scraped_table_dir: Scraped_Table_Data
paper_pdfs_root: paper_pdfs
summaries_root: summaries
articles_dir: wiki_articles
logs_dir: logs

paper_limit: 0
score_threshold_mode: relative
score_threshold_fraction: 0.28
score_stddevs_below_max: 1.0
max_chunks_per_paper: 0
chunk_size_chars: 3500

provider: openai
summary_model: gpt-5.6-luna
article_model: gpt-5.6-terra

article_template: prompts/article.txt
summary_chunk_template: prompts/summary_chunk.txt
summary_merge_template: prompts/summary_merge.txt
#--------------------------------------------------------

In this example:
- source_parameters_csv_path points to the master source-parameter table.
- scraped_table_dir, paper_pdfs_root, summaries_root, articles_dir, and logs_dir control where intermediate and final files are stored.
- paper_limit is an optional hard cap on papers per target. A value of 0 enables dynamic selection with no count cap.
- score_threshold_mode controls dynamic paper selection. The default relative mode compares each paper with that target's maximum SIMBAD score; stddev mode uses a number of standard deviations below that maximum; absolute mode uses score_limit for legacy/manual runs.
- score_threshold_fraction is the minimum fraction of the target's maximum SIMBAD score. The default 0.28 selects papers scoring at least 28% of that target's maximum.
- score_stddevs_below_max controls the stddev mode.
- max_chunks_per_paper sets how much of each paper is summarized. A value of 0 processes the whole paper, subject to the safety limit in the configuration.
- chunk_size_chars controls the approximate size of each text chunk sent to the model.
- provider, summary_model, and article_model choose the LLM backend and models.
- article_template, summary_chunk_template, and summary_merge_template point to the prompt files used during generation.

## Custom Prompt Files
Most of the writing behavior of the pipeline is controlled by prompt template files rather than hardcoded text in the Python code. This makes it easy to experiment with article style, summary style, section structure, and emphasis without modifying the implementation.

The main prompt files are:

article_template
Controls the final article-generation prompt, including article style, structure, depth, citation behavior, and output formatting.

summary_chunk_template
Controls how individual chunks of PDF text are summarized during the first summary pass.

summary_merge_template
Controls how the chunk-level summaries are combined into a single paper-level summary.

To use a custom prompt, create a new text file and point the corresponding config field to it. 

For example:
article_template: prompts/my_custom_article_prompt.txt
summary_chunk_template: prompts/my_custom_summary_chunk.txt
summary_merge_template: prompts/my_custom_summary_merge.txt

This lets users test prompt variants or adapt the pipeline to different writing styles without editing the underlying Python modules.

## Custom CSV files

The path to the master source-parameter table is controlled by source_parameters_csv_path in the config file. This CSV lets you provide a catalog of fixed source values drawn from your own observational dataset or curated table.

These values are meant to act as the authoritative global parameters for each source and are injected directly into the article prompt rather than being inferred from the literature summaries. For example, values such as distance, stellar mass, and disk inclination can be supplied in this file and treated as fixed inputs for article generation.

This allows you to keep a consistent set of source-level parameters across all generated articles, even when the literature contains multiple or conflicting measurements.

## Output Files and Folders

The pipeline writes several kinds of output files during a run:
- scraped literature tables
- downloaded or reused paper PDFs
- paper summary files
- final article text files
- run logs

By default, these are written to the directories listed in the config file. If output_root is set, multiple output folders can be redirected under a single run-specific directory. This is useful for keeping different experiments or prompt tests separate.

## Python Usage

In addition to the command-line workflow, the pipeline can also be run directly from Python. This is the more programmatic option and is mainly useful if you want to control runs from your own scripts or notebooks.

A simple example looks like this:

#Code-----------------------------------------------------
from source_article_generator import RunConfig, run_targets

cfg = RunConfig(
    provider="openai",
    summary_model="gpt-5.6-luna",
    article_model="gpt-5.6-terra",
    paper_limit=0,
    score_threshold_mode="relative",
    score_threshold_fraction=0.28,
    max_chunks_per_paper=0,
    article_template="prompts/article.txt",
)

run_targets(["DG Tau"], config=cfg)
#--------------------------------------------------------

This provides an alternative to the CLI, but for most users the config-file plus command-line workflow will be the easier and more natural way to run the tool.

## Important Gotchas


## Repo Map

The repository is organized around a refactored pipeline in src/, with configuration files in configs/, prompt templates in prompts/, and generated outputs written to data and run directories. Most users will mainly interact with the config files, prompt files, target list, and the main CLI entry point.

### Main Directories

src/
Main package code for the refactored pipeline.

configs/
YAML config files for standard runs and experiments.

prompts/
Prompt templates used for summary generation and final article generation.

Scraped_Table_Data/
Cached literature tables scraped from SIMBAD.

paper_pdfs/
Local paper PDFs organized by target.

summaries/
Cached paper summary files organized by target.

wiki_articles/
Default location for generated article text files.

logs/
Run logs and saved resolved configs.

runs/
Separate run-specific outputs for experiments and prompt comparisons.

### Main Scripts and Files

run_generator.py
Thin wrapper that launches the source article generation pipeline.

wiki_writer.py
Older monolithic script kept mainly for historical reference.

ads_downloader.py
Portable Python script for ADS downloads and PDF inspection, converted from the former exploratory notebook.

make_article_summary_text.py
Helper script for stitching an article together with its summaries into one text file.

test_summarize_pdf.py
Small standalone test script for PDF summarization.

requirements.txt
Python dependency list for the project.

targets.txt
Example newline-delimited list of targets to run.

Master_Sample_Source_Parameters.csv
Example master source-parameter table used by the pipeline.

## Notes on Current State
