#!/usr/bin/env python3
"""Download and inspect ADS papers for astronomical sources.

This is the Python-script version of the former ``ads_downloader.ipynb``
notebook.  The original notebook mixed reusable functions with exploratory
cells; this version keeps the same core logic while making the operations
explicit and portable across operating systems.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ads
import pandas as pd
import pdfplumber
import requests
import tiktoken
from bs4 import BeautifulSoup
from google import genai


AI_MODEL = "gemini-2.0-flash"


def configure_ads(token: str | None = None) -> None:
    """Configure ADS from an argument or the ``ADS_API_TOKEN`` environment variable."""
    token = token or os.getenv("ADS_API_TOKEN")
    if token:
        ads.config.token = token


def get_gemini_client(api_key: str | None = None):
    """Create a Gemini client from an argument or ``GEMINI_API_KEY``."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or pass --gemini-api-key.")
    return genai.Client(api_key=api_key)


def scrape_table(url: str, table_index: int = 0) -> pd.DataFrame | None:
    """Scrape one HTML table into a DataFrame."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tables = BeautifulSoup(response.text, "html.parser").find_all("table")
    if not tables:
        print("No tables found on the webpage.")
        return None

    table = tables[table_index]
    headers = [header.get_text(strip=True) for header in table.find_all("th")]
    rows = [
        [cell.get_text(strip=True) for cell in row.find_all("td")]
        for row in table.find_all("tr")[1:]
    ]
    return pd.DataFrame(rows, columns=headers if headers else None)


def get_disk_lit(disk_name: str, save_dir: str = "Scraped_Table_Data") -> Path | None:
    """Retrieve SIMBAD literature scores and save them as a CSV."""
    url = "https://simbad.cds.unistra.fr/simbad/sim-id-refs?Ident=" + disk_name.replace(" ", "%20")
    table = scrape_table(url, table_index=2)
    if table is None or "Score" not in table:
        return None

    scored = table.loc[table["Score"].ne("")].copy()
    scored["Score"] = scored["Score"].astype(int)
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{disk_name.replace(' ', '_')}_scraped_table.csv"
    scored.sort_values("Score", ascending=False).to_csv(output_path, index=False)
    return output_path


def select_papers(
    disk_name: str,
    score_limit: int = 0,
    paper_limit: int = 5,
    table_dir: str = "Scraped_Table_Data",
) -> list[str]:
    """Select the highest-scoring ADS bibcodes from a saved SIMBAD table."""
    table_path = Path(table_dir) / f"{disk_name.replace(' ', '_')}_scraped_table.csv"
    literature = pd.read_csv(table_path)
    selected = literature.loc[literature["Score"].gt(score_limit)].head(paper_limit)
    bibcodes = selected["Bibcode/DOI"].tolist()

    if not bibcodes:
        print(f"No literature exists for {disk_name} within the selected parameters.")
    else:
        print(f"{len(bibcodes)} bibcode(s) found for {disk_name}.")
    return bibcodes


def download_pdfs(bibcodes: list[str], save_dir: str) -> None:
    """Download available ADS or publisher PDFs for the supplied bibcodes."""
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bibcode in bibcodes:
        print(f"Checking bibcode: {bibcode}")
        papers = list(ads.SearchQuery(bibcode=bibcode, fl=["bibcode", "title", "esources"]))
        if not papers:
            print(f"No results found for {bibcode}")
            continue

        esources = papers[0].esources
        if "EPRINT_PDF" in esources:
            kind = "EPRINT_PDF"
        elif "PUB_PDF" in esources:
            kind = "PUB_PDF"
        else:
            print(f"No PDF available for {bibcode}")
            continue

        url = f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/{kind}"
        response = requests.get(url, allow_redirects=True, timeout=120)
        if response.status_code == 200:
            path = output_dir / f"{bibcode}.pdf"
            path.write_bytes(response.content)
            print(f"Saved {path}")
        else:
            print(f"Failed to download {bibcode}, status code: {response.status_code}")


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all page text from a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def chunk_text(text: str, max_tokens: int = 1500, model: str = "gpt-4") -> list[str]:
    """Break PDF text into token-limited chunks."""
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return [encoder.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def gemini_api_call(client, prompt: str, paper_text: str, model: str = AI_MODEL) -> str:
    """Generate a Gemini response for a prompt and paper text."""
    response = client.models.generate_content(model=model, contents=f"{prompt}\n\n{paper_text}")
    return response.text


def collect_region(
    source_names_csv: str,
    region: str,
    table_dir: str = "Scraped_Table_Data",
    pdf_root: str = "ADS_Papers",
    paper_limit: int = 5,
) -> None:
    """Scrape literature and download papers for every source in a region."""
    sources = pd.read_csv(source_names_csv)
    region_sources = sources.loc[sources["Region"].eq(region), "Source"]
    for source in region_sources:
        get_disk_lit(source, table_dir)
        save_dir = Path(pdf_root) / f"{region}_Papers" / source.replace(" ", "_")
        download_pdfs(select_papers(source, paper_limit=paper_limit, table_dir=table_dir), str(save_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-names", default="Source_Names.csv")
    parser.add_argument("--region", help="Scrape and download every source in this region")
    parser.add_argument("--table-dir", default="Scraped_Table_Data")
    parser.add_argument("--pdf-root", default="ADS_Papers")
    parser.add_argument("--paper-limit", type=int, default=5)
    parser.add_argument("--ads-api-token")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_ads(args.ads_api_token)
    if not args.region:
        print("No region selected. Use --region to collect literature and download PDFs.")
        return
    collect_region(
        args.source_names,
        args.region,
        table_dir=args.table_dir,
        pdf_root=args.pdf_root,
        paper_limit=args.paper_limit,
    )


if __name__ == "__main__":
    main()
