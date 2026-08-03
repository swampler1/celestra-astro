###################################################################
#This program will allow ChatGPT to parse a given list of papers
#for a source and write the proporiate wiki article for that source 
#with given promt constraints.

#Writen by Margaret Huan and Cole Wampler /5/30/25
###################################################################
#Global imports
from __future__ import annotations
#Basic python
import numpy as np
import pandas as pd
import os
import re
import textwrap

#Astro database querying and data structure
from astroquery.simbad import Simbad
#from astroquery.ads import ADS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

#SIMBAD scrape and ADS
from bs4 import BeautifulSoup
import ads
import requests

#Logging setup
import datetime
import time
import logging
logging.basicConfig(level=logging.INFO)
os.makedirs("logs", exist_ok=True)

# Generate a timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join("logs", f"article_generation_{timestamp}.log")

#Enable ChatGPT
import os
from dotenv import load_dotenv
load_dotenv()  # Load env vars from .env file
from openai import OpenAI
client = OpenAI()  # Now it can find OPENAI_API_KEY in your env
CHEAP_MODEL = "gpt-3.5-turbo"

#CSV File load
script_dir = os.path.dirname(os.path.abspath(__file__))
SOURCE_PARAMETERS_CSV_PATH = os.path.join(script_dir, "Master_Sample_Source_Parameters.csv")


###################################################################
#safe file names with aliases from SIMBAD
def safe_slug(s: str) -> str:
    """Return a filesystem-safe version of a string."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')

###################################################################
# Code originally developed in ads_downloader.py by Margaret Haun (UVA),
# edited by Cole Wampler, last updated 6/8/25.
###################################################################

# Scrapes an HTML table from a URL and returns it as a pandas DataFrame
def scrape_table(url, table_index=0):
    """
    Finds tables from webpage using given url and scrapes their contents into a DataFrame
    ---
    Parameters
    url: (str) Link to webpage of interest containing at least one table
    (opt) table_index: (int) Selects which table to parse if page contains more than one, defaults to 0 (first one)
    ---
    Returns
    df: (Data Frame) Webpage table contents
    """
    
    # Send a GET request to the webpage
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for failed requests
    
    # Parse the webpage content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all tables on the webpage
    tables = soup.find_all('table')
    
    if not tables:
        print("No tables found on the webpage.")
        return None

    # Select the desired table (default is the first one)
    table = tables[table_index]
    
    # Extract table headers
    headers = [header.text.strip() for header in table.find_all('th')]
    
    # Extract table rows
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all('td')
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)
    
    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers if headers else None)
    
    return df


def get_simbad_aliases(name: str):
    """Return (canonical_main_id, aliases_list) for a SIMBAD object."""
    try:
        s = Simbad()
        s.add_votable_fields('ids')
        t = s.query_object(name)
        if t is None or len(t) == 0:
            return name, []
        canonical = str(t['MAIN_ID'][0])

        ids_tbl = Simbad.query_objectids(name)
        aliases = [str(row['ID']) for row in ids_tbl] if ids_tbl is not None else []
        return canonical, aliases
    except Exception:
        return name, []



def get_disk_lit(disk_name_orig, save_dir = "Scraped_Table_Data"):
    """
    Retrieves scored literature information from SIMBAD for a given disk and saves it as a .csv to desired folder
    ---
    Parameters
    disk_name_orig: (str) Plain text name of disk of interest. Any name recognized by SIMBAD should work 
                    (i.e. "2MASS J16004452-4155310" should return the same result as "MY Lup")
    (opt) save_dir: (str) Name of folder to which the .csv will be saved, defaults to "Scraped_Table_Data"
    """

    # Create versions of disk name for web-searchability and file saving
    disk_name=disk_name_orig.replace(" ", "%20")
    disk_name_file = disk_name_orig.replace(" ", "_")

    # Dynamically construct the SIMBAD URL without a hardcoded identifier
    baseurl = 'https://simbad.cds.unistra.fr/simbad/sim-id-refs?Ident='
    url = baseurl+disk_name

    # Scrape SIMBAD
    df = scrape_table(url,table_index=2)

    if df is not None:
        filtered_df = df[df['Score'] != '']
        filtered_df.loc[:, 'Score'] = filtered_df['Score'].astype(int) # so not treated as strings
        df_sorted = filtered_df.sort_values(by='Score',ascending=False)
        #print(df_sorted)
        print(f"\nTop SIMBAD literature for {disk_name_orig} (sorted by score):")
        print(df_sorted.head(10))  # show top 10 rows nicely
        os.makedirs(save_dir, exist_ok=True)
        df_sorted.to_csv(save_dir+"/%s_scraped_table.csv"%(disk_name_file), index=False)  # Save to CSV

#Selects top papers for a disk from the scraped SIMBAD table based on score and limit
def select_papers(disk_name_orig, score_limit = 0, paper_limit=5):   
    # Create version of disk name for file reading
    disk_name_file = disk_name_orig.replace(" ", "_")

    home_path = os.getcwd() + "/"

    papers = []

    # Read CSV
    df_Literature = pd.read_csv(home_path + "Scraped_Table_Data/" + disk_name_file + r"_scraped_table.csv", index_col=False)

    count = 0
    for i in range(len(df_Literature)):
        if count >= paper_limit:
            break
        if df_Literature['Score'][i] > score_limit:
            bibcode = df_Literature['Bibcode/DOI'][i]
            authors = df_Literature['First \n3\n Authors'][i]  # Use exact column name
            papers.append({"bibcode": bibcode, "authors": authors})
            count += 1

    num_papers = len(papers)
    if num_papers == 0:
        print(f'No literature exists for {disk_name_orig} within given parameters. Consider adjusting your score limit.')
    elif num_papers == 1:
        print(f'Only 1 paper found for {disk_name_orig} with SIMBAD score over {score_limit}')
    elif num_papers < paper_limit:
        print(f'Only {num_papers} papers found for {disk_name_orig} with SIMBAD scores over {score_limit}')
    else:
        print(f'{num_papers} papers found for {disk_name_orig}')

    return papers

###################################################################
#This section loads the info for each source from the given CSV file
###################################################################

#Loading source-parameter table info
#      - Dist_DR3           -> Distance_pc
#      - Mstar_PPVII        -> StellarMass_Msun
#      - vsys_v2            -> SystemicVelocity_kms
#      - PA_deg            -> PA_deg
#      - Inc_deg           -> Inclination_deg
def load_source_parameters_table(csv_path: str) -> dict[str, dict[str, float | str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not load Source-parameter basic-facts table from {csv_path}: {e}")
        return {}

    required_cols = ["Source", "Dist_DR3", "Mstar_PPVII", "vsys_v2", "PA_deg", "Inc_deg"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] source-parameter table is missing required column '{col}'.")
            return {}

    table: dict[str, dict[str, float | str]] = {}

    for _, row in df.iterrows():
        source_raw = row["Source"]
        if pd.isna(source_raw):
            continue
        key = str(source_raw).strip()
        if not key:
            continue

        # Use lowercased key for robust lookup
        key_lower = key.lower()

        table[key_lower] = {
            "Source": key,
            "Distance_pc": None if pd.isna(row["Dist_DR3"]) else float(row["Dist_DR3"]),
            "StellarMass_Msun": None if pd.isna(row["Mstar_PPVII"]) else float(row["Mstar_PPVII"]),
            "SystemicVelocity_kms": None if pd.isna(row["vsys_v2"]) else float(row["vsys_v2"]),
            "PA_deg": None if pd.isna(row["PA_deg"]) else float(row["PA_deg"]),
            "Inclination_deg": None if pd.isna(row["Inc_deg"]) else float(row["Inc_deg"]),
        }

    print(f"[INFO] Loaded Source-parameter basic-facts table with {len(table)} entries.")
    return table

#Looks up source parameters for a target using its exact Source name
def get_source_parameters_for_source_name(source_name: str) -> dict[str, float | str]:
    if not source_name:
        return {}
    key_lower = source_name.strip().lower()
    return SOURCE_PARAMETERS.get(key_lower, {})

#Loads the Source-parameter basic-facts table at startup and reports how many entries were read
SOURCE_PARAMETERS: dict[str, dict[str, float | str]] = load_source_parameters_table(SOURCE_PARAMETERS_CSV_PATH)
if not SOURCE_PARAMETERS:
    print(f"[WARN] SOURCE_PARAMETERS is empty. Check SOURCE_PARAMETERS_CSV_PATH = {SOURCE_PARAMETERS_CSV_PATH}")
else:
    print(f"[INFO] SOURCE_PARAMETERS has {len(SOURCE_PARAMETERS)} entries.")

#Formats source parameters for a target into a plain-text table block
def format_source_parameters_block(source_name: str) -> str:
    row = get_source_parameters_for_source_name(source_name)
    if not row:
        return f"No source parameters found for {source_name} in the source-parameter table."

    def fmt_val(val: float | str | None, fmt: str = "{:.2f}") -> str:
        if val is None:
            return "unknown"
        if isinstance(val, (float, int)):
            try:
                return fmt.format(val)
            except Exception:
                return str(val)
        return str(val)

    lines = []
    lines.append(f"Source (catalog)          : {row['Source']}")
    lines.append(f"Distance (pc, DR3)     : {fmt_val(row['Distance_pc'])}")
    lines.append(f"Stellar mass (M_sun)   : {fmt_val(row['StellarMass_Msun'])}")
    lines.append(f"Systemic velocity (km/s): {fmt_val(row['SystemicVelocity_kms'])}")
    lines.append(f"Disk PA (deg)          : {fmt_val(row['PA_deg'])}")
    lines.append(f"Disk inclination (deg) : {fmt_val(row['Inclination_deg'])}")

    return "\n".join(lines)

###################################################################
#This block works to download the papers coresponding with each BIB code 
###################################################################

#Returns an ADS API token from environment variables (or None if not found)
def get_ads_token() -> str:
    return (
        os.getenv("ADS_API_TOKEN")
        or os.getenv("ADS_TOKEN")
        or os.getenv("ADS_KEY")
    )

#Fetches ADS metadata (title, authors, abstract, etc.) for a given bibcode
def fetch_ads_metadata_for_bibcode(bibcode: str) -> dict:
    token = get_ads_token()
    if not token:
        print("[WARN] No ADS API token found (ADS_API_TOKEN / ADS_TOKEN / ADS_KEY). "
              "Skipping ADS metadata lookup.")
        return {}

    url = "https://api.adsabs.harvard.edu/v1/search/query"
    params = {
        "q": f"bibcode:{bibcode}",
        "fl": "id,identifier,doi,arxiv_id",
        "rows": 1,
    }
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] ADS query failed for {bibcode}: {e}")
        return {}

    docs = data.get("response", {}).get("docs", [])
    if not docs:
        print(f"[WARN] ADS returned no docs for bibcode {bibcode}")
        return {}

    doc = docs[0]
    return doc

#Extracts an arXiv ID from an ADS metadata document if one is present
def extract_arxiv_id_from_ads_doc(doc: dict) -> str:
    # ADS sometimes uses 'arxiv_id' directly
    arxiv_id = doc.get("arxiv_id")
    if arxiv_id:
        return arxiv_id

    # or packs them into 'identifier'
    identifiers = doc.get("identifier", []) or []
    for ident in identifiers:
        if isinstance(ident, str) and ident.lower().startswith("arxiv:"):
            return ident.split(":", 1)[1].strip()
    return ""

#Extracts a DOI from an ADS metadata document if one is present
def extract_doi_from_ads_doc(doc: dict) -> str:
    doi = doc.get("doi")
    if not doi:
        return ""
    if isinstance(doi, list):
        return doi[0]
    return str(doi)

#Downloads the PDF for a given bibcode into the specified directory
#Strategy:Query ADS for identifiers. If arXiv id is present, download from arxiv.org. If DOI present, try doi.org as a last resort.
def download_pdf_for_bibcode(bibcode: str, pdf_dir: str) -> bool:
    doc = fetch_ads_metadata_for_bibcode(bibcode)
    if not doc:
        return False

    filename = safe_slug(bibcode) + ".pdf"
    dest_path = os.path.join(pdf_dir, filename)

    # 1) Try arXiv
    arxiv_id = extract_arxiv_id_from_ads_doc(doc)
    if arxiv_id:
        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"[INFO] Attempting arXiv PDF for {bibcode}: {arxiv_url}")
        try:
            r = requests.get(arxiv_url, timeout=30)
            if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                print(f"[INFO] Downloaded arXiv PDF for {bibcode} → {dest_path}")
                return True
            else:
                print(f"[WARN] arXiv PDF request for {bibcode} returned status {r.status_code} "
                      f"and content-type {r.headers.get('content-type')}")
        except Exception as e:
            print(f"[ERROR] Failed to download arXiv PDF for {bibcode}: {e}")

    # 2) Try DOI (this may or may not work depending on access / publisher)
    doi = extract_doi_from_ads_doc(doc)
    if doi:
        doi_url = f"https://doi.org/{doi}"
        print(f"[INFO] Attempting DOI-based PDF fetch for {bibcode}: {doi_url}")
        try:
            r = requests.get(doi_url, timeout=30, allow_redirects=True)
            # Some publishers serve PDF directly, some HTML. We'll only accept real PDF.
            if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                print(f"[INFO] Downloaded DOI PDF for {bibcode} → {dest_path}")
                return True
            else:
                print(f"[WARN] DOI fetch for {bibcode} yielded status {r.status_code} "
                      f"and content-type {r.headers.get('content-type')}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch DOI PDF for {bibcode}: {e}")

    print(f"[WARN] Could not obtain PDF for bibcode {bibcode} via ADS/arXiv/DOI.")
    return False

###################################################################
#This block downloads the PDF's of the papers from the SIMBAD bibcodes
###################################################################

#Obtains path and file name for PDF
def get_pdf_path_for_bibcode(bibcode: str, pdf_dir: str) -> str:
    filename = safe_slug(bibcode) + ".pdf"
    path = os.path.join(pdf_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found for {bibcode}. Expected at: {path}. "
            "Place the PDF there or implement automatic download."
        )
    return path

#Extracts text of the PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("Please install pymupdf: pip install pymupdf")

    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    full_text = "\n".join(texts)
    # Light cleanup
    full_text = "\n".join(line.strip() for line in full_text.splitlines() if line.strip())
    return full_text


#Splits text of PDF,s into chunks to be read by chatGPT
def chunk_text(text: str, max_chars: int = 3500) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end
    return chunks

# Generates a technical summary of a paper's text using the selected LLM.
# Takes the raw extracted PDF text and produces a concise, domain-relevant summary focusing on observational methods, key measurements, and results.
def summarize_paper_text(bibcode: str, raw_text: str, model: str = CHEAP_MODEL) -> str:
    chunks = chunk_text(raw_text, max_chars=3500)
    max_chunks = 7  # cap for cost control
    if len(chunks) > max_chunks:
        print(f"[INFO] {bibcode}: {len(chunks)} chunks → truncating to {max_chunks}.")
        chunks = chunks[:max_chunks]

    chunk_summaries: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Summarizing {bibcode} chunk {i}/{len(chunks)}...")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert astrophysicist. You write concise but highly technical summaries "
                        "of scientific texts. You do not invent facts or numbers; you only use information "
                        "explicitly present in the text you are given."
                    ),
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(f"""
                    Paper bibcode: {bibcode}
                    Chunk {i} of {len(chunks)}.

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
                    {chunk}
                    """).strip(),
                },
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        summary = resp.choices[0].message.content.strip()
        chunk_summaries.append(summary)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    joined = "\n\n".join(chunk_summaries)
    print(f"[INFO] Combining {len(chunk_summaries)} chunk summaries for {bibcode}...")
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert astrophysicist. You combine section-level summaries of a paper into a single, "
                    "coherent, technical summary. You do not invent facts; you only use what is present in the input summaries."
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(f"""
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
                {joined}
                """).strip(),
            },
        ],
        temperature=0.1,
        max_tokens=1000,
    )
    return resp2.choices[0].message.content.strip()

#Builds the filesystem path where the summary for a given bibcode should be saved
def summary_path_for_bibcode(bibcode: str, summary_dir: str) -> str:
    return os.path.join(summary_dir, safe_slug(bibcode) + "_summary.txt")

#Ensures every paper has a generated summary by downloading PDFs and creating summaries if missing
def ensure_summaries_for_papers(papers, pdf_dir, summary_dir) -> None:
    for p in papers:
        bib = (
            p.get("bibcode")
            or p.get("Bibcode")
            or p.get("Bibcode/DOI")
        )
        if not bib:
            print("[WARN] Paper entry missing 'bibcode', skipping:", p)
            continue

        spath = summary_path_for_bibcode(bib, summary_dir)
        if os.path.exists(spath):
            print(f"[INFO] Summary already exists for {bib}: {spath}")
            continue

        # Make sure we have a PDF locally; if not, try to download it
        pdf_filename = safe_slug(bib) + ".pdf"
        pdf_path = os.path.join(pdf_dir, pdf_filename)

        if not os.path.exists(pdf_path):
            print(f"[INFO] No local PDF found for {bib} in {pdf_dir}. Attempting download...")
            ok = download_pdf_for_bibcode(bib, pdf_dir)
            if not ok:
                print(f"[ERROR] Could not obtain PDF for {bib}; skipping summary.")
                continue
            # pdf_path should now exist if download succeeded

        try:
            print(f"[INFO] Creating summary for {bib}...")
            raw_text = extract_text_from_pdf(pdf_path)
            summary_text = summarize_paper_text(bib, raw_text)
            with open(spath, "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"[INFO] Summary written: {spath}")
        except Exception as e:
            print(f"[ERROR] Failed to summarize {bib}: {e}")


#Builds the full text block of all paper summaries to inject into the article prompt
def build_summaries_block(papers: list[dict], summary_dir: str) -> str:
    blocks = []
    for i, p in enumerate(papers, start=1):
        bib = (
            p.get("bibcode")
            or p.get("Bibcode")
            or p.get("Bibcode/DOI")
            or f"UNKNOWN_{i}"
        )
        spath = summary_path_for_bibcode(bib, summary_dir)
        if not os.path.exists(spath):
            blocks.append(f"[{i}] {bib}\n(No summary available.)")
            continue
        with open(spath, "r", encoding="utf-8") as f:
            stext = f.read().strip()
        blocks.append(f"[{i}] {bib}\n{stext}")
    return "\n\n".join(blocks)


###################################################################
#This block generates the plain text article from the given bib codes.
###################################################################
def generate_article(
    prompt: str,
    output_basename: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 5000,
    out_dir: str = "wiki_articles",
) -> str:
    """
    Generate a PLAIN TEXT wiki-style article and write it to <out_dir>/<output_basename>.txt.
    Returns the file path to the written article.

    Notes:
    - No markdown is produced; the model is instructed to output plain text only.
    - A run log with the prompt and a short response preview is saved under ./logs.
    - This function assumes a global `client` is available and configured (OpenAI).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join("logs", f"logs_for_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    log_path = os.path.join(run_log_dir, f"{output_basename}_plaintext_generation_{timestamp}.log")

    system_message = (
        "You are a scientific writer producing a detailed wiki article for experts in PLAIN TEXT only. "
        "Do not use markdown, bullets, or code blocks. Use simple section titles on their own lines and "
        "tight, well-structured paragraphs. Prioritize methods, measurements, uncertainties, and concrete "
        "results over generalities. Use inline numeric citation markers like [n] that map to a provided source list."
    )

    # Call the model (expects a global `client` already created elsewhere in this file)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    article_txt = response.choices[0].message.content
    usage = getattr(response, "usage", None)

    # Write the article
    out_path = os.path.join(out_dir, f"{output_basename}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(article_txt)

    # Minimal run log
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"=== Plaintext Article for: {output_basename} ===\n\n")
        log.write("[System Message]\n")
        log.write(system_message + "\n\n")
        log.write("[Prompt]\n")
        log.write(prompt + "\n\n")
        log.write("[Response Preview]\n")
        log.write((article_txt[:1000] if article_txt else "") + ("...\n\n" if article_txt and len(article_txt) > 1000 else "\n\n"))
        if usage:
            log.write("[Token Usage]\n")
            log.write(f"  prompt_tokens: {usage.prompt_tokens}\n")
            log.write(f"  completion_tokens: {usage.completion_tokens}\n")
            log.write(f"  total_tokens: {usage.total_tokens}\n")
        log.write("=" * 60 + "\n\n")

    logging.info("Plain-text article written to %s", out_path)
    return out_path

###################################################################
#Main call to generate articles, from a list of targets.
###################################################################

# targets_list = ["AA Tau", "DG Tau", "TW Hya", "HL Tau"]
targets_list = ["V1094Sco","DL Tau","GM Aur"]

for target in targets_list:
    # A) Resolve canonical name + aliases
    canonical, aliases = get_simbad_aliases(target)

    # ---------- HEADER ----------
    print("\n" + "="*72)
    print(f"Target: {target}  (canonical: {canonical})")
    print("="*72 + "\n")

    # ---------- ALIASES ----------
    print(f"Aliases ({len(aliases)} total):")
    for a in aliases[:10]:
        print(f"  • {a}")
    if len(aliases) > 10:
        print(f"  … ({len(aliases) - 10} more not shown)")
    print()

    # Alias note that will be prepended to the prompt
    alias_note = (
        f"TARGET (canonical): {canonical}\n"
        f"Treat the following as exact aliases for the same object: {', '.join(aliases)}\n\n"
        if aliases else
        f"TARGET (canonical): {canonical}\n\n"
    )

    # Use a safe slug for any filenames / directories
    canonical_path = safe_slug(canonical)

    # ===== Source-parameter BASIC FACTS LOOKUP =====
    source_parameters_block = format_source_parameters_block(target)
    print("source parameters:")
    print(source_parameters_block)
    print()

    # --- Top-level base directories ---
    print("Directory structure:")
    BASE_PDF_ROOT = "paper_pdfs"
    BASE_SUMMARY_ROOT = "summaries"
    os.makedirs(BASE_PDF_ROOT, exist_ok=True)
    os.makedirs(BASE_SUMMARY_ROOT, exist_ok=True)

    PDF_DIR = os.path.join(BASE_PDF_ROOT, f"{canonical_path}_papers_pdf")
    SUMMARY_DIR = os.path.join(BASE_SUMMARY_ROOT, f"{canonical_path}_paper_summaries")
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    print(f"  • PDF Directory:      {PDF_DIR}")
    print(f"  • Summary Directory:  {SUMMARY_DIR}\n")

    # ---------- STEP 1: CSV SCRAPE ----------
    csv_path = f"Scraped_Table_Data/{canonical_path}_scraped_table.csv"
    if not os.path.exists(csv_path):
        try:
            print(f"[INFO] No CSV found → scraping SIMBAD for '{canonical}' ...")
            get_disk_lit(canonical)
        except Exception as e:
            print(f"[ERROR] Error scraping SIMBAD for {target}: {e}")
            continue
    else:
        print(f"[INFO] Using existing CSV: {csv_path}")

    # ---------- STEP 2: SELECT PAPERS ----------
    print("[STEP] Selecting top papers...")    
    papers = select_papers(canonical, score_limit=0, paper_limit=10)

    if not papers:
        print("No valid papers found.\n")
        continue

    print(f"Selected {len(papers)} paper(s):")
    for p in papers:
        print(f"    - {p['bibcode']}")
    print()

    # ---------- STEP 3: ENSURE SUMMARIES ----------
    print("[STEP] Preparing PDF + Summary pipeline...")
    ensure_summaries_for_papers(papers, PDF_DIR, SUMMARY_DIR)
    print("PDFs + summaries ready\n")
    summaries_block = build_summaries_block(papers, SUMMARY_DIR)

    # Simple list of sources by index (1-based)
    sources_str = "\n".join(f"[{i+1}] {paper['bibcode']}" for i, paper in enumerate(papers))

    # Step 4: Build the article-generation prompt
    print("[STEP] Building article prompt...")
    sources_str = "\n".join(f"[{i+1}] {paper['bibcode']}" for i, paper in enumerate(papers))
    print("Prompt built\n")

    prompt = alias_note + f"""
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
  index of the paper, e.g. [1], [2], [3].
- If a sentence combines information from multiple papers, include all relevant indices,
  e.g. [1, 3, 5].
- If a summary block says "(No summary available.)", ignore that source and do not cite it.
- If the summaries do not provide a detail (e.g., a specific mass or radius), state that the
  value is not well constrained instead of inventing a number.
- Avoid speculation or hallucination; stay within what can be reasonably inferred from the
  provided summaries.

In the References section, list each source as:
[n] FirstAuthor et al., Year, Title (or description if unknown), Journal (if known), Bibcode

The first reference should be the Source-parameter Table in the style "[1] source-parameter data table"
""".strip()


    # ---------- STEP 5: GENERATE ARTICLE ----------
    print("[STEP] Generating final article...")
    generate_article(
        prompt=prompt,
        output_basename=f"{canonical_path}_article"
    )
    print(f"Article written → {canonical_path}_article.txt\n")
    print("-"*72)
    print("Article(s) complete.\n")   



















