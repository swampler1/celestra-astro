from __future__ import annotations

import os
import requests

from .utils import safe_slug


# Returns an ADS API token from environment variables (or None if not found)
def get_ads_token() -> str:
    return (
        os.getenv("ADS_API_TOKEN")
        or os.getenv("ADS_TOKEN")
        or os.getenv("ADS_KEY")
    )


# Fetches ADS metadata (title, authors, abstract, etc.) for a given bibcode
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


# Extracts an arXiv ID from an ADS metadata document if one is present
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


# Extracts a DOI from an ADS metadata document if one is present
def extract_doi_from_ads_doc(doc: dict) -> str:
    doi = doc.get("doi")
    if not doi:
        return ""
    if isinstance(doi, list):
        return doi[0]
    return str(doi)


# Downloads the PDF for a given bibcode into the specified directory
# Strategy:Query ADS for identifiers. If arXiv id is present, download from arxiv.org. If DOI present, try doi.org as a last resort.
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


# Obtains path and file name for PDF
def get_pdf_path_for_bibcode(bibcode: str, pdf_dir: str) -> str:
    filename = safe_slug(bibcode) + ".pdf"
    path = os.path.join(pdf_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found for {bibcode}. Expected at: {path}. "
            "Place the PDF there or implement automatic download."
        )
    return path


# Extracts text of the PDF
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
