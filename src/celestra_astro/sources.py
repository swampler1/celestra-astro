from __future__ import annotations

import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
from astroquery.simbad import Simbad
from .utils import safe_slug


# Scrapes an HTML table from a URL and returns it as a pandas DataFrame
def _parse_html_table(table) -> pd.DataFrame:
    headers = [header.text.strip() for header in table.find_all("th")]
    rows = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)
    return pd.DataFrame(rows, columns=headers if headers else None)


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
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all tables on the webpage
    tables = soup.find_all("table")

    if not tables:
        print("No tables found on the webpage.")
        return None

    # Select the desired table (default is the first one)
    table = tables[table_index]

    return _parse_html_table(table)


def get_simbad_aliases(name: str):
    """Return (canonical_main_id, aliases_list) for a SIMBAD object."""
    canonical = name

    # Resolve the canonical SIMBAD name independently. Some SIMBAD responses
    # contain no MAIN_ID even though the object-ID query can still succeed.
    try:
        s = Simbad()
        s.add_votable_fields("ids")
        t = s.query_object(name)
        if t is not None and len(t) > 0:
            main_column = next(
                (column for column in t.colnames if column.upper() == "MAIN_ID"),
                None,
            )
            if main_column:
                canonical = str(t[main_column][0]).strip()
    except Exception as exc:
        print(f"[WARN] SIMBAD canonical-name lookup failed for {name!r}: {exc}")

    # Always attempt the alias query, even when canonical lookup returned no
    # rows. This is the important distinction between the two SIMBAD APIs.
    try:
        ids_tbl = Simbad.query_objectids(name)
        if ids_tbl is None or len(ids_tbl) == 0:
            print(f"[WARN] SIMBAD returned no aliases for {name!r} (canonical: {canonical!r}).")
            return canonical, []

        id_column = next(
            (column for column in ids_tbl.colnames if column.upper() == "ID"),
            ids_tbl.colnames[0],
        )
        aliases = [str(value).strip() for value in ids_tbl[id_column] if str(value).strip()]
        return canonical, aliases
    except Exception as exc:
        print(f"[WARN] SIMBAD alias lookup failed for {name!r}: {exc}")
        return canonical, []


def get_disk_lit(disk_name_orig, save_dir="Scraped_Table_Data"):
    """
    Retrieves scored literature information from SIMBAD for a given disk and saves it as a .csv to desired folder
    ---
    Parameters
    disk_name_orig: (str) Plain text name of disk of interest. Any name recognized by SIMBAD should work
                    (i.e. "2MASS J16004452-4155310" should return the same result as "MY Lup")
    (opt) save_dir: (str) Name of folder to which the .csv will be saved, defaults to "Scraped_Table_Data"
    """

    # Create versions of disk name for web-searchability and file saving
    disk_name = disk_name_orig.replace(" ", "%20")
    disk_name_file = safe_slug(disk_name_orig)

    # Dynamically construct the SIMBAD URL without a hardcoded identifier
    baseurl = "https://simbad.cds.unistra.fr/simbad/sim-id-refs?Ident="
    url = baseurl + disk_name

    # Scrape SIMBAD and detect the literature table by headers
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")

    df = None
    required_cols = {"Score", "Bibcode/DOI"}
    for table in tables:
        try:
            candidate = _parse_html_table(table)
        except Exception:
            continue
        if candidate is not None and required_cols.issubset(set(candidate.columns)):
            df = candidate
            break

    # Fallback to legacy index for compatibility
    if df is None:
        try:
            df = scrape_table(url, table_index=2)
        except Exception:
            df = None

    if df is not None:
        # SIMBAD can return mixed-type score values and, depending on the
        # table markup, duplicate column labels. Normalize both before doing
        # numeric filtering so pandas does not attempt an unsafe string cast.
        df = df.loc[:, ~df.columns.duplicated()].copy()
        scores = pd.to_numeric(df["Score"].astype(str).str.strip(), errors="coerce")
        filtered_df = df.loc[scores.notna()].copy()
        # Drop the original string column before adding the numeric one. This
        # avoids pandas' incompatible-dtype assignment error on recent pandas.
        filtered_df = filtered_df.drop(columns=["Score"])
        filtered_df.insert(0, "Score", scores.loc[filtered_df.index].astype(int).to_numpy())
        df_sorted = filtered_df.sort_values(by="Score", ascending=False)
        # print(df_sorted)
        print(f"\nTop SIMBAD literature for {disk_name_orig} (sorted by score):")
        print(df_sorted.head(10))  # show top 10 rows nicely
        os.makedirs(save_dir, exist_ok=True)
        df_sorted.to_csv(save_dir + "/%s_scraped_table.csv" % (disk_name_file), index=False)  # Save to CSV


# Select papers from a SIMBAD literature table using a target-relative score
# threshold. SIMBAD scores are target-dependent, so an absolute score is not a
# useful default across disks.
def select_papers(
    disk_name_orig,
    score_limit=0,
    paper_limit=0,
    scraped_table_dir="Scraped_Table_Data",
    score_threshold_mode="relative",
    score_threshold_fraction=0.28,
    score_stddevs_below_max=1.0,
):
    """Select relevant papers using a target-relative SIMBAD score rule.

    By default, every paper scoring at least ``score_threshold_fraction`` of
    the target's maximum score is selected. ``paper_limit`` is only a hard
    cap when it is positive; zero leaves the dynamic result uncapped.
    """
    # Create version of disk name for file reading
    disk_name_file = safe_slug(disk_name_orig)

    table_path = os.path.join(
        scraped_table_dir,
        disk_name_file + "_scraped_table.csv",
    )
    df_literature = pd.read_csv(table_path, index_col=False)
    if "Score" not in df_literature.columns or "Bibcode/DOI" not in df_literature.columns:
        raise ValueError(f"Literature table {table_path} is missing Score or Bibcode/DOI columns.")

    scores = pd.to_numeric(df_literature["Score"], errors="coerce")
    valid = df_literature.loc[scores.notna()].copy()
    valid["Score"] = scores.loc[valid.index].astype(float)
    valid = valid.sort_values("Score", ascending=False, kind="stable")

    if valid.empty:
        print(f"No scored literature exists for {disk_name_orig}.")
        return []

    max_score = float(valid["Score"].max())
    mode = (score_threshold_mode or "relative").strip().lower()
    if mode == "relative":
        if not 0 <= float(score_threshold_fraction) <= 1:
            raise ValueError("score_threshold_fraction must be between 0 and 1.")
        threshold = max_score * float(score_threshold_fraction)
        criterion = f"at least {float(score_threshold_fraction):.0%} of max score {max_score:.0f}"
    elif mode in {"stddev", "standard_deviation", "standard-deviation"}:
        stddevs = float(score_stddevs_below_max)
        if stddevs < 0:
            raise ValueError("score_stddevs_below_max must be non-negative.")
        score_std = float(valid["Score"].std(ddof=0))
        threshold = max_score - stddevs * score_std
        criterion = f"within {stddevs:g} standard deviation(s) below max score {max_score:.0f}"
    elif mode in {"absolute", "fixed"}:
        threshold = float(score_limit)
        criterion = f"score at least {threshold:g}"
    else:
        raise ValueError(
            "score_threshold_mode must be 'relative', 'stddev', or 'absolute'."
        )

    selected = valid.loc[valid["Score"] >= threshold]
    threshold_count = len(selected)
    if paper_limit and int(paper_limit) > 0:
        selected = selected.head(int(paper_limit))

    papers = []
    authors_column = "First \n3\n Authors"
    for _, row in selected.iterrows():
        authors = row[authors_column] if authors_column in selected.columns else ""
        papers.append({"bibcode": row["Bibcode/DOI"], "authors": authors})

    if not papers:
        print(f"No literature exists for {disk_name_orig} meeting the dynamic score criterion ({criterion}).")
    else:
        cap_note = f"; hard cap {int(paper_limit)}" if paper_limit and int(paper_limit) > 0 else ""
        print(
            f"Selected {len(papers)} of {threshold_count} papers for {disk_name_orig} "
            f"({criterion}{cap_note})."
        )

    return papers
