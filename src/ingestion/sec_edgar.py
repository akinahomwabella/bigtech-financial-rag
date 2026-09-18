"""
SEC EDGAR ingestion for the Big Tech Financial Intelligence RAG project.

What this does:
1. For each company, hits the EDGAR "submissions" API to get its full filing
   history (metadata only — form type, date, accession number).
2. Filters down to the most recent N 10-Ks and 10-Qs per company (see
   config/companies.py for limits).
3. Downloads the actual filing document (the primary .htm file) for each
   match and saves it to data/raw/<TICKER>/<ACCESSION>.htm
4. Writes a manifest (data/processed/manifest.jsonl) recording metadata for
   every filing pulled — this is what your ingestion pipeline will read from
   next (chunking, embedding, etc).

SEC fair-access requirements (https://www.sec.gov/os/webmaster-faq#developers):
- You MUST set a descriptive User-Agent with a real contact (name/email or
  company + email). Requests without one get blocked.
- Stay under 10 requests/second. We sleep between requests to be well under
  that and to be a good citizen generally.

IMPORTANT: This script needs outbound internet access to www.sec.gov and
data.sec.gov. It will NOT run inside a network-restricted sandbox — run it
on your own machine (or wherever you're developing this project).

Usage:
    python src/ingestion/sec_edgar.py --user-agent "Your Name your.email@example.com"
"""

import argparse
import json
import time
from pathlib import Path

import requests

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.companies import COMPANIES, TARGET_FORMS, FORM_LIMITS

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REQUEST_DELAY_SECONDS = 0.25  # ~4 req/sec, well under SEC's 10/sec limit


def get_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    })
    return session


def fetch_filing_index(session: requests.Session, cik: str) -> dict:
    """Fetch the full submissions JSON for one company."""
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = session.get(url)
    resp.raise_for_status()
    return resp.json()


def select_target_filings(filing_index: dict) -> list[dict]:
    """
    From the full filing history, pick the most recent N 10-Ks and 10-Qs
    per FORM_LIMITS. The 'recent' block in EDGAR's response is a set of
    parallel arrays (not a list of dicts), so we zip them together first.
    """
    recent = filing_index["filings"]["recent"]
    fields = ["form", "filingDate", "accessionNumber", "primaryDocument", "reportDate"]
    rows = [
        dict(zip(fields, values))
        for values in zip(*(recent[f] for f in fields))
    ]

    selected = []
    counts = {form: 0 for form in TARGET_FORMS}
    # rows are already in reverse-chronological order from EDGAR
    for row in rows:
        form = row["form"]
        if form in TARGET_FORMS and counts[form] < FORM_LIMITS[form]:
            selected.append(row)
            counts[form] += 1
        if all(counts[f] >= FORM_LIMITS[f] for f in TARGET_FORMS):
            break
    return selected


def download_filing(session: requests.Session, cik: str, ticker: str, filing: dict) -> Path:
    """Download the primary document of one filing to data/raw/<TICKER>/."""
    accession_nodash = filing["accessionNumber"].replace("-", "")
    cik_int = str(int(cik))  # archives URLs use CIK without leading zeros
    doc_url = f"{ARCHIVES_BASE}/{cik_int}/{accession_nodash}/{filing['primaryDocument']}"

    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filing['form']}_{filing['filingDate']}_{filing['accessionNumber']}.htm"

    # www.sec.gov needs its own Host header, not data.sec.gov
    headers = dict(session.headers)
    headers["Host"] = "www.sec.gov"

    resp = session.get(doc_url, headers=headers)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user-agent",
        required=True,
        help='Required by SEC. Format: "Your Name your.email@example.com"',
    )
    args = parser.parse_args()

    session = get_session(args.user_agent)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_DIR / "manifest.jsonl"

    manifest_rows = []

    for ticker, meta in COMPANIES.items():
        print(f"\n=== {ticker} ({meta['name']}) ===")
        cik = meta["cik"]

        filing_index = fetch_filing_index(session, cik)
        time.sleep(REQUEST_DELAY_SECONDS)

        targets = select_target_filings(filing_index)
        print(f"Selected {len(targets)} filings: "
              f"{[(f['form'], f['filingDate']) for f in targets]}")

        for filing in targets:
            try:
                local_path = download_filing(session, cik, ticker, filing)
                print(f"  downloaded {filing['form']} {filing['filingDate']} -> {local_path.name}")
            except requests.HTTPError as e:
                print(f"  FAILED {filing['form']} {filing['filingDate']}: {e}")
                continue
            finally:
                time.sleep(REQUEST_DELAY_SECONDS)

            manifest_rows.append({
                "ticker": ticker,
                "company_name": meta["name"],
                "sector": meta["sector"],
                "cik": cik,
                "form": filing["form"],
                "filing_date": filing["filingDate"],
                "report_date": filing["reportDate"],
                "accession_number": filing["accessionNumber"],
                "local_path": str(local_path.relative_to(RAW_DIR.parents[0])),
            })

    with open(manifest_path, "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. {len(manifest_rows)} filings downloaded. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
