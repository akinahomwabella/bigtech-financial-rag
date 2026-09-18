"""
Pull clean structured financial facts from the SEC companyconcept API.

Only facts belonging to filings selected in manifest.jsonl are retained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests


sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.companies import COMPANIES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "xbrl_facts.jsonl"

COMPANYCONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/"
    "CIK{cik}/us-gaap/{concept}.json"
)

REQUEST_DELAY_SECONDS = 0.25

CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "rnd_expense": [
        "ResearchAndDevelopmentExpense",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ],
}


def get_session(user_agent: str) -> requests.Session:
    """Create an SEC request session."""

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
    )

    return session


def fetch_concept(
    session: requests.Session,
    cik: str,
    concept: str,
) -> dict | None:
    """Download one XBRL concept for one company."""

    url = COMPANYCONCEPT_URL.format(
        cik=cik,
        concept=concept,
    )

    response = session.get(url, timeout=30)

    if response.status_code == 404:
        return None

    response.raise_for_status()
    return response.json()


def load_manifest() -> dict[str, dict]:
    """Load selected filings, indexed by accession number."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    filings = {}

    with MANIFEST_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            accession = record["accession_number"]
            filings[accession] = record

    return filings


def classify_period(
    start: str | None,
    end: str,
    form: str,
) -> str:
    """Classify a fact as annual, quarter, YTD, or instant."""

    if not start:
        return "instant"

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    duration = (end_date - start_date).days + 1

    if form == "10-K" and 300 <= duration <= 400:
        return "annual"

    if 70 <= duration <= 110:
        return "quarter"

    if 150 <= duration <= 300:
        return "ytd"

    return "other"


def build_source_url(cik: str, accession: str) -> str:
    """Build the SEC filing-index URL used for citations."""

    cik_without_leading_zeros = str(int(cik))
    accession_directory = accession.replace("-", "")

    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_without_leading_zeros}/"
        f"{accession_directory}/"
        f"{accession}-index.html"
    )


def make_deduplication_key(record: dict) -> tuple:
    """Create a key representing one economic fact."""

    return (
        record["ticker"],
        record["metric"],
        record["accession_number"],
        record["start"],
        record["end"],
        record["period_type"],
        record["value"],
        record["unit"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--user-agent",
        required=True,
        help='Example: "Your Name your.email@example.com"',
    )

    args = parser.parse_args()

    manifest_by_accession = load_manifest()
    allowed_accessions = set(manifest_by_accession)

    print(
        f"Loaded {len(allowed_accessions)} selected filings "
        "from manifest.jsonl"
    )

    session = get_session(args.user_agent)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    unique_rows = {}

    for ticker, company_meta in COMPANIES.items():
        cik = company_meta["cik"]

        print(f"\n=== {ticker} ===")

        for metric, concept_candidates in CONCEPTS.items():
            count_before = len(unique_rows)
            concepts_found = []

            # Check all aliases instead of stopping at the first one.
            for concept in concept_candidates:
                data = fetch_concept(
                    session=session,
                    cik=cik,
                    concept=concept,
                )

                time.sleep(REQUEST_DELAY_SECONDS)

                if data is None:
                    continue

                concepts_found.append(concept)

                usd_facts = data.get("units", {}).get("USD", [])

                for fact in usd_facts:
                    accession = fact.get("accn")
                    form = fact.get("form")
                    end = fact.get("end")

                    # Keep only the 25 selected manifest filings.
                    if accession not in allowed_accessions:
                        continue

                    # Exclude DEF 14A, 8-K, and other forms.
                    if form not in {"10-K", "10-Q"}:
                        continue

                    filing = manifest_by_accession[accession]

                    # Ensure this fact belongs to the expected company.
                    if filing["ticker"] != ticker:
                        continue

                    # Remove earlier comparison periods repeated in
                    # a newer filing.
                    if end != filing["report_date"]:
                        continue

                    period_type = classify_period(
                        start=fact.get("start"),
                        end=end,
                        form=form,
                    )

                    if period_type == "other":
                        continue

                    row = {
                        "ticker": ticker,
                        "company_name": filing["company_name"],
                        "sector": filing["sector"],
                        "cik": filing["cik"],
                        "metric": metric,
                        "xbrl_concept": concept,
                        "value": fact.get("val"),
                        "unit": "USD",
                        "start": fact.get("start"),
                        "end": end,
                        "period_type": period_type,
                        "fiscal_year": fact.get("fy"),
                        "fiscal_period": fact.get("fp"),
                        "form": form,
                        "filed": fact.get("filed"),
                        "accession_number": accession,
                        "frame": fact.get("frame"),
                        "local_path": filing["local_path"],
                        "source_url": build_source_url(
                            cik=filing["cik"],
                            accession=accession,
                        ),
                    }

                    key = make_deduplication_key(row)

                    if key not in unique_rows:
                        unique_rows[key] = row

            clean_count = len(unique_rows) - count_before

            if not concepts_found:
                print(f"  {metric}: no matching concept")
            else:
                print(
                    f"  {metric}: {clean_count} clean facts "
                    f"from {', '.join(concepts_found)}"
                )

    rows = list(unique_rows.values())

    rows.sort(
        key=lambda row: (
            row["ticker"],
            row["end"],
            row["metric"],
            row["period_type"],
        )
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    print(f"\nDone. {len(rows)} clean facts written to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()