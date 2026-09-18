import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "filings_clean.jsonl"
)


def main():
    with CLEAN_PATH.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    required_fields = {
        "ticker",
        "company_name",
        "form",
        "filing_date",
        "report_date",
        "accession_number",
        "local_path",
        "source_url",
        "word_count",
        "text",
    }

    missing_fields = []
    empty_text = []
    unusually_short = []
    accessions = []

    for row_number, row in enumerate(rows, start=1):
        missing = required_fields - set(row)

        if missing:
            missing_fields.append(
                (row_number, sorted(missing))
            )

        text = row.get("text", "").strip()

        if not text:
            empty_text.append(row_number)

        if len(text.split()) < 1_000:
            unusually_short.append(
                (
                    row.get("ticker"),
                    row.get("form"),
                    row.get("filing_date"),
                    len(text.split()),
                )
            )

        accessions.append(
            row.get("accession_number")
        )

    duplicate_accessions = (
        len(accessions)
        - len(set(accessions))
    )

    print("Total filings:", len(rows))
    print(
        "Companies:",
        Counter(row["ticker"] for row in rows),
    )
    print(
        "Forms:",
        Counter(row["form"] for row in rows),
    )
    print("Missing fields:", missing_fields)
    print("Empty text records:", empty_text)
    print("Unusually short filings:", unusually_short)
    print("Duplicate accessions:", duplicate_accessions)

    if len(rows) != 25:
        raise ValueError(
            f"Expected 25 filings, found {len(rows)}"
        )

    if missing_fields:
        raise ValueError("Some records have missing fields.")

    if empty_text:
        raise ValueError("Some records have empty text.")

    if duplicate_accessions:
        raise ValueError(
            "Duplicate filing accessions were found."
        )

    print("\nClean filing validation passed.")


if __name__ == "__main__":
    main()