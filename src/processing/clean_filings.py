"""
Clean raw SEC 10-K and 10-Q HTML filings.

Input:
    data/processed/manifest.jsonl
    data/raw/<ticker>/*.htm

Output:
    data/processed/filings_clean.jsonl

The raw HTML files and manifest are never modified.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MANIFEST_PATH = (
    DATA_DIR
    / "processed"
    / "manifest.jsonl"
)

OUTPUT_PATH = (
    DATA_DIR
    / "processed"
    / "filings_clean.jsonl"
)


def load_manifest() -> list[dict]:
    """Load the selected filing records."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest was not found: {MANIFEST_PATH}"
        )

    records = []

    with MANIFEST_PATH.open(
        encoding="utf-8"
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON on manifest line "
                    f"{line_number}"
                ) from error

            records.append(record)

    return records


def build_source_url(
    cik: str,
    accession: str,
) -> str:
    """Create the SEC filing-index URL."""

    cik_without_leading_zeros = str(int(cik))
    accession_directory = accession.replace("-", "")

    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_without_leading_zeros}/"
        f"{accession_directory}/"
        f"{accession}-index.html"
    )


def remove_hidden_elements(
    soup: BeautifulSoup,
) -> None:
    """
    Remove scripts, styles, metadata, and hidden XBRL content.

    Visible inline-XBRL elements are preserved because they contain
    text that appears in the filing.
    """

    removable_tags = [
        "script",
        "style",
        "noscript",
        "svg",
        "meta",
        "link",
        "head",
        "ix:header",
        "ix:hidden",
    ]

    for tag_name in removable_tags:
        tags = list(soup.find_all(tag_name))

        for tag in tags:
            # A parent element may already have removed this tag.
            if tag.parent is not None:
                tag.decompose()

    hidden_tags = list(
        soup.find_all(
            attrs={"hidden": True}
        )
    )

    for tag in hidden_tags:
        if tag.parent is not None:
            tag.decompose()

    aria_hidden_tags = list(
        soup.find_all(
            attrs={"aria-hidden": "true"}
        )
    )

    for tag in aria_hidden_tags:
        if tag.parent is not None:
            tag.decompose()

    styled_tags = list(
        soup.find_all(style=True)
    )

    for tag in styled_tags:
        # Decomposing a parent can invalidate its child tags.
        if tag.parent is None or tag.attrs is None:
            continue

        style = tag.attrs.get("style") or ""

        normalized_style = re.sub(
            r"\s+",
            "",
            style.lower(),
        )

        if (
            "display:none" in normalized_style
            or "visibility:hidden" in normalized_style
        ):
            tag.decompose()


def normalize_line(line: str) -> str:
    """Normalize whitespace and Unicode inside one line."""

    line = unicodedata.normalize(
        "NFKC",
        line,
    )

    line = line.replace("\xa0", " ")
    line = re.sub(r"[ \t]+", " ", line)

    return line.strip()


def is_noise_line(line: str) -> bool:
    """Identify obvious HTML and page-formatting noise."""

    if not line:
        return True

    # Lines containing only a page number.
    if re.fullmatch(
        r"(page\s+)?\d+",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    # Lines made almost entirely from punctuation.
    if re.fullmatch(
        r"[-_=•·.\s]+",
        line,
    ):
        return True

    return False


def extract_clean_text(html: str) -> str:
    """Convert one raw SEC HTML filing into readable text."""

    try:
        soup = BeautifulSoup(
            html,
            "lxml",
        )
    except Exception:
        soup = BeautifulSoup(
            html,
            "html.parser",
        )

    remove_hidden_elements(soup)

    # Newlines preserve headings, paragraphs, and table rows.
    raw_text = soup.get_text(separator="\n")

    clean_lines = []
    previous_line = None

    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)

        if is_noise_line(line):
            continue

        # Remove consecutive duplicate lines.
        if line == previous_line:
            continue

        clean_lines.append(line)
        previous_line = line

    text = "\n".join(clean_lines)

    # Prevent excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def process_filing(filing: dict) -> dict:
    """Read and clean one filing from the manifest."""

    raw_path = (
        DATA_DIR
        / filing["local_path"]
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw filing was not found: {raw_path}"
        )

    html = raw_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    clean_text = extract_clean_text(html)

    if not clean_text:
        raise ValueError(
            f"No text was extracted from {raw_path}"
        )

    accession = filing["accession_number"]

    return {
        "ticker": filing["ticker"],
        "company_name": filing["company_name"],
        "sector": filing["sector"],
        "cik": filing["cik"],
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "report_date": filing["report_date"],
        "accession_number": accession,
        "local_path": filing["local_path"],
        "source_url": build_source_url(
            cik=filing["cik"],
            accession=accession,
        ),
        "word_count": len(clean_text.split()),
        "character_count": len(clean_text),
        "text": clean_text,
    }


def main() -> None:
    """Clean every filing listed in the manifest."""

    filings = load_manifest()

    print(
        f"Loaded {len(filings)} filings "
        "from manifest.jsonl"
    )

    cleaned_filings = []
    failures = []

    for filing in filings:
        ticker = filing["ticker"]
        form = filing["form"]
        filing_date = filing["filing_date"]

        try:
            cleaned = process_filing(filing)
            cleaned_filings.append(cleaned)

            print(
                f"Cleaned {ticker} {form} "
                f"{filing_date}: "
                f"{cleaned['word_count']:,} words"
            )

        except Exception as error:
            failure = {
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "error": str(error),
            }

            failures.append(failure)

            print(
                f"FAILED {ticker} {form} "
                f"{filing_date}: {error}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Opening with "w" replaces the incomplete previous output.
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for filing in cleaned_filings:
            file.write(
                json.dumps(filing) + "\n"
            )

    print("\nCleaning complete")
    print(
        f"Successful filings: "
        f"{len(cleaned_filings)}"
    )
    print(
        f"Failed filings: "
        f"{len(failures)}"
    )
    print(f"Output: {OUTPUT_PATH}")

    if failures:
        print("\nFailures:")

        for failure in failures:
            print(failure)

        raise RuntimeError(
            f"{len(failures)} filings "
            "failed cleaning."
        )

    if len(cleaned_filings) != len(filings):
        raise RuntimeError(
            "Output count does not match "
            "manifest count."
        )


if __name__ == "__main__":
    main()