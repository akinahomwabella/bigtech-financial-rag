"""
Split cleaned SEC filings into retrieval-sized chunks.

Input:
    data/processed/filings_clean.jsonl

Output:
    data/processed/filing_chunks.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "filings_clean.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "filing_chunks.jsonl"

CHUNK_SIZE_WORDS = 550
CHUNK_OVERLAP_WORDS = 75
MIN_CHUNK_WORDS = 75


ITEM_HEADING_PATTERN = re.compile(
    r"^(?:PART\s+[IVX]+|ITEM\s+\d+[A-Z]?(?:\s*[\.:—-]\s*.*)?)$",
    flags=re.IGNORECASE,
)

KNOWN_SECTION_PATTERNS = [
    re.compile(
        r"^business$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^risk factors$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^management[’']s discussion and analysis",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^financial statements",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^controls and procedures$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^legal proceedings$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^market risk",
        flags=re.IGNORECASE,
    ),
]


def load_clean_filings() -> list[dict]:
    """Load cleaned filing records."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Clean filing file was not found: {INPUT_PATH}"
        )

    filings = []

    with INPUT_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                filing = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            filings.append(filing)

    return filings


def is_section_heading(line: str) -> bool:
    """Return True if a line resembles a filing section heading."""

    normalized = " ".join(line.split()).strip()

    if not normalized:
        return False

    if len(normalized) > 180:
        return False

    if ITEM_HEADING_PATTERN.match(normalized):
        return True

    return any(
        pattern.match(normalized)
        for pattern in KNOWN_SECTION_PATTERNS
    )


def normalize_section_name(line: str) -> str:
    """Normalize a section heading for metadata."""

    section = " ".join(line.split()).strip()
    section = section.strip(" .:-—")

    return section or "Unknown"


def split_into_sections(text: str) -> list[dict]:
    """
    Split filing text using SEC item and narrative headings.

    Text before the first detected heading receives the section
    name 'Document introduction'.
    """

    sections = []
    current_section = "Document introduction"
    current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if is_section_heading(line):
            if current_lines:
                sections.append(
                    {
                        "section": current_section,
                        "text": "\n".join(current_lines),
                    }
                )

            current_section = normalize_section_name(line)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section": current_section,
                "text": "\n".join(current_lines),
            }
        )

    return sections


def chunk_words(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping word-based chunks."""

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += step

    return chunks


def build_chunks(filing: dict) -> list[dict]:
    """Create section-aware chunks for one filing."""

    sections = split_into_sections(filing["text"])
    chunks = []
    chunk_index = 0

    for section in sections:
        section_name = section["section"]

        section_chunks = chunk_words(
            text=section["text"],
            chunk_size=CHUNK_SIZE_WORDS,
            overlap=CHUNK_OVERLAP_WORDS,
        )

        for text in section_chunks:
            word_count = len(text.split())

            # Keep short chunks if they are the only content
            # available for a meaningful section.
            if (
                word_count < MIN_CHUNK_WORDS
                and len(section_chunks) > 1
            ):
                continue

            chunk_id = (
                f"{filing['ticker']}_"
                f"{filing['accession_number']}_"
                f"{chunk_index:04d}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "ticker": filing["ticker"],
                    "company_name": filing["company_name"],
                    "sector": filing["sector"],
                    "cik": filing["cik"],
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "report_date": filing["report_date"],
                    "accession_number": filing[
                        "accession_number"
                    ],
                    "section": section_name,
                    "word_count": word_count,
                    "local_path": filing["local_path"],
                    "source_url": filing["source_url"],
                    "text": text,
                }
            )

            chunk_index += 1

    return chunks


def main() -> None:
    filings = load_clean_filings()

    print(
        f"Loaded {len(filings)} cleaned filings"
    )

    all_chunks = []

    for filing in filings:
        chunks = build_chunks(filing)
        all_chunks.extend(chunks)

        print(
            f"Chunked {filing['ticker']} "
            f"{filing['form']} "
            f"{filing['filing_date']}: "
            f"{len(chunks)} chunks"
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk) + "\n"
            )

    print("\nChunking complete")
    print(f"Filings processed: {len(filings)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()