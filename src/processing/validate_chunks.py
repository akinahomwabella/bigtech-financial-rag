import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHUNKS_PATH = PROCESSED_DIR / "filing_chunks.jsonl"
FILINGS_PATH = PROCESSED_DIR / "filings_clean.jsonl"


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    chunks = load_jsonl(CHUNKS_PATH)
    filings = load_jsonl(FILINGS_PATH)

    expected_accessions = {
        filing["accession_number"]
        for filing in filings
    }

    chunk_accessions = {
        chunk["accession_number"]
        for chunk in chunks
    }

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    empty_chunks = [
        chunk["chunk_id"]
        for chunk in chunks
        if not chunk.get("text", "").strip()
    ]

    incorrect_word_counts = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["word_count"]
        != len(chunk["text"].split())
    ]

    oversized_chunks = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["word_count"] > 550
    ]

    duplicate_ids = (
        len(chunk_ids)
        - len(set(chunk_ids))
    )

    missing_filings = (
        expected_accessions
        - chunk_accessions
    )

    word_counts = [
        chunk["word_count"]
        for chunk in chunks
    ]

    print("Total chunks:", len(chunks))
    print(
        "Chunks by company:",
        Counter(chunk["ticker"] for chunk in chunks),
    )
    print(
        "Chunks by form:",
        Counter(chunk["form"] for chunk in chunks),
    )
    print(
        "Filings represented:",
        len(chunk_accessions),
    )
    print("Minimum words:", min(word_counts))
    print("Maximum words:", max(word_counts))
    print(
        "Average words:",
        round(sum(word_counts) / len(word_counts), 2),
    )
    print("Duplicate chunk IDs:", duplicate_ids)
    print("Empty chunks:", len(empty_chunks))
    print(
        "Incorrect word counts:",
        len(incorrect_word_counts),
    )
    print("Oversized chunks:", len(oversized_chunks))
    print("Missing filings:", missing_filings)

    if duplicate_ids:
        raise ValueError("Duplicate chunk IDs found.")

    if empty_chunks:
        raise ValueError("Empty chunks found.")

    if incorrect_word_counts:
        raise ValueError(
            "Some stored word counts are incorrect."
        )

    if oversized_chunks:
        raise ValueError(
            "Some chunks exceed 550 words."
        )

    if missing_filings:
        raise ValueError(
            "Some filings produced no chunks."
        )

    if len(chunk_accessions) != 25:
        raise ValueError(
            "Expected chunks from 25 filings."
        )

    print("\nChunk validation passed.")


if __name__ == "__main__":
    main()