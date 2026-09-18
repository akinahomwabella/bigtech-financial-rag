"""
Create a persistent Chroma vector index from SEC filing chunks.

Input:
    data/processed/filing_chunks.jsonl

Output:
    data/index/chroma/
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "filing_chunks.jsonl"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "chroma"
)

COLLECTION_NAME = "sec_filing_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    """Load the processed filing chunks."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file was not found: {CHUNKS_PATH}"
        )

    chunks = []

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            chunks.append(chunk)

    if not chunks:
        raise ValueError("No chunks were loaded.")

    return chunks


def build_metadata(chunk: dict) -> dict:
    """
    Create Chroma-compatible metadata.

    Chroma metadata values must be strings, integers,
    floats, or booleans.
    """

    return {
        "ticker": str(chunk["ticker"]),
        "company_name": str(chunk["company_name"]),
        "sector": str(chunk["sector"]),
        "cik": str(chunk["cik"]),
        "form": str(chunk["form"]),
        "filing_date": str(chunk["filing_date"]),
        "report_date": str(chunk["report_date"]),
        "accession_number": str(
            chunk["accession_number"]
        ),
        "section": str(chunk["section"]),
        "chunk_index": int(chunk["chunk_index"]),
        "word_count": int(chunk["word_count"]),
        "local_path": str(chunk["local_path"]),
        "source_url": str(chunk["source_url"]),
    }


def main() -> None:
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(INDEX_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
        },
    )

    print(
        f"Existing vectors in collection: "
        f"{collection.count()}"
    )

    total_batches = (
        len(chunks) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(chunks), BATCH_SIZE),
        start=1,
    ):
        batch = chunks[start : start + BATCH_SIZE]

        ids = [
            chunk["chunk_id"]
            for chunk in batch
        ]

        documents = [
            chunk["text"]
            for chunk in batch
        ]

        metadata = [
            build_metadata(chunk)
            for chunk in batch
        ]

        embeddings = model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadata,
            embeddings=embeddings.tolist(),
        )

        print(
            f"Indexed batch {batch_number}/"
            f"{total_batches}"
        )

    indexed_count = collection.count()

    print("\nIndexing complete")
    print(f"Expected chunks: {len(chunks)}")
    print(f"Indexed vectors: {indexed_count}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Index directory: {INDEX_DIR}")

    if indexed_count != len(chunks):
        raise ValueError(
            "Indexed vector count does not match "
            "the chunk count. The collection may "
            "contain stale records from an older run."
        )


if __name__ == "__main__":
    main()