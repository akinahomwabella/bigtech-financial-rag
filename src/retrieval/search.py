"""
Search the SEC filing vector index.

Examples:
    python src/retrieval/search.py \
        "Why did Microsoft cloud revenue increase?"

    python src/retrieval/search.py \
        "What risks did the company discuss?" \
        --ticker META \
        --form 10-K \
        --top-k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "chroma"
)

COLLECTION_NAME = "sec_filing_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE recommends this instruction for retrieval queries.
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


def build_filter(
    ticker: str | None,
    form: str | None,
) -> dict | None:
    """Create an optional Chroma metadata filter."""

    conditions = []

    if ticker:
        conditions.append(
            {"ticker": ticker.upper()}
        )

    if form:
        conditions.append(
            {"form": form.upper()}
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions
    }


def search(
    query: str,
    top_k: int,
    ticker: str | None,
    form: str | None,
) -> None:
    """Search for relevant filing chunks."""

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Vector index was not found: {INDEX_DIR}"
        )

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(INDEX_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    query_text = QUERY_PREFIX + query

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    where_filter = build_filter(
        ticker=ticker,
        form=form,
    )

    query_arguments = {
        "query_embeddings": [
            query_embedding.tolist()
        ],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter is not None:
        query_arguments["where"] = where_filter

    results = collection.query(
        **query_arguments
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    print("\nQuery:", query)
    print("Results:", len(ids))

    if ticker:
        print("Ticker filter:", ticker.upper())

    if form:
        print("Form filter:", form.upper())

    for rank, (
        chunk_id,
        document,
        meta,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadata,
            distances,
        ),
        start=1,
    ):
        similarity = 1 - distance

        preview = " ".join(
            document.split()
        )[:800]

        print("\n" + "=" * 80)
        print(f"Rank: {rank}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Chunk ID: {chunk_id}")
        print(
            f"Company: {meta['company_name']} "
            f"({meta['ticker']})"
        )
        print(f"Form: {meta['form']}")
        print(
            f"Filing date: "
            f"{meta['filing_date']}"
        )
        print(
            f"Report date: "
            f"{meta['report_date']}"
        )
        print(f"Section: {meta['section']}")
        print(
            f"Accession: "
            f"{meta['accession_number']}"
        )
        print(f"Source: {meta['source_url']}")
        print("\nText preview:")
        print(preview)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search cleaned SEC filing chunks."
        )
    )

    parser.add_argument(
        "query",
        help="Question or search query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )

    parser.add_argument(
        "--ticker",
        help="Optional ticker filter, such as MSFT",
    )

    parser.add_argument(
        "--form",
        choices=["10-K", "10-Q"],
        help="Optional filing-form filter",
    )

    args = parser.parse_args()

    search(
        query=args.query,
        top_k=args.top_k,
        ticker=args.ticker,
        form=args.form,
    )


if __name__ == "__main__":
    main()