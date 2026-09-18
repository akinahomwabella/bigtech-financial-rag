"""
Execute hybrid retrieval across:

1. Structured SEC XBRL facts
2. Narrative SEC filing chunks

The script retrieves evidence but does not generate the final LLM answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.router import route_question


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "xbrl_facts.jsonl"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "chroma"
)

COLLECTION_NAME = "sec_filing_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

ALL_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
]


def load_facts() -> list[dict]:
    """Load structured XBRL facts."""

    with FACTS_PATH.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def fact_matches_route(
    fact: dict,
    route: dict,
    metric: str,
) -> bool:
    """Check whether a fact satisfies routed metadata."""

    tickers = route["tickers"]

    if tickers and fact["ticker"] not in tickers:
        return False

    if fact["metric"] != metric:
        return False

    if (
        route["fiscal_year"] is not None
        and fact.get("fiscal_year")
        != route["fiscal_year"]
    ):
        return False

    if (
        route["fiscal_period"] is not None
        and fact.get("fiscal_period")
        != route["fiscal_period"]
    ):
        return False

    if (
        route["period_type"] is not None
        and fact.get("period_type")
        != route["period_type"]
    ):
        return False

    return True


def select_latest_per_company(
    facts: list[dict],
) -> list[dict]:
    """Keep the latest matching fact for each company."""

    latest = {}

    period_priority = {
        "annual": 3,
        "quarter": 2,
        "ytd": 1,
        "instant": 0,
    }

    for fact in facts:
        ticker = fact["ticker"]

        sort_key = (
            fact["end"],
            period_priority.get(
                fact.get("period_type"),
                -1,
            ),
            fact.get("filed", ""),
        )

        existing = latest.get(ticker)

        if existing is None:
            latest[ticker] = fact
            continue

        existing_key = (
            existing["end"],
            period_priority.get(
                existing.get("period_type"),
                -1,
            ),
            existing.get("filed", ""),
        )

        if sort_key > existing_key:
            latest[ticker] = fact

    return sorted(
        latest.values(),
        key=lambda fact: fact["ticker"],
    )


def retrieve_direct_facts(
    facts: list[dict],
    route: dict,
    metric: str,
) -> list[dict]:
    """Retrieve direct values such as revenue or net income."""

    matches = [
        fact
        for fact in facts
        if fact_matches_route(
            fact=fact,
            route=route,
            metric=metric,
        )
    ]

    # If the question provides no time information,
    # return only the latest match for each company.
    if (
        route["fiscal_year"] is None
        and route["fiscal_period"] is None
        and route["period_type"] is None
    ):
        matches = select_latest_per_company(matches)

    return matches


def retrieve_margin_facts(
    facts: list[dict],
    route: dict,
    margin_metric: str,
) -> list[dict]:
    """Calculate operating or net margin from matching facts."""

    numerator_metric = {
        "operating_margin": "operating_income",
        "net_margin": "net_income",
    }[margin_metric]

    revenue_facts = retrieve_direct_facts(
        facts=facts,
        route=route,
        metric="revenue",
    )

    numerator_facts = retrieve_direct_facts(
        facts=facts,
        route=route,
        metric=numerator_metric,
    )

    numerator_lookup = {}

    for fact in numerator_facts:
        key = (
            fact["ticker"],
            fact["end"],
            fact["period_type"],
            fact.get("fiscal_period"),
        )

        numerator_lookup[key] = fact

    results = []

    for revenue in revenue_facts:
        key = (
            revenue["ticker"],
            revenue["end"],
            revenue["period_type"],
            revenue.get("fiscal_period"),
        )

        numerator = numerator_lookup.get(key)

        if numerator is None:
            continue

        if revenue["value"] == 0:
            continue

        margin = (
            numerator["value"]
            / revenue["value"]
            * 100
        )

        results.append(
            {
                "ticker": revenue["ticker"],
                "company_name": revenue["company_name"],
                "metric": margin_metric,
                "value": margin,
                "unit": "percent",
                "fiscal_year": revenue.get("fiscal_year"),
                "fiscal_period": revenue.get(
                    "fiscal_period"
                ),
                "period_type": revenue["period_type"],
                "start": revenue.get("start"),
                "end": revenue["end"],
                "revenue": revenue["value"],
                "numerator_metric": numerator_metric,
                "numerator_value": numerator["value"],
                "accession_number": revenue[
                    "accession_number"
                ],
                "source_url": revenue["source_url"],
            }
        )

    results.sort(
        key=lambda result: result["value"],
        reverse=True,
    )

    return results


def retrieve_numeric_evidence(
    route: dict,
) -> list[dict]:
    """Retrieve or calculate structured numeric evidence."""

    metric = route["metric"]

    if metric is None:
        return []

    facts = load_facts()

    if metric in {
        "operating_margin",
        "net_margin",
    }:
        return retrieve_margin_facts(
            facts=facts,
            route=route,
            margin_metric=metric,
        )

    return retrieve_direct_facts(
        facts=facts,
        route=route,
        metric=metric,
    )


def build_chroma_filter(
    tickers: list[str],
) -> dict | None:
    """Create a ticker filter for narrative retrieval."""

    if not tickers:
        return None

    if len(tickers) == 1:
        return {
            "ticker": tickers[0]
        }

    return {
        "ticker": {
            "$in": tickers
        }
    }


def retrieve_narrative_evidence(
    question: str,
    tickers: list[str],
    top_k: int,
) -> list[dict]:
    """Retrieve semantically relevant filing chunks."""

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(INDEX_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    query_embedding = model.encode(
        [QUERY_PREFIX + question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

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

    where_filter = build_chroma_filter(tickers)

    if where_filter is not None:
        query_arguments["where"] = where_filter

    results = collection.query(
        **query_arguments
    )

    evidence = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        evidence.append(
            {
                "chunk_id": chunk_id,
                "similarity": round(
                    1 - distance,
                    4,
                ),
                "ticker": metadata["ticker"],
                "company_name": metadata[
                    "company_name"
                ],
                "form": metadata["form"],
                "filing_date": metadata[
                    "filing_date"
                ],
                "report_date": metadata[
                    "report_date"
                ],
                "section": metadata["section"],
                "accession_number": metadata[
                    "accession_number"
                ],
                "source_url": metadata[
                    "source_url"
                ],
                "text": document,
            }
        )

    return evidence


def hybrid_retrieve(
    question: str,
    top_k: int = 5,
) -> dict:
    """Route a question and retrieve supporting evidence."""

    route = route_question(question)

    numeric_evidence = []
    narrative_evidence = []

    if route["route"] in {
        "numeric",
        "mixed",
    }:
        numeric_evidence = (
            retrieve_numeric_evidence(route)
        )

    if route["route"] in {
        "narrative",
        "mixed",
    }:
        narrative_evidence = (
            retrieve_narrative_evidence(
                question=question,
                tickers=route["tickers"],
                top_k=top_k,
            )
        )

    return {
        "route": route,
        "numeric_evidence": numeric_evidence,
        "narrative_evidence": narrative_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hybrid SEC retrieval."
    )

    parser.add_argument(
        "question",
        help="Financial question",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    result = hybrid_retrieve(
        question=args.question,
        top_k=args.top_k,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()