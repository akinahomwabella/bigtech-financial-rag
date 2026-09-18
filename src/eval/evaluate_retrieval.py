"""
Evaluate structured and narrative retrieval.

Run:
    python -m src.evaluation.evaluate_retrieval
"""

from __future__ import annotations

from src.retrieval.hybrid_retriever import hybrid_retrieve

from src.retrieval.router import route_question
TEST_CASES = [
    {
        "question": (
            "What was Microsoft's revenue in fiscal year 2026?"
        ),
        "expected_route": "numeric",
        "expected_tickers": {"MSFT"},
        "expected_metric": "revenue",
    },
    {
        "question": (
            "Which company had the highest operating margin "
            "in fiscal year 2025?"
        ),
        "expected_route": "numeric",
        "expected_tickers": {
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
        },
        "expected_metric": "operating_margin",
    },
    {
        "question": (
            "Why did Microsoft's revenue increase "
            "in fiscal year 2026?"
        ),
        "expected_route": "mixed",
        "expected_tickers": {"MSFT"},
        "expected_metric": "revenue",
    },
    {
        "question": (
            "What risks did Meta discuss in its latest filing?"
        ),
        "expected_route": "narrative",
        "expected_tickers": {"META"},
        "expected_metric": None,
    },
]


def evaluate_test(test: dict) -> dict:
    """Run retrieval and calculate basic quality checks."""

    result = hybrid_retrieve(
        question=test["question"],
        top_k=5,
    )

    route = result["route"]
    numeric = result["numeric_evidence"]
    narrative = result["narrative_evidence"]

    returned_numeric_tickers = {
        fact["ticker"]
        for fact in numeric
    }

    returned_narrative_tickers = {
        chunk["ticker"]
        for chunk in narrative
    }

    checks = {
        "route_correct": (
            route["route"]
            == test["expected_route"]
        ),
        "numeric_evidence_present": (
            bool(numeric)
            if test["expected_route"] in {"numeric", "mixed"}
            else True
        ),
        "narrative_evidence_present": (
            bool(narrative)
            if test["expected_route"] in {"narrative", "mixed"}
            else True
        ),
    }

    if test["expected_metric"] is not None:
        checks["metric_correct"] = all(
            fact["metric"]
            == test["expected_metric"]
            for fact in numeric
        ) and bool(numeric)

    if numeric:
        checks["numeric_ticker_coverage"] = (
            test["expected_tickers"]
            .issubset(returned_numeric_tickers)
        )

    if narrative:
        checks["narrative_tickers_correct"] = (
            returned_narrative_tickers
            .issubset(test["expected_tickers"])
        )

        checks["citations_present"] = all(
            chunk.get("source_url")
            and chunk.get("accession_number")
            for chunk in narrative
        )

    passed = all(checks.values())

    return {
        "question": test["question"],
        "passed": passed,
        "checks": checks,
        "numeric_count": len(numeric),
        "narrative_count": len(narrative),
        "numeric_tickers": sorted(
            returned_numeric_tickers
        ),
        "narrative_tickers": sorted(
            returned_narrative_tickers
        ),
    }


def main() -> None:
    print("Retrieval Evaluation")
    print("=" * 80)

    results = []

    for index, test in enumerate(
        TEST_CASES,
        start=1,
    ):
        result = evaluate_test(test)
        results.append(result)

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"\n{index}. [{status}] "
            f"{result['question']}"
        )
        print(
            "   Numeric evidence:",
            result["numeric_count"],
        )
        print(
            "   Narrative evidence:",
            result["narrative_count"],
        )
        print(
            "   Numeric tickers:",
            result["numeric_tickers"],
        )
        print(
            "   Narrative tickers:",
            result["narrative_tickers"],
        )

        for check, passed in result["checks"].items():
            symbol = "PASS" if passed else "FAIL"
            print(f"   {symbol}: {check}")

    passed_count = sum(
        result["passed"]
        for result in results
    )

    total_count = len(results)

    print("\nSummary")
    print("=" * 80)
    print(
        f"Passed: {passed_count}/{total_count} "
        f"({passed_count / total_count:.1%})"
    )


if __name__ == "__main__":
    main()