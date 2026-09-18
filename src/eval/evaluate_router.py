"""
Evaluate the rule-based financial question router.

Run from the project root:

    python -m src.evaluation.evaluate_router
"""

from __future__ import annotations

from src.retrieval.router import route_question


TEST_CASES = [
    {
        "question": (
            "What was Microsoft's revenue in fiscal year 2026?"
        ),
        "expected": {
            "route": "numeric",
            "tickers": ["MSFT"],
            "metric": "revenue",
            "fiscal_year": 2026,
            "period_type": "annual",
            "comparative": False,
        },
    },
    {
        "question": (
            "Why did Microsoft's revenue increase in fiscal year 2026?"
        ),
        "expected": {
            "route": "mixed",
            "tickers": ["MSFT"],
            "metric": "revenue",
            "fiscal_year": 2026,
            "period_type": "annual",
            "comparative": False,
        },
    },
    {
        "question": (
            "Which company had the highest operating margin in 2025?"
        ),
        "expected": {
            "route": "numeric",
            "tickers": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
            ],
            "metric": "operating_margin",
            "fiscal_year": 2025,
            "period_type": None,
            "comparative": True,
        },
    },
    {
        "question": (
            "Compare Apple and Amazon net income in fiscal year 2025."
        ),
        "expected": {
            "route": "numeric",
            "tickers": ["AAPL", "AMZN"],
            "metric": "net_income",
            "fiscal_year": 2025,
            "period_type": "annual",
            "comparative": True,
        },
    },
    {
        "question": (
            "What risks did Meta discuss in its latest filing?"
        ),
        "expected": {
            "route": "narrative",
            "tickers": ["META"],
            "metric": None,
            "fiscal_year": None,
            "period_type": None,
            "comparative": False,
        },
    },
    {
        "question": (
            "Why did Amazon's operating income change?"
        ),
        "expected": {
            "route": "mixed",
            "tickers": ["AMZN"],
            "metric": "operating_income",
            "fiscal_year": None,
            "period_type": None,
            "comparative": False,
        },
    },
    {
        "question": (
            "What was Apple's revenue in Q2 2026?"
        ),
        "expected": {
            "route": "numeric",
            "tickers": ["AAPL"],
            "metric": "revenue",
            "fiscal_year": 2026,
            "period_type": "quarter",
            "comparative": False,
        },
    },
    {
        "question": (
            "Describe Alphabet's artificial intelligence risks."
        ),
        "expected": {
            "route": "narrative",
            "tickers": ["GOOGL"],
            "metric": None,
            "fiscal_year": None,
            "period_type": None,
            "comparative": False,
        },
    },
]


def evaluate_case(test_case: dict) -> dict:
    """Compare one router result with its expected fields."""

    question = test_case["question"]
    expected = test_case["expected"]
    actual = route_question(question)

    field_results = {}

    for field, expected_value in expected.items():
        actual_value = actual.get(field)

        field_results[field] = {
            "expected": expected_value,
            "actual": actual_value,
            "correct": actual_value == expected_value,
        }

    passed_fields = sum(
        result["correct"]
        for result in field_results.values()
    )

    total_fields = len(field_results)

    return {
        "question": question,
        "passed": passed_fields == total_fields,
        "passed_fields": passed_fields,
        "total_fields": total_fields,
        "fields": field_results,
    }


def print_case_result(
    index: int,
    result: dict,
) -> None:
    """Print a readable evaluation result."""

    status = "PASS" if result["passed"] else "FAIL"

    print(
        f"\n{index}. [{status}] "
        f"{result['question']}"
    )

    print(
        f"   Fields correct: "
        f"{result['passed_fields']}/"
        f"{result['total_fields']}"
    )

    for field, comparison in result["fields"].items():
        if not comparison["correct"]:
            print(f"   Incorrect field: {field}")
            print(
                f"      Expected: "
                f"{comparison['expected']}"
            )
            print(
                f"      Actual:   "
                f"{comparison['actual']}"
            )


def main() -> None:
    results = [
        evaluate_case(test_case)
        for test_case in TEST_CASES
    ]

    print("Router Evaluation")
    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print_case_result(index, result)

    passed_cases = sum(
        result["passed"]
        for result in results
    )

    total_cases = len(results)

    total_fields = sum(
        result["total_fields"]
        for result in results
    )

    passed_fields = sum(
        result["passed_fields"]
        for result in results
    )

    case_accuracy = passed_cases / total_cases
    field_accuracy = passed_fields / total_fields

    print("\nSummary")
    print("=" * 80)
    print(
        f"Exact-match cases: "
        f"{passed_cases}/{total_cases} "
        f"({case_accuracy:.1%})"
    )
    print(
        f"Field accuracy: "
        f"{passed_fields}/{total_fields} "
        f"({field_accuracy:.1%})"
    )


if __name__ == "__main__":
    main()