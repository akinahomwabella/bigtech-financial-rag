"""
Calculate financial metrics from exact SEC XBRL facts.

Examples:
    python src/retrieval/financial_metrics.py \
        --ticker MSFT \
        --metric operating_margin \
        --fiscal-year 2026 \
        --period-type annual

    python src/retrieval/financial_metrics.py \
        --ticker AAPL MSFT GOOGL AMZN META \
        --metric net_margin \
        --fiscal-year 2025 \
        --period-type annual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "xbrl_facts.jsonl"
)


def load_facts() -> list[dict]:
    """Load structured XBRL facts."""

    with FACTS_PATH.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def find_fact(
    facts: list[dict],
    ticker: str,
    metric: str,
    fiscal_year: int,
    fiscal_period: str | None,
    period_type: str,
) -> dict | None:
    """Find one matching financial fact."""

    matches = []

    for fact in facts:
        if fact["ticker"] != ticker:
            continue

        if fact["metric"] != metric:
            continue

        if fact.get("fiscal_year") != fiscal_year:
            continue

        if fact.get("period_type") != period_type:
            continue

        if (
            fiscal_period is not None
            and fact.get("fiscal_period")
            != fiscal_period
        ):
            continue

        matches.append(fact)

    if not matches:
        return None

    # If more than one match exists, use the latest period.
    matches.sort(
        key=lambda fact: (
            fact["end"],
            fact["filed"],
        ),
        reverse=True,
    )

    return matches[0]


def calculate_margin(
    numerator: dict,
    revenue: dict,
) -> float:
    """Calculate a percentage margin."""

    if revenue["value"] == 0:
        raise ZeroDivisionError(
            "Revenue is zero; margin cannot be calculated."
        )

    if (
        numerator["end"] != revenue["end"]
        or numerator["period_type"]
        != revenue["period_type"]
    ):
        raise ValueError(
            "Numerator and revenue periods do not match."
        )

    return (
        numerator["value"]
        / revenue["value"]
        * 100
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate exact financial margins."
    )

    parser.add_argument(
        "--ticker",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--metric",
        required=True,
        choices=[
            "operating_margin",
            "net_margin",
        ],
    )

    parser.add_argument(
        "--fiscal-year",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--fiscal-period",
        choices=["Q1", "Q2", "Q3", "FY"],
    )

    parser.add_argument(
        "--period-type",
        required=True,
        choices=["quarter", "ytd", "annual"],
    )

    args = parser.parse_args()
    facts = load_facts()

    numerator_metric = {
        "operating_margin": "operating_income",
        "net_margin": "net_income",
    }[args.metric]

    results = []

    for raw_ticker in args.ticker:
        ticker = raw_ticker.upper()

        revenue = find_fact(
            facts=facts,
            ticker=ticker,
            metric="revenue",
            fiscal_year=args.fiscal_year,
            fiscal_period=args.fiscal_period,
            period_type=args.period_type,
        )

        numerator = find_fact(
            facts=facts,
            ticker=ticker,
            metric=numerator_metric,
            fiscal_year=args.fiscal_year,
            fiscal_period=args.fiscal_period,
            period_type=args.period_type,
        )

        if revenue is None or numerator is None:
            print(
                f"{ticker}: missing revenue or "
                f"{numerator_metric}"
            )
            continue

        margin = calculate_margin(
            numerator=numerator,
            revenue=revenue,
        )

        results.append(
            {
                "ticker": ticker,
                "company_name": revenue["company_name"],
                "margin": margin,
                "revenue": revenue,
                "numerator": numerator,
            }
        )

    results.sort(
        key=lambda result: result["margin"],
        reverse=True,
    )

    print(
        f"\nMetric: {args.metric.replace('_', ' ')}"
    )
    print(
        f"Fiscal year: {args.fiscal_year}"
    )
    print(f"Period type: {args.period_type}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        revenue_billions = (
            result["revenue"]["value"]
            / 1_000_000_000
        )

        numerator_billions = (
            result["numerator"]["value"]
            / 1_000_000_000
        )

        print("\n" + "=" * 72)
        print(f"Rank: {rank}")
        print(
            f"Company: {result['company_name']} "
            f"({result['ticker']})"
        )
        print(f"Margin: {result['margin']:.2f}%")
        print(
            f"Revenue: ${revenue_billions:,.3f}B"
        )
        print(
            f"{numerator_metric}: "
            f"${numerator_billions:,.3f}B"
        )
        print(
            "Period:",
            result["revenue"]["start"],
            "to",
            result["revenue"]["end"],
        )
        print(
            "Source:",
            result["revenue"]["source_url"],
        )


if __name__ == "__main__":
    main()