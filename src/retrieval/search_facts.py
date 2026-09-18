"""
Search exact financial facts from xbrl_facts.jsonl.

Examples:
    python src/retrieval/search_facts.py \
        --ticker MSFT \
        --metric revenue \
        --fiscal-year 2026 \
        --period-type annual

    python src/retrieval/search_facts.py \
        --ticker AAPL MSFT GOOGL AMZN META \
        --metric operating_income \
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

VALID_METRICS = {
    "revenue",
    "operating_income",
    "net_income",
    "rnd_expense",
    "capex",
}


def load_facts() -> list[dict]:
    """Load structured SEC facts."""

    if not FACTS_PATH.exists():
        raise FileNotFoundError(
            f"XBRL file was not found: {FACTS_PATH}"
        )

    facts = []

    with FACTS_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                fact = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            facts.append(fact)

    return facts


def format_value(value: float, unit: str) -> str:
    """Format large financial values for display."""

    if unit == "USD":
        absolute_value = abs(value)

        if absolute_value >= 1_000_000_000:
            return f"${value / 1_000_000_000:,.3f}B"

        if absolute_value >= 1_000_000:
            return f"${value / 1_000_000:,.3f}M"

        return f"${value:,.0f}"

    return f"{value:,.2f} {unit}"


def search_facts(
    tickers: list[str],
    metric: str,
    fiscal_year: int | None,
    fiscal_period: str | None,
    period_type: str | None,
) -> list[dict]:
    """Filter exact facts using financial metadata."""

    ticker_set = {
        ticker.upper()
        for ticker in tickers
    }

    matches = []

    for fact in load_facts():
        if fact["ticker"] not in ticker_set:
            continue

        if fact["metric"] != metric:
            continue

        if (
            fiscal_year is not None
            and fact.get("fiscal_year") != fiscal_year
        ):
            continue

        if (
            fiscal_period is not None
            and fact.get("fiscal_period")
            != fiscal_period
        ):
            continue

        if (
            period_type is not None
            and fact.get("period_type")
            != period_type
        ):
            continue

        matches.append(fact)

    matches.sort(
        key=lambda fact: (
            fact["ticker"],
            fact["end"],
            fact["period_type"],
        )
    )

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search structured SEC XBRL facts."
    )

    parser.add_argument(
        "--ticker",
        nargs="+",
        required=True,
        help="One or more company tickers",
    )

    parser.add_argument(
        "--metric",
        required=True,
        choices=sorted(VALID_METRICS),
    )

    parser.add_argument(
        "--fiscal-year",
        type=int,
    )

    parser.add_argument(
        "--fiscal-period",
        choices=["Q1", "Q2", "Q3", "FY"],
    )

    parser.add_argument(
        "--period-type",
        choices=["quarter", "ytd", "annual", "instant"],
    )

    args = parser.parse_args()

    results = search_facts(
        tickers=args.ticker,
        metric=args.metric,
        fiscal_year=args.fiscal_year,
        fiscal_period=args.fiscal_period,
        period_type=args.period_type,
    )

    print(f"Matches: {len(results)}")

    for fact in results:
        print("\n" + "=" * 72)
        print(
            f"{fact['company_name']} "
            f"({fact['ticker']})"
        )
        print(f"Metric: {fact['metric']}")
        print(
            "Value:",
            format_value(
                fact["value"],
                fact["unit"],
            ),
        )
        print(
            f"Period: {fact['start']} "
            f"to {fact['end']}"
        )
        print(
            f"Fiscal period: "
            f"{fact['fiscal_year']} "
            f"{fact['fiscal_period']}"
        )
        print(
            f"Period type: "
            f"{fact['period_type']}"
        )
        print(f"Form: {fact['form']}")
        print(
            f"Accession: "
            f"{fact['accession_number']}"
        )
        print(f"Source: {fact['source_url']}")


if __name__ == "__main__":
    main()