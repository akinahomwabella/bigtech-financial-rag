"""
Classify financial questions for the hybrid RAG system.

Routes:
    numeric   -> structured XBRL facts
    narrative -> vector search over filing chunks
    mixed     -> both paths
"""

from __future__ import annotations

import argparse
import json
import re


COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["alphabet", "google", "googl"],
    "AMZN": ["amazon", "amzn"],
    "META": ["meta", "facebook"],
}


METRIC_PATTERNS = {
    "operating_margin": [
        "operating margin",
    ],
    "net_margin": [
        "net margin",
        "profit margin",
    ],
    "operating_income": [
        "operating income",
        "income from operations",
    ],
    "net_income": [
        "net income",
        "net earnings",
        "profit",
    ],
    "rnd_expense": [
        "research and development",
        "r&d",
        "rnd",
    ],
    "capex": [
        "capital expenditures",
        "capital expenditure",
        "capex",
    ],
    "revenue": [
        "revenue",
        "sales",
        "net sales",
    ],
}


NUMERIC_TERMS = {
    "what was",
    "what were",
    "how much",
    "amount",
    "value",
    "margin",
    "highest",
    "lowest",
    "rank",
    "compare",
    "comparison",
    "percentage",
    "percent",
    "increase by",
    "decrease by",
}


NARRATIVE_TERMS = {
    "why",
    "explain",
    "reason",
    "reasons",
    "driver",
    "drivers",
    "factor",
    "factors",
    "discuss",
    "describe",
    "risk",
    "risks",
    "outlook",
    "management said",
    "management explained",
    "affected",
    "impact",
}


COMPARATIVE_TERMS = {
    "compare",
    "comparison",
    "versus",
    " vs ",
    "highest",
    "lowest",
    "most",
    "least",
    "rank",
    "ranking",
    "which company",
    "among",
}


def detect_tickers(question: str) -> list[str]:
    """Find company tickers mentioned in the question."""

    question_lower = question.lower()
    tickers = []

    for ticker, aliases in COMPANY_ALIASES.items():
        if any(
            re.search(
                rf"\b{re.escape(alias)}\b",
                question_lower,
            )
            for alias in aliases
        ):
            tickers.append(ticker)

    # Phrases such as "all companies" mean all five.
    if any(
        phrase in question_lower
        for phrase in [
            "all companies",
            "all five",
            "big tech companies",
            "which company",
            "among the companies",
        ]
    ):
        return list(COMPANY_ALIASES)

    return tickers


def detect_metric(question: str) -> str | None:
    """Detect the requested financial metric."""

    question_lower = question.lower()

    # Dictionary order ensures operating margin is checked
    # before the more general operating-income concept.
    for metric, phrases in METRIC_PATTERNS.items():
        if any(
            phrase in question_lower
            for phrase in phrases
        ):
            return metric

    return None


def detect_fiscal_year(question: str) -> int | None:
    """Extract a four-digit fiscal year."""

    match = re.search(
        r"\b(20\d{2})\b",
        question,
    )

    if match:
        return int(match.group(1))

    return None


def detect_fiscal_period(question: str) -> str | None:
    """Detect Q1, Q2, Q3, or FY."""

    question_upper = question.upper()

    quarter_match = re.search(
        r"\bQ([1-3])\b",
        question_upper,
    )

    if quarter_match:
        return f"Q{quarter_match.group(1)}"

    if any(
        phrase in question.lower()
        for phrase in [
            "full year",
            "fiscal year",
            "annual",
            "yearly",
        ]
    ):
        return "FY"

    return None


def detect_period_type(
    question: str,
    fiscal_period: str | None,
) -> str | None:
    """Determine annual, quarter, or YTD period type."""

    question_lower = question.lower()

    if any(
        phrase in question_lower
        for phrase in [
            "year to date",
            "year-to-date",
            "ytd",
            "nine months",
            "six months",
        ]
    ):
        return "ytd"

    if fiscal_period == "FY":
        return "annual"

    if fiscal_period in {"Q1", "Q2", "Q3"}:
        return "quarter"

    if any(
        phrase in question_lower
        for phrase in [
            "annual",
            "full year",
            "fiscal year",
        ]
    ):
        return "annual"

    return None


def contains_term(
    question: str,
    terms: set[str],
) -> bool:
    """Check whether a question contains any routing term."""

    question_lower = question.lower()

    return any(
        term in question_lower
        for term in terms
    )


def route_question(question: str) -> dict:
    """Classify the question and extract metadata."""

    tickers = detect_tickers(question)
    metric = detect_metric(question)
    fiscal_year = detect_fiscal_year(question)
    fiscal_period = detect_fiscal_period(question)

    period_type = detect_period_type(
        question=question,
        fiscal_period=fiscal_period,
    )

    numeric_signal = (
        metric is not None
        or contains_term(question, NUMERIC_TERMS)
    )

    narrative_signal = contains_term(
        question,
        NARRATIVE_TERMS,
    )

    if numeric_signal and narrative_signal:
        route = "mixed"
    elif numeric_signal:
        route = "numeric"
    else:
        route = "narrative"

    comparative = (
        len(tickers) > 1
        or contains_term(
            question,
            COMPARATIVE_TERMS,
        )
    )

    return {
        "question": question,
        "route": route,
        "comparative": comparative,
        "tickers": tickers,
        "metric": metric,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "period_type": period_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route a financial question."
    )

    parser.add_argument(
        "question",
        help="Financial question to classify",
    )

    args = parser.parse_args()

    result = route_question(args.question)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()