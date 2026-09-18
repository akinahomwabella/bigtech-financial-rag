"""
Generate grounded answers using hybrid SEC evidence and local Ollama.

The model must use only retrieved SEC evidence. Exact financial
numbers come from XBRL, not narrative text.

Example:
    python src/generation/generate_answer.py \
        "Why did Microsoft's revenue increase in fiscal year 2026?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"

sys.path.insert(
    0,
    str(RETRIEVAL_DIR),
)

from hybrid_retriever import hybrid_retrieve


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:4b"

MAX_NARRATIVE_CHARACTERS = 2500


SYSTEM_PROMPT = """
You are a grounded financial research assistant.

Answer the user's question using only the supplied SEC evidence.

Rules:
1. Never invent a financial value, date, cause, or comparison.
2. Financial numbers must come from NUMERIC EVIDENCE.
3. Use NARRATIVE EVIDENCE only for explanations and management commentary.
4. Cite every material claim using its evidence label, such as [N1] or [T2].
5. Preserve the distinction between quarterly, year-to-date, and annual data.
6. If the evidence is insufficient, clearly say what is missing.
7. Do not perform unsupported calculations.
8. Keep the answer concise and direct.
9. Do not include a separate sources section; the application adds it.
""".strip()


def format_number(
    value: float,
    unit: str,
) -> str:
    """Format financial values for the prompt."""

    if unit == "percent":
        return f"{value:.2f}%"

    if unit == "USD":
        absolute_value = abs(value)

        if absolute_value >= 1_000_000_000:
            return (
                f"${value / 1_000_000_000:,.3f}B"
            )

        if absolute_value >= 1_000_000:
            return (
                f"${value / 1_000_000:,.3f}M"
            )

        return f"${value:,.0f}"

    return f"{value} {unit}"


def build_numeric_evidence(
    facts: list[dict],
) -> tuple[str, list[dict]]:
    """Format structured evidence with citation labels."""

    if not facts:
        return "No numeric evidence retrieved.", []

    lines = []
    sources = []

    for index, fact in enumerate(facts, start=1):
        label = f"N{index}"

        value = format_number(
            value=fact["value"],
            unit=fact["unit"],
        )

        line = (
            f"[{label}] "
            f"Company: {fact['company_name']} "
            f"({fact['ticker']}); "
            f"Metric: {fact['metric']}; "
            f"Value: {value}; "
            f"Period: {fact.get('start')} to "
            f"{fact.get('end')}; "
            f"Fiscal year: "
            f"{fact.get('fiscal_year')}; "
            f"Fiscal period: "
            f"{fact.get('fiscal_period')}; "
            f"Period type: "
            f"{fact.get('period_type')}."
        )

        if "revenue" in fact:
            line += (
                f" Revenue used: "
                f"${fact['revenue'] / 1_000_000_000:,.3f}B;"
                f" {fact['numerator_metric']} used: "
                f"${fact['numerator_value'] / 1_000_000_000:,.3f}B."
            )

        lines.append(line)

        sources.append(
            {
                "label": label,
                "ticker": fact["ticker"],
                "accession_number": fact[
                    "accession_number"
                ],
                "source_url": fact["source_url"],
            }
        )

    return "\n".join(lines), sources


def build_narrative_evidence(
    chunks: list[dict],
) -> tuple[str, list[dict]]:
    """Format narrative evidence with citation labels."""

    if not chunks:
        return "No narrative evidence retrieved.", []

    lines = []
    sources = []

    for index, chunk in enumerate(chunks, start=1):
        label = f"T{index}"

        text = " ".join(
            chunk["text"].split()
        )

        text = text[
            :MAX_NARRATIVE_CHARACTERS
        ]

        lines.append(
            f"[{label}] "
            f"Company: {chunk['company_name']} "
            f"({chunk['ticker']}); "
            f"Form: {chunk['form']}; "
            f"Filing date: {chunk['filing_date']}; "
            f"Report date: {chunk['report_date']}; "
            f"Section: {chunk['section']}; "
            f"Similarity: {chunk['similarity']}.\n"
            f"Text: {text}"
        )

        sources.append(
            {
                "label": label,
                "ticker": chunk["ticker"],
                "section": chunk["section"],
                "accession_number": chunk[
                    "accession_number"
                ],
                "source_url": chunk["source_url"],
            }
        )

    return "\n\n".join(lines), sources


def call_ollama(prompt: str) -> str:
    """Send a grounded prompt to the local Ollama model."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 600,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300,
        )
    except requests.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Run "
            "`ollama serve` in another terminal."
        ) from error

    response.raise_for_status()

    result = response.json()

    answer = (
        result.get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty answer."
        )

    return answer


def generate_answer(
    question: str,
    top_k: int,
) -> dict:
    """Retrieve SEC evidence and generate an answer."""

    retrieval = hybrid_retrieve(
        question=question,
        top_k=top_k,
    )

    numeric_text, numeric_sources = (
        build_numeric_evidence(
            retrieval["numeric_evidence"]
        )
    )

    narrative_text, narrative_sources = (
        build_narrative_evidence(
            retrieval["narrative_evidence"]
        )
    )

    if (
        not retrieval["numeric_evidence"]
        and not retrieval["narrative_evidence"]
    ):
        raise ValueError(
            "No evidence was retrieved for this question."
        )

    prompt = f"""
QUESTION:
{question}

ROUTING RESULT:
{json.dumps(retrieval["route"], indent=2)}

NUMERIC EVIDENCE:
{numeric_text}

NARRATIVE EVIDENCE:
{narrative_text}

Write a grounded answer to the question. Cite claims using the
evidence labels. If evidence is incomplete, say so explicitly.
""".strip()

    answer = call_ollama(prompt)

    return {
        "question": question,
        "route": retrieval["route"],
        "answer": answer,
        "sources": (
            numeric_sources
            + narrative_sources
        ),
    }


def print_result(result: dict) -> None:
    """Print the answer and deterministic source list."""

    print("\nAnswer")
    print("=" * 80)
    print(result["answer"])

    print("\nSources")
    print("=" * 80)

    seen = set()

    for source in result["sources"]:
        key = (
            source["label"],
            source["source_url"],
        )

        if key in seen:
            continue

        seen.add(key)

        description = (
            f"[{source['label']}] "
            f"{source['ticker']} "
            f"{source['accession_number']}"
        )

        if source.get("section"):
            description += (
                f" — {source['section']}"
            )

        print(description)
        print(source["source_url"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grounded SEC filing answer "
            "using local Ollama."
        )
    )

    parser.add_argument(
        "question",
        help="Financial question to answer",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of narrative chunks to retrieve",
    )

    args = parser.parse_args()

    result = generate_answer(
        question=args.question,
        top_k=args.top_k,
    )

    print_result(result)


if __name__ == "__main__":
    main()