# Big Tech Financial Intelligence RAG

An evaluated RAG system that answers and cites comparative financial
questions across Apple, Microsoft, Alphabet, Amazon, and Meta — using their
own SEC filings and earnings call transcripts as the source of truth.

## Why this project

Most RAG portfolio projects are "chat with a PDF." This one is designed to
surface real, discussable engineering problems:

- **Comparative questions across documents** ("who had the highest operating
  margin?") that a single-passage retrieval can't answer — the system has to
  retrieve evidence from *all five* companies and reason over it.
- **Numbers vs. narrative**: the LLM never computes or invents financial
  figures. Numeric facts come from SEC's structured XBRL data; only
  *explanations* ("why did revenue change") come from retrieved text.
- **Mixed document types**: dense, formal 10-K/10-Q prose vs. conversational
  earnings-call transcripts — different chunking and retrieval behavior.
- **A real evaluation harness**: retrieval precision, answer faithfulness
  (does the answer's numbers actually match the cited source), and a
  measured "I don't know" rate for out-of-scope questions.

## Architecture

```
                     ┌─────────────────────┐
                     │   User question      │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┴───────────────┐
                 │        Query router           │
                 │  (single-co / comparative /   │
                 │   numeric / mixed)             │
                 └───────┬────────────┬──────────┘
                         │            │
              ┌──────────▼───┐   ┌────▼─────────────┐
              │  Text RAG     │   │  Structured XBRL  │
              │ (10-K/10-Q +  │   │  lookup (exact     │
              │  transcripts) │   │  numeric facts)     │
              └──────────┬───┘   └────┬─────────────┘
                         │            │
                 ┌───────▼────────────▼───────┐
                 │   Evidence assembly +        │
                 │   LLM answer generation      │
                 │   (cites source, period,     │
                 │    company for every claim)  │
                 └───────────────┬───────────────┘
                                 │
                         ┌───────▼────────┐
                         │  Answer + cites │
                         └─────────────────┘
```

## Project stages

- [x] **Stage 0 — Scaffold** (this commit): repo structure, company config,
      ingestion scripts for SEC filings + XBRL facts.
- [x] **Stage 1 — SEC corpus**: 2x 10-K + 3x 10-Q per company (25 filings),
      plus structured XBRL facts (revenue, operating income, R&D, capex, net
      income) for the same companies.
- [x] **Stage 2 — Parsing & chunking**: strip filing HTML down to clean text,
      chunk with overlap, preserve section/company/period metadata on every
      chunk.
- [x] **Stage 3 — Embedding & retrieval**: embed chunks into a vector store,
      build a retriever, manually sanity-check retrieval quality on ~15
      hand-picked queries before touching the LLM.
- [x] **Stage 4 — Generation**: wire retrieved evidence (+ XBRL facts for
      numeric questions) into an LLM prompt that must cite sources and
      decline to answer without sufficient evidence.
- [x] **Stage 5 — Eval harness**: build a 40-50 question eval set (single-
      company factual, cross-company comparative, trend, and
      out-of-scope/unanswerable), score retrieval precision@k and answer
      faithfulness.
- [x] **Stage 6 — Earnings call transcripts**: add transcripts, re-run eval,
      compare retrieval behavior on formal vs. conversational text.
- [x] **Stage 7 — Improve retrieval**: hybrid search (BM25 + semantic)
      and/or re-ranking; re-run eval, report before/after numbers.
- [x] **Stage 8 — Deploy**: Streamlit chat UI showing answer + sources.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running ingestion (Stage 1)

SEC requires a descriptive User-Agent identifying you — requests without
one are blocked. Use your real name/email.

```bash
python src/ingestion/sec_edgar.py --user-agent "Your Name your.email@example.com"
python src/ingestion/sec_xbrl.py  --user-agent "Your Name your.email@example.com"
```

This populates:
- `data/raw/<TICKER>/*.htm` — raw filing documents
- `data/processed/manifest.jsonl` — metadata for every filing pulled
- `data/processed/xbrl_facts.jsonl` — structured financial facts

**Note:** these scripts need real internet access to `sec.gov` and were
written and reviewed but not execution-tested in this environment (sandbox
network restrictions block sec.gov). Run them locally; if you hit errors,
the likely culprits are (a) User-Agent formatting, or (b) a company's XBRL
concept tag not matching one of the candidates in `CONCEPTS` — see the
comments in `src/ingestion/sec_xbrl.py`.
# Running ingestion (Stage 1)
## Repo structure

```
config/
  companies.py          # ticker -> CIK, sector, form limits
src/
  ingestion/
    sec_edgar.py         # pulls 10-K / 10-Q filing documents
    sec_xbrl.py           # pulls structured financial facts
  retrieval/              # (Stage 3) chunking, embedding, vector store
  generation/              # (Stage 4) prompt construction, LLM calls
  eval/                    # (Stage 5) eval harness, scoring
data/
  raw/                    # downloaded filing documents (gitignored)
  processed/               # manifest + structured facts (gitignored)
  eval/                    # hand-built eval question set
notebooks/                 # exploratory analysis
```

## What didn't work / open questions

(Keep this section updated as you build — it's the most valuable part of
the README for interviews.)

-
