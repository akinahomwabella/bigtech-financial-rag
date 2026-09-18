"""
Company registry for the Big Tech Financial Intelligence RAG project.

CIK = SEC's Central Index Key, a unique company identifier used by EDGAR.
These must be zero-padded to 10 digits when used in SEC API URLs.

Source of truth for CIKs: https://www.sec.gov/cgi-bin/browse-edgar
or https://www.sec.gov/files/company_tickers.json
"""

COMPANIES = {
    "AAPL": {
        "name": "Apple Inc.",
        "cik": "0000320193",
        "sector": "Consumer Tech / Devices",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "cik": "0000789019",
        "sector": "Cloud / Enterprise Software",
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "cik": "0001652044",
        "sector": "Search / Advertising / Cloud",
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "cik": "0001018724",
        "sector": "E-commerce / Cloud (AWS)",
    },
    "META": {
        "name": "Meta Platforms, Inc.",
        "cik": "0001326801",
        "sector": "Social / Advertising",
    },
}

# Forms we care about for Version 1 of the corpus
TARGET_FORMS = ["10-K", "10-Q"]

# How many of each form to pull per company (most recent first)
FORM_LIMITS = {
    "10-K": 2,
    "10-Q": 3,
}
