"""
eCFR source: the official versioner API serves every CFR title as structured
XML for free, so there is no scraping and no PDF parsing.

Cost/ingest savers:
  - titles.json gives each title's latest_amended_on date, so unchanged
    titles are skipped before downloading anything.
  - Downloads are cached on disk keyed by issue date, so re-running after a
    crash or a chunking change never re-downloads.
"""
from pathlib import Path

import httpx

from regrag.config import ECFR_API, RAW_DIR

# The full-XML endpoint rejects requests that don't accept compression.
HEADERS = {"Accept-Encoding": "gzip, deflate"}


def list_titles() -> list[dict]:
    resp = httpx.get(f"{ECFR_API}/titles.json", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return [t for t in resp.json()["titles"] if not t.get("reserved")]


def download_title(number: int, date: str) -> Path:
    """Stream title XML to data/raw/title-{n}_{date}.xml; reuse if cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"title-{number}_{date}.xml"
    if path.exists() and path.stat().st_size > 0:
        return path

    # Old cached versions of this title are now stale.
    for old in RAW_DIR.glob(f"title-{number}_*.xml"):
        old.unlink()

    tmp = path.with_suffix(".part")
    url = f"{ECFR_API}/full/{date}/title-{number}.xml"
    with httpx.stream("GET", url, headers=HEADERS, timeout=httpx.Timeout(60, read=600)) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for block in resp.iter_bytes(1 << 20):
                f.write(block)
    tmp.replace(path)
    return path
