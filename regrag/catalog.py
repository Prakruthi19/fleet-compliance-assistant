"""
What the router chooses from: every indexed CFR title, described by its name
plus its chapter headings. Title names alone are too vague to route on
("Title 49 — Transportation" doesn't say it holds FMCSA trucking rules);
chapter headings name the agencies, which is what questions usually map to.

Built once from the cached XML and saved to data/catalog.json.
"""
import json
import re
import sqlite3
from functools import lru_cache

from lxml import etree

from regrag.config import DATA_DIR, RAW_DIR, STATE_DB
from regrag.ingest.ecfr import list_titles

CATALOG = DATA_DIR / "catalog.json"
_ws = re.compile(r"\s+")


def _chapters(path) -> list[str]:
    heads = []
    for _, el in etree.iterparse(str(path), events=("end",), tag="HEAD", huge_tree=True):
        parent = el.getparent()
        if parent is not None and parent.get("TYPE") == "CHAPTER":
            heads.append(_ws.sub(" ", "".join(el.itertext())).strip())
        el.clear()
    return heads


def indexed_titles() -> list[int]:
    db = sqlite3.connect(STATE_DB)
    return sorted(n for (n,) in db.execute("SELECT number FROM titles"))


def build() -> dict:
    names = {t["number"]: t["name"] for t in list_titles()}
    catalog = {}
    for n in indexed_titles():
        xml = next(RAW_DIR.glob(f"title-{n}_*.xml"), None)
        catalog[str(n)] = {"name": names.get(n, ""), "chapters": _chapters(xml) if xml else []}
    CATALOG.write_text(json.dumps(catalog, indent=1))
    load.cache_clear()
    return catalog


@lru_cache
def load() -> dict[int, dict]:
    if not CATALOG.exists():
        build()
    return {int(k): v for k, v in json.loads(CATALOG.read_text()).items()}


def describe() -> str:
    lines = []
    for n, t in sorted(load().items()):
        chapters = "; ".join(c.split("—", 1)[-1].strip().title() for c in t["chapters"])
        lines.append(f"Title {n} — {t['name']}: {chapters}")
    return "\n".join(lines)


if __name__ == "__main__":
    build()
    print(describe())
