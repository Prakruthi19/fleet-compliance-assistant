"""
Obligations finder: business profile -> checklist of plain-English requirement
cards, each backed by cited CFR sections.

Cards are generated ONCE (build) and served from data/obligations.json, so the
UI is instant and costs no LLM quota per visitor. Each topic is a curated
search scoped to the CFR parts that govern it; the LLM only rewrites what the
retrieved sections say into a structured card.

    python -m regrag.obligations build [--profile trucking] [--only hos]
"""
import argparse
import json
import sys
from datetime import datetime, timezone

from regrag import llm
from regrag.config import DATA_DIR
from regrag.rag import citation, ecfr_url
from regrag.retrieval import search

OUT = DATA_DIR / "obligations.json"
SOURCES_PER_CARD = 5

# `when` names a profile question; the card only applies if the user answers yes.
PROFILES = [
    {
        "id": "trucking", "name": "Trucking company", "title": 49,
        "blurb": "You operate commercial trucks or buses across state lines.",
        "questions": [{"id": "hazmat", "label": "We transport hazardous materials"}],
        "topics": [
            {"id": "usdot", "name": "Register and mark your vehicles", "parts": ["390"],
             "query": "USDOT number registration identification marking of commercial motor vehicles"},
            {"id": "dq", "name": "Keep a qualification file for every driver", "parts": ["391"],
             "query": "driver qualification file contents annual review of driving record medical certificate"},
            {"id": "cdl", "name": "Make sure drivers hold a valid CDL", "parts": ["383"],
             "query": "employer must not allow a driver to operate without a valid commercial driver's license"},
            {"id": "hos", "name": "Limit and log driving hours", "parts": ["395"],
             "query": ["maximum driving time property-carrying vehicles 11 hours 14 hour period 60 70 hours",
                       "drivers record of duty status every 24 hour period motor carrier"]},
            {"id": "drug", "name": "Run a drug and alcohol testing program", "parts": ["382"],
             "query": "employer pre-employment random drug and alcohol testing requirements"},
            {"id": "maint", "name": "Inspect and maintain your vehicles", "parts": ["396"],
             "query": "systematic inspection repair maintenance records periodic annual inspection driver vehicle inspection report"},
            {"id": "insurance", "name": "Carry the minimum insurance", "parts": ["387"],
             "query": "minimum levels of financial responsibility insurance for motor carriers"},
            {"id": "hazmat", "name": "Hazmat shipping papers and placards", "parts": ["172", "177"],
             "when": "hazmat", "query": "shipping papers placarding hazardous materials transported by highway"},
        ],
    },
    {
        "id": "employer", "name": "Employer", "title": 29,
        "blurb": "You have employees in the United States.",
        "questions": [{"id": "ten_plus", "label": "We have more than 10 employees"},
                      {"id": "fifty_plus", "label": "We have 50 or more employees"},
                      {"id": "chemicals", "label": "Workers handle hazardous chemicals"}],
        "topics": [
            {"id": "payroll", "name": "Keep payroll and hours records", "parts": ["516"],
             "query": "records employer must keep of wages and hours worked"},
            {"id": "overtime", "name": "Pay overtime over 40 hours a week", "parts": ["778"],
             "query": "overtime compensation one and one-half times regular rate for hours over 40 in a workweek"},
            {"id": "injury_log", "name": "Record work injuries and illnesses", "parts": ["1904"],
             "when": "ten_plus", "query": "OSHA 300 log recording work-related injuries and illnesses"},
            {"id": "injury_report", "name": "Report serious injuries to OSHA", "parts": ["1904"],
             "query": "report fatality in-patient hospitalization amputation or loss of an eye to OSHA within hours"},
            {"id": "hazcom", "name": "Warn and train workers about chemicals", "parts": ["1910"],
             "when": "chemicals", "query": "hazard communication program safety data sheets labels employee training"},
            {"id": "fmla", "name": "Give family and medical leave", "parts": ["825"],
             "when": "fifty_plus", "query": "covered employer eligible employee 12 workweeks of FMLA leave notice"},
            {"id": "eeo_records", "name": "Keep hiring and personnel records", "parts": ["1602"],
             "query": "preservation of personnel and employment records retention period"},
        ],
    },
    {
        "id": "food", "name": "Food or supplement maker", "title": 21,
        "blurb": "You make, pack or hold food or dietary supplements.",
        "questions": [{"id": "supplements", "label": "We make dietary supplements"}],
        "topics": [
            {"id": "register", "name": "Register your facility with FDA", "parts": ["1"],
             "query": "registration of food facilities who must register and when"},
            {"id": "food_gmp", "name": "Follow food safety manufacturing practices", "parts": ["117"],
             "query": "current good manufacturing practice hazard analysis risk-based preventive controls for human food"},
            {"id": "food_label", "name": "Label your food correctly", "parts": ["101"],
             "query": "food label principal display panel ingredient list net quantity nutrition facts"},
            {"id": "supp_gmp", "name": "Follow supplement manufacturing practices", "parts": ["111"],
             "when": "supplements", "query": "dietary supplement current good manufacturing practice quality control records"},
            {"id": "supp_label", "name": "Put a Supplement Facts panel on the label", "parts": ["101"],
             "when": "supplements", "query": "dietary supplement labeling supplement facts panel"},
        ],
    },
    {
        "id": "facility", "name": "Industrial facility", "title": 40,
        "blurb": "You store or use chemicals, oil or hazardous waste on site.",
        "questions": [{"id": "oil", "label": "We store oil above ground"},
                      {"id": "waste", "label": "We generate hazardous waste"}],
        "topics": [
            {"id": "release", "name": "Report hazardous substance releases", "parts": ["302"],
             "query": "notification of release of a hazardous substance at or above reportable quantity to the National Response Center"},
            {"id": "tier2", "name": "Report your chemical inventory every year", "parts": ["370"],
             "query": "hazardous chemical inventory form tier II submit annually by March 1"},
            {"id": "tri", "name": "Report toxic chemical releases", "parts": ["372"],
             "query": "toxic chemical release reporting form R threshold who must report"},
            {"id": "waste", "name": "Manage hazardous waste you generate", "parts": ["262"],
             "when": "waste", "query": "hazardous waste generator category determination and requirements"},
            {"id": "spcc", "name": "Have an oil spill prevention plan", "parts": ["112"],
             "when": "oil", "query": "spill prevention control and countermeasure plan aboveground oil storage capacity"},
        ],
    },
]

CARD_SYSTEM = """You turn US federal regulations into a plain-English \
compliance card for a small-business owner. Use ONLY the numbered sources. \
Write short, everyday sentences; no legal jargon. Every requirement must name \
the one source number it comes from. If the sources don't cover something, \
leave that field empty rather than guessing."""

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One sentence: what this rule means for the business."},
        "applies_to": {"type": "string", "description": "Who or what this applies to, in a few words."},
        "requirements": {
            "type": "array", "description": "2 to 5 concrete things the business must do.",
            "items": {"type": "object",
                      "properties": {"text": {"type": "string"}, "source": {"type": "integer"}},
                      "required": ["text", "source"]},
        },
        "how_often": {"type": "string", "description": "Deadline or frequency, e.g. 'Every 12 months'. Empty if none."},
        "records": {"type": "array", "items": {"type": "string"},
                    "description": "Records to keep, each a few words. Empty if none."},
    },
    "required": ["summary", "applies_to", "requirements", "how_often", "records"],
}


def _source(h: dict) -> dict:
    return {"citation": citation(h), "heading": h["heading"], "breadcrumb": h["breadcrumb"],
            "url": ecfr_url(h), "text": h["text"], "chunk_id": h["chunk_id"]}


def _retrieve(profile: dict, topic: dict) -> list[dict]:
    """Top sources for the topic; several queries are interleaved so each sub-rule is covered."""
    queries = topic["query"] if isinstance(topic["query"], list) else [topic["query"]]
    lists = [search(q, SOURCES_PER_CARD, titles=[profile["title"]], parts=topic["parts"]) for q in queries]
    hits, seen = [], set()
    for rank in range(SOURCES_PER_CARD):
        for lst in lists:
            if rank < len(lst) and lst[rank]["chunk_id"] not in seen and len(hits) < SOURCES_PER_CARD:
                seen.add(lst[rank]["chunk_id"])
                hits.append(lst[rank])
    return hits


def build_card(profile: dict, topic: dict) -> dict:
    hits = _retrieve(profile, topic)
    if not hits:
        raise RuntimeError(f"no sources found for {profile['id']}/{topic['id']}")
    ctx = "\n\n".join(f"[{i}] {citation(h)} — {h['heading']}\n{h['text']}" for i, h in enumerate(hits, 1))
    card = llm.chat_json([
        {"role": "system", "content": CARD_SYSTEM},
        {"role": "user", "content": f"Sources:\n{ctx}\n\nTopic: {topic['name']} ({profile['name']})"},
    ], CARD_SCHEMA)
    # Drop requirements whose citation doesn't point at a real source.
    card["requirements"] = [r for r in card.get("requirements", [])
                            if isinstance(r.get("source"), int) and 1 <= r["source"] <= len(hits)]
    return {"id": topic["id"], "name": topic["name"], "when": topic.get("when"),
            **card, "sources": [_source(h) for h in hits]}


def load() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"profiles": []}


def build(only_profile: str | None = None, only_topic: str | None = None) -> None:
    existing = {p["id"]: {c["id"]: c for c in p["cards"]} for p in load()["profiles"]}
    profiles = []
    for p in PROFILES:
        cards = []
        for t in p["topics"]:
            old = existing.get(p["id"], {}).get(t["id"])
            if (only_profile and p["id"] != only_profile) or (only_topic and t["id"] != only_topic):
                if old:
                    cards.append(old)
                continue
            try:
                cards.append(build_card(p, t))
                print(f"built {p['id']}/{t['id']}", flush=True)
            except Exception as e:  # keep the previous card rather than lose it
                print(f"FAILED {p['id']}/{t['id']}: {e!r}", file=sys.stderr, flush=True)
                if old:
                    cards.append(old)
        profiles.append({k: p[k] for k in ("id", "name", "title", "blurb", "questions")} | {"cards": cards})
    OUT.write_text(json.dumps({"built_at": datetime.now(timezone.utc).isoformat(), "profiles": profiles},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--profile")
    ap.add_argument("--only", help="topic id")
    args = ap.parse_args()
    build(args.profile, args.only)


if __name__ == "__main__":
    main()
