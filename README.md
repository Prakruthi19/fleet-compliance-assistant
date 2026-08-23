# Fleet Compliance Multi-Hop Assistant

Agentic, multi-hop RAG over public FMCSA trucking regulations. A LangGraph
ReAct-style agent decides whether one retrieval is enough to answer a
question, or whether it needs a second, reformulated retrieval to pull in a
related-but-separate regulation area before answering.

**Note on origin:** this project was deliberately scoped to demonstrate
multi-hop RAG as a technique — it wasn't derived by first observing that real
FMCSA questions require multi-hop lookup more often than not. The example
question below is a realistic, constructed case chosen because inspection
rules (Part 396) and driver-qualification rules (Part 391/383) genuinely do
live in separate regulation Parts, not a discovered production failure.

## Problem

FMCSA trucking regulations are split across many separate Parts (inspection,
driver qualification, hazardous-materials handling, CDL standards). A single
compliance question — like "what applies to a hazmat-hauling truck?" — often
needs content from two of these Parts at once. Standard single-pass RAG
retrieves for the literal query once and stops, which means it can silently
return an incomplete answer when the question is actually compositional.

## User

A fleet compliance officer or dispatcher checking what rules apply to a
specific load/vehicle/driver combination.

## Input / Output

**Input:** a natural-language compliance question.
**Output:** a generated answer citing the specific regulation section(s) it
drew from, plus (in the eval dashboard) visibility into whether the system
took one retrieval hop or two, and what it retrieved at each step.

## Demo

Run locally:
```bash
streamlit run app.py
```
- **Chat tab** — ask a question, see the answer, hop count, retrieved
  chunks, and (if a second hop happened) the reformulated query and which
  domain it targeted.
- **Eval Dashboard tab** — per-question single-hop vs multi-hop comparison,
  and a top_k sweep comparison table (see Evaluation below).

No hosted demo link — this requires your own Azure AI Search + Azure OpenAI
credentials in `.env` to run.

## Architecture

```
files/raw/*.txt (untouched eCFR source text, with a SOURCE: header)
        │
        ▼
files/chunk_corpus.py (deterministic, section-based chunker)
        │
        ▼
data/fmcsa_corpus.json (76 chunks: id, part, section, domain, source_url, title, text)
        │
        ▼
src/build_index.py (embeds via Azure OpenAI, uploads to Azure AI Search)
        │
        ▼
fleet-compliance-index (hybrid vector+keyword index)
        │
        ▼
src/retrieval.py (hybrid search, optional domain filter)
        │
        ▼
src/agent.py — LangGraph:
  retrieve_initial → decide_next_hop → [generate | retrieve_second_hop → generate]
```

- **Azure AI Search** (`fleet-compliance-index`) — hybrid (BM25 + vector)
  search, with an optional server-side domain filter for the second hop.
- **LangGraph** — orchestrates the ReAct-style reasoning loop.
- **Azure OpenAI** (`gpt-4o-mini` for generation, `text-embedding-3-small`
  for embeddings) — both embedding and generation calls go through Azure
  OpenAI.

## Project layout

```
app.py              Streamlit demo UI (chat + eval dashboard)
check_env.py        Sanity-checks that required .env variables are set
files/
  raw/*.txt         Untouched eCFR source text per regulation Part
  chunk_corpus.py   Section-based chunker (files/raw/ -> data/fmcsa_corpus.json)
data/
  fmcsa_corpus.json Chunked corpus consumed by src/build_index.py
src/
  build_index.py    Creates the Azure AI Search index and uploads embedded chunks
  retrieval.py      Hybrid (vector + keyword) search against the index
  agent.py          LangGraph agent (retrieval + hop decision + generation)
  requirements.txt  Python dependencies
eval/
  eval_set.json     20 hand-labeled questions (12 single-hop, 8 multi-hop)
  run_eval.py       Retrieval eval across a top_k sweep
  compare_hops.py   Single-hop vs multi-hop head-to-head comparison
```

## Data

76 chunks, hand-selected from 5 real eCFR Parts (383, 385 Subpart E, 391,
396, 397) — hazmat, inspection, and driver-qualification rules. Chunked
section-by-section (splitting on `§` markers), not fixed-size. This is a
targeted subset, not a full FMCSA regulation mirror — see Limitations.

## Decisions

**Chunking: section-based, not fixed-size.**
I chose to split on regulation section boundaries (`§ 396.17`, etc.) over
fixed-size chunking because FMCSA text has legally meaningful units — a
fixed-size splitter risks cutting a rule from its own exception mid-sentence.
The downside: chunk sizes are uneven (104 to 5,683 characters), so some
chunks carry much more retrieval weight than others. I accepted this because
losing a rule's own context seemed worse than uneven sizing. If the project
grew, I'd consider sub-section splitting for the largest chunks (e.g. the
Appendix A brake-defect list).

**top_k = 5, not wider.**
I tested top_k = 3, 5, and 7 on the eval set. Hit rate rose from 85% to 95%
between top_k=3 and 5, but going to 7 gave zero additional improvement on
either hit-rate metric while further degrading hop-decision accuracy (55%→
40% overall, 25%→12% on multi-hop questions specifically). I chose top_k=5
because it captured the accuracy gain without paying that extra cost. The
downside I accepted: at top_k=5, some multi-hop-labeled questions get
answered correctly in a single hop (the first search accidentally retrieves
both needed topics), which means my `hop_decision_correct` metric under-
reports actual system quality at this setting — see Evaluation.

**Dynamic domain filtering on the second hop, not a hardcoded filter.**
When I found the second hop was drifting into the wrong regulation domain
(see Failures), I added a `missing_domain` field to the decision node's
output, constrained to the three valid domain tags, and used it as a
server-side Azure filter on the second retrieval. I explicitly rejected
hardcoding a single filter value, since a fixed domain would fix one
question's failure while breaking any question whose second hop actually
needs a different domain. The downside: `VALID_DOMAINS` is currently
duplicated by hand between `src/agent.py` and `files/chunk_corpus.py`'s
`DOMAIN_BY_FILE` — not derived from a single source of truth, so it would
silently drift out of sync if a new domain were ever added. If the project
grew, I'd derive valid domains from the corpus file at runtime instead.

## Failures

**Failure 1 — the decision prompt was too lenient about "coverage."**
Early version: the decision node judged "do I have enough?" holistically,
with no structured criteria. On the compositional example question, it
retrieved hazmat + inspection content, judged that "enough," and generated
an answer that silently omitted driver-qualification rules — no error, no
warning, just an incomplete answer.
*How I found it:* ran the example question manually and read the generated
answer, which explicitly said it couldn't determine driver-qualification
rules from the retrieved text — the model was honest about the gap even
though the routing logic wasn't catching it upstream.
*What I changed:* rewrote the decision prompt to require listing distinct
required topics and checking each one individually against retrieved text,
rather than one vague judgment.

**Failure 2 — even after fixing the decision logic, the second hop still missed the target domain.**
*How I found it:* ran the 20-question eval set. Hop-decision accuracy was
only 50% on multi-hop-labeled questions — worse than the 90% overall
retrieval hit rate — meaning the reformulated second-hop query wasn't
reliably reaching the missing regulation Part even when the decision to
search again was correct.
*What I changed:* added a `missing_domain` field to the decision output and
applied it as a hard Azure AI Search filter on the second hop, so the second
retrieval is constrained to the correct domain rather than hoping the
reformulated query's wording drifts there naturally.

**Failure 3 (partially open) — domain filtering fixes cross-domain misses, not within-domain misses.**
After the domain-filter fix, the second hop correctly landed in the
`driver_qualification` domain — but that domain spans two different
regulation Parts (391 and 383, 29 chunks combined). On a retest, the second
hop retrieved only Part 383 chunks and still missed the specific Part 391
chunk (§391.25, annual driver record review) the question needed. The
domain filter narrows the search *pool* correctly; it doesn't guarantee the
reformulated query is specific enough to find the right chunk *within* that
pool. **Not yet fixed** — the next step would be making the reformulated
query name the specific missing sub-topic, not just the domain.

**Failure 4 — an eval metric turned out to be a misleading proxy.**
While testing top_k, I found that widening retrieval could make the
decision node correctly conclude "I already have enough" for questions
labeled multi-hop — meaning the system got the right answer in fewer,
cheaper steps. But `hop_decision_correct` compares hop count against a
fixed label, so this correct, more-efficient behavior was scored as
"wrong." The lesson: I now read hop-decision-accuracy alongside hit-rate,
not as a standalone quality signal — a process metric that doesn't account
for outcome can penalize genuinely good behavior.

## Evaluation

**Eval set:** 20 hand-written questions (12 single-hop, 8 multi-hop), each
labeled with the exact chunk id(s) needed to answer it. I authored both the
questions and the ground-truth labels myself — a question was labeled
multi-hop when no single chunk in the corpus contains everything needed to
answer it. This was not independently validated by a second reviewer, and 8
multi-hop examples is a small sample — one flipped result moves the
reported percentage by 12.5 points.

**Baseline (top_k=3):** 85% overall hit rate, 62% multi-hop hit rate, 65%
hop-decision accuracy, 50% multi-hop-specific hop-decision accuracy.

**top_k sweep (3 / 5 / 7):**

| top_k | overall hit | multi-hop hit | hop-decision acc. | multi-hop hop-decision acc. | multi-hop Qs solved in 1 hop |
|-------|-------------|----------------|--------------------|-------------------------------|-------------------------------|
| 3     | 85%         | 62%            | 65%                | 50%                           | 4/8                           |
| 5     | 95%         | 88%            | 55%                | 25%                           | 6/8                           |
| 7     | 95%         | 88%            | 40%                | 12%                           | 7/8                           |

top_k=5 was chosen as the working setting — see Decisions above for why 7
wasn't worth the additional hop-decision-accuracy cost given zero hit-rate
gain over 5.

**What I'd monitor after deployment:** retrieval hit rate against a larger,
independently-reviewed eval set; how often the second hop fires versus how
often it's actually needed (proxy for over/under-triggering); generation
faithfulness (does the answer only state what's in retrieved text, or does
it ever fill gaps from the model's own training knowledge) — not currently
measured here, only checked ad hoc by reading individual answers.

## Limitations

- **Narrow corpus.** 76 chunks across 5 of ~20 FMCSA regulation Parts. A
  question about hours-of-service (Part 395) or drug/alcohol testing
  (Part 382) would retrieve nothing relevant, since those Parts were never
  ingested.
- **No independent eval validation.** Ground-truth labels were written by
  the same person who built the system.
- **Domain filtering is coarse.** `driver_qualification` spans two
  regulation Parts (391, 383); filtering narrows the search pool but doesn't
  guarantee retrieval of the exact right chunk within that pool (Failure 3).
- **`VALID_DOMAINS` is hand-duplicated**, not derived from the corpus file —
  would silently drift out of sync if new domains were added without
  updating both places.
- **No answer-faithfulness evaluation.** The system is checked for
  retrieval correctness, not whether generated answers ever state something
  not actually supported by retrieved text.
- **Max two hops, hardcoded.** The agent can't take a third hop even if a
  question genuinely needs content from three separate domains.

## Links to key files

- Chunking strategy: [`files/chunk_corpus.py`](files/chunk_corpus.py)
- Agent orchestration: [`src/agent.py`](src/agent.py)
- Retrieval + domain filtering: [`src/retrieval.py`](src/retrieval.py)
- Index creation + upload: [`src/build_index.py`](src/build_index.py)
- Eval set: [`eval/eval_set.json`](eval/eval_set.json)
- top_k sweep script: [`eval/run_eval.py`](eval/run_eval.py)
- Single vs multi-hop comparison: [`eval/compare_hops.py`](eval/compare_hops.py)
- Demo UI: [`app.py`](app.py)

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r src/requirements.txt
```

Create a `.env` file in the project root with:

```
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=fleet-compliance-index
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

You'll need an Azure AI Search resource/index, and an Azure OpenAI resource
with both a chat model (e.g. `gpt-4o-mini`) and an embedding model (e.g.
`text-embedding-3-small`) deployed.

Optionally run `python check_env.py` to confirm your `.env` is set up
(note: it also checks `GITHUB_MODELS_*` variables left over from an earlier
iteration — those aren't used by the current agent/retrieval code).

## Run

```bash
# 1. Chunk the raw eCFR text (only needed if you edit files/raw/*.txt)
python files/chunk_corpus.py

# 2. Build the index and upload the corpus
python src/build_index.py

# 3. Try the agent on the example multi-hop question
python src/agent.py

# 4. Run the retrieval eval across a top_k sweep
python eval/run_eval.py

# 5. Compare single-hop vs multi-hop head-to-head
python eval/compare_hops.py

# 6. Launch the demo UI
streamlit run app.py
```
