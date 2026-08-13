"""
LangGraph agent for the Fleet Compliance Multi-Hop Assistant.

Flow:
    retrieve_initial -> decide_next_hop -> [generate | retrieve_second_hop -> generate]

The agent reasons about whether the question is fully answerable from the
first retrieval, or whether it's compositional (spans two regulation areas,
e.g. hazmat rules + inspection/driver qualification rules) and needs a
second, reformulated query before it can generate a complete answer.

Generation uses Azure OpenAI (not GitHub Models) so this project
demonstrates direct Azure OpenAI usage alongside Azure AI Search.
"""
import os
import json
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI

from retrieval import retrieve

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    temperature=0,
)

# Default retrieval breadth if not overridden per-call. ask(..., top_k=N)
# takes priority over this — see the ask() function below.
DEFAULT_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "3"))


class AgentState(TypedDict):
    question: str
    retrieved_chunks: list[dict]
    needs_second_hop: bool
    reformulated_query: Optional[str]
    missing_domain: Optional[str]
    hop_count: int
    answer: Optional[str]
    force_single_hop: bool
    top_k: int


def retrieve_initial(state: AgentState) -> AgentState:
    chunks = retrieve(state["question"], top_k=state["top_k"])
    return {**state, "retrieved_chunks": chunks, "hop_count": 1}


VALID_DOMAINS = {"hazmat", "inspection", "driver_qualification"}

DECISION_PROMPT = """You are deciding whether you have enough information to \
answer an FMCSA trucking-regulation question, or whether you need a second \
retrieval to pull in a related-but-separate regulation area.

Question: {question}

Retrieved so far:
{context}

Follow these steps explicitly:
1. List the distinct regulatory topics this question requires (e.g. "hazmat \
routing", "inspection interval", "driver qualification", "CDL endorsement"). \
A question can require 1, 2, or more distinct topics.
2. For EACH topic listed, check whether the retrieved text above actually \
contains content addressing it. Be strict: a chunk about inspection does \
NOT count as covering driver qualification, even if both relate to trucking.
3. If any listed topic has zero supporting chunks, you need a second hop.
4. If you need a second hop, identify which ONE of these three exact domain \
values the missing topic belongs to: "hazmat", "inspection", or \
"driver_qualification". Use exactly one of these three strings, lowercase, \
with the underscore as shown — this value is used as a database filter, so \
any other spelling or casing will return zero results.

FMCSA questions are often compositional: a single question can require \
combining hazardous-materials rules with inspection or driver-qualification \
rules, which live in different Parts of the regulations and are rarely \
covered by one retrieval. Do not assume a topic is covered just because a \
retrieved chunk is thematically nearby (e.g. hazmat permit chunks do not \
cover driver medical certification).

Respond ONLY with JSON, no other text:
{{"topics_required": ["list", "of", "distinct", "topics"], \
"topics_covered": ["list", "of", "topics", "actually", "found", "in", "retrieved", "text"], \
"needs_second_hop": true or false, "reformulated_query": "a short, \
specific search query targeting the missing topic, or null if not needed", \
"missing_domain": "hazmat" or "inspection" or "driver_qualification" or null, \
"reasoning": "one sentence"}}
"""


def decide_next_hop(state: AgentState) -> AgentState:
    context = "\n\n".join(
        f"[{c['section']}] {c['title']}: {c['text']}" for c in state["retrieved_chunks"]
    )
    prompt = DECISION_PROMPT.format(question=state["question"], context=context)
    response = llm.invoke(prompt)

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        decision = {"needs_second_hop": False, "reformulated_query": None}

    # Validate missing_domain against the known valid set. An LLM can return
    # a slightly-off string (wrong casing, wrong spelling) — using that
    # directly as an Azure filter would silently return zero results rather
    # than erroring, so we fall back to "no filter" instead of trusting it.
    missing_domain = decision.get("missing_domain")
    if missing_domain not in VALID_DOMAINS:
        if missing_domain is not None:
            print(f"  [decide_next_hop] WARNING: missing_domain='{missing_domain}' "
                  f"is not a valid domain {VALID_DOMAINS} — ignoring filter")
        missing_domain = None

    # Printed for visibility during dev/tuning — shows exactly which topics
    # the decision node thought were required vs. covered.
    print(f"  [decide_next_hop] topics_required={decision.get('topics_required')} "
          f"topics_covered={decision.get('topics_covered')} "
          f"needs_second_hop={decision.get('needs_second_hop')} "
          f"missing_domain={missing_domain}")

    return {
        **state,
        "needs_second_hop": decision.get("needs_second_hop", False),
        "reformulated_query": decision.get("reformulated_query"),
        "missing_domain": missing_domain,
    }


def retrieve_second_hop(state: AgentState) -> AgentState:
    query = state["reformulated_query"] or state["question"]
    new_chunks = retrieve(query, top_k=state["top_k"], domain=state.get("missing_domain"))

    existing_ids = {c["id"] for c in state["retrieved_chunks"]}
    merged = state["retrieved_chunks"] + [
        c for c in new_chunks if c["id"] not in existing_ids
    ]
    return {**state, "retrieved_chunks": merged, "hop_count": 2}


GENERATE_PROMPT = """Answer the FMCSA trucking-regulation question using ONLY \
the retrieved regulation text below. Cite the relevant Part/section for each \
claim. If the retrieved text does not fully answer the question, say so \
explicitly rather than guessing.

Question: {question}

Retrieved regulation text:
{context}

Answer:
"""


def generate(state: AgentState) -> AgentState:
    context = "\n\n".join(
        f"[{c['part']} {c['section']}] {c['title']}: {c['text']}"
        for c in state["retrieved_chunks"]
    )
    prompt = GENERATE_PROMPT.format(question=state["question"], context=context)
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


def route_after_decision(state: AgentState) -> str:
    if state.get("force_single_hop"):
        return "generate"
    return "retrieve_second_hop" if state["needs_second_hop"] else "generate"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("decide_next_hop", decide_next_hop)
    graph.add_node("retrieve_second_hop", retrieve_second_hop)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve_initial")
    graph.add_edge("retrieve_initial", "decide_next_hop")
    graph.add_conditional_edges(
        "decide_next_hop",
        route_after_decision,
        {"retrieve_second_hop": "retrieve_second_hop", "generate": "generate"},
    )
    graph.add_edge("retrieve_second_hop", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def ask(question: str, force_single_hop: bool = False, top_k: int = None) -> AgentState:
    """
    force_single_hop=True disables the second retrieval entirely, even if the
    decision node would have requested one. This lets single-hop and
    multi-hop be compared head-to-head on the same questions, using the same
    retrieval/generation code, with only the routing behavior different.

    top_k controls how many chunks each retrieval call returns (both hops
    use the same value). Defaults to DEFAULT_TOP_K if not passed.
    """
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "retrieved_chunks": [],
        "needs_second_hop": False,
        "reformulated_query": None,
        "missing_domain": None,
        "hop_count": 0,
        "answer": None,
        "force_single_hop": force_single_hop,
        "top_k": top_k if top_k is not None else DEFAULT_TOP_K,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    result = ask(
        "If a Class 8 truck is hauling hazardous materials, what inspection "
        "interval and driver qualification rules apply?"
    )
    print(f"Hops used: {result['hop_count']}")
    print(f"Chunks retrieved: {[c['id'] for c in result['retrieved_chunks']]}")
    print(f"\nAnswer:\n{result['answer']}")