"""
Retrieval evaluation for the Fleet Compliance Multi-Hop Assistant.

For each labeled question, runs it through the LangGraph agent and checks:
  - hit_rate: did retrieved chunks cover the expected chunk id(s)?
  - hop_accuracy (multi-hop questions only): did the agent's hop count match
    the expected label? (Note: at wider top_k this metric can under-count
    genuinely good behavior — see README's note on top_k experiments.)

Supports sweeping multiple top_k values in one run, one at a time, each
writing its own results file: eval_results_top{K}.csv

Usage:
    python eval/run_eval.py              # uses default top_k list: 3, 5, 7
    python eval/run_eval.py 3             # just top_k=3
    python eval/run_eval.py 3 5 7 10      # custom list of top_k values
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent import ask  # noqa: E402

EVAL_PATH = Path(__file__).parent / "eval_set.json"
DEFAULT_TOP_K_VALUES = [3, 5, 7]


def run_for_top_k(eval_set: list[dict], top_k: int) -> dict:
    """Runs the full eval set once at a given top_k. Returns summary stats."""
    results_path = Path(__file__).parent / f"eval_results_top{top_k}.csv"
    rows = []
    hits, total = 0, 0
    hop_correct, hop_total = 0, 0

    print(f"\n=== Running eval at top_k={top_k} ===")

    for item in eval_set:
        result = ask(item["question"], top_k=top_k)
        retrieved_ids = {c["id"] for c in result["retrieved_chunks"]}
        expected_ids = set(item["expected_chunk_ids"])

        covered = expected_ids.issubset(retrieved_ids)
        overlap = len(expected_ids & retrieved_ids) / max(len(expected_ids), 1)

        expected_multi = item["hop_type"] == "multi"
        actual_multi = result["hop_count"] == 2
        hop_match = expected_multi == actual_multi

        hits += 1 if covered else 0
        total += 1
        hop_correct += 1 if hop_match else 0
        hop_total += 1

        rows.append(
            {
                "id": item["id"],
                "hop_type": item["hop_type"],
                "question": item["question"],
                "expected_ids": ";".join(expected_ids),
                "retrieved_ids": ";".join(retrieved_ids),
                "full_hit": covered,
                "partial_overlap": round(overlap, 2),
                "agent_hop_count": result["hop_count"],
                "hop_decision_correct": hop_match,
            }
        )
        print(
            f"[{item['id']}] hop={item['hop_type']:6s} "
            f"full_hit={covered!s:5s} overlap={overlap:.2f} "
            f"agent_hops={result['hop_count']} hop_ok={hop_match}"
        )

    single_hits = sum(r["full_hit"] for r in rows if r["hop_type"] == "single")
    single_total = sum(1 for r in rows if r["hop_type"] == "single")
    multi_hits = sum(r["full_hit"] for r in rows if r["hop_type"] == "multi")
    multi_total = sum(1 for r in rows if r["hop_type"] == "multi")
    multi_only = [r for r in rows if r["hop_type"] == "multi"]
    multi_hop_correct = sum(r["hop_decision_correct"] for r in multi_only)
    multi_took_one_hop = sum(1 for r in multi_only if r["agent_hop_count"] == 1)

    summary = {
        "top_k": top_k,
        "overall_hit_rate": hits / total,
        "single_hit_rate": single_hits / max(single_total, 1),
        "multi_hit_rate": multi_hits / max(multi_total, 1),
        "hop_decision_accuracy": hop_correct / hop_total,
        "multi_hop_decision_accuracy": multi_hop_correct / max(len(multi_only), 1),
        "multi_questions_that_took_only_1_hop": multi_took_one_hop,
        "multi_questions_total": len(multi_only),
    }

    print(f"\n--- Summary (top_k={top_k}) ---")
    print(f"Overall hit rate: {hits}/{total} ({summary['overall_hit_rate']:.0%})")
    print(f"Single-hop hit rate: {single_hits}/{single_total} ({summary['single_hit_rate']:.0%})")
    print(f"Multi-hop hit rate: {multi_hits}/{multi_total} ({summary['multi_hit_rate']:.0%})")
    print(f"Hop-decision accuracy (overall): {hop_correct}/{hop_total} ({summary['hop_decision_accuracy']:.0%})")
    print(f"Hop-decision accuracy (multi-hop only): {multi_hop_correct}/{len(multi_only)} "
          f"({summary['multi_hop_decision_accuracy']:.0%})")
    print(f"Multi-hop questions answered in just 1 hop: {multi_took_one_hop}/{len(multi_only)}")

    with open(results_path, "w") as f:
        f.write(
            "id,hop_type,full_hit,partial_overlap,agent_hop_count,hop_decision_correct\n"
        )
        for r in rows:
            f.write(
                f"{r['id']},{r['hop_type']},{r['full_hit']},"
                f"{r['partial_overlap']},{r['agent_hop_count']},"
                f"{r['hop_decision_correct']}\n"
            )
    print(f"Results written to {results_path}")

    return summary


def run():
    eval_set = json.loads(EVAL_PATH.read_text())

    top_k_values = (
        [int(v) for v in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_TOP_K_VALUES
    )

    all_summaries = []
    for top_k in top_k_values:
        summary = run_for_top_k(eval_set, top_k)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print("\n\n=== Cross-top_k comparison ===")
        header = (
            f"{'top_k':>6} | {'overall_hit':>11} | {'multi_hit':>9} | "
            f"{'hop_acc':>7} | {'multi_hop_acc':>13} | {'multi_in_1_hop':>14}"
        )
        print(header)
        print("-" * len(header))
        for s in all_summaries:
            print(
                f"{s['top_k']:>6} | {s['overall_hit_rate']:>10.0%}  | "
                f"{s['multi_hit_rate']:>8.0%}  | {s['hop_decision_accuracy']:>6.0%}  | "
                f"{s['multi_hop_decision_accuracy']:>12.0%}  | "
                f"{s['multi_questions_that_took_only_1_hop']:>3}/{s['multi_questions_total']}"
            )


if __name__ == "__main__":
    run()