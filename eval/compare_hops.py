"""
Runs the full eval set twice through the same LangGraph agent:
  1. force_single_hop=True  — always answers from the first retrieval only
  2. force_single_hop=False — the normal agent, free to take a second hop

Same retrieval code, same generation prompt, same corpus — the only
difference is whether a second, reformulated retrieval is allowed. This
isolates the effect of the multi-hop mechanism itself.

Usage:
    python eval/compare_hops.py
"""
import json
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent import ask  # noqa: E402

EVAL_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH = Path(__file__).parent / "hop_comparison_results.csv"


def hit_rate_for(retrieved_ids: set, expected_ids: set) -> tuple[bool, float]:
    full_hit = expected_ids.issubset(retrieved_ids)
    overlap = len(expected_ids & retrieved_ids) / max(len(expected_ids), 1)
    return full_hit, overlap


def run():
    eval_set = json.loads(EVAL_PATH.read_text())
    rows = []

    single_hits, multi_hits = 0, 0
    single_overlap_sum, multi_overlap_sum = 0.0, 0.0
    wanted_second_hop_count = 0

    for item in eval_set:
        expected_ids = set(item["expected_chunk_ids"])

        single_result = ask(item["question"], force_single_hop=True)
        single_retrieved = {c["id"] for c in single_result["retrieved_chunks"]}
        single_full_hit, single_overlap = hit_rate_for(single_retrieved, expected_ids)

        multi_result = ask(item["question"], force_single_hop=False)
        multi_retrieved = {c["id"] for c in multi_result["retrieved_chunks"]}
        multi_full_hit, multi_overlap = hit_rate_for(multi_retrieved, expected_ids)

        wanted_second_hop = single_result["needs_second_hop"]
        took_second_hop = multi_result["hop_count"] == 2

        single_hits += 1 if single_full_hit else 0
        multi_hits += 1 if multi_full_hit else 0
        single_overlap_sum += single_overlap
        multi_overlap_sum += multi_overlap
        wanted_second_hop_count += 1 if wanted_second_hop else 0

        improved = multi_full_hit and not single_full_hit

        rows.append(
            {
                "id": item["id"],
                "hop_type": item["hop_type"],
                "question": item["question"],
                "single_hop_full_hit": single_full_hit,
                "single_hop_overlap": round(single_overlap, 2),
                "multi_hop_full_hit": multi_full_hit,
                "multi_hop_overlap": round(multi_overlap, 2),
                "took_second_hop": took_second_hop,
                "multi_hop_improved_over_single": improved,
            }
        )

        marker = " <-- multi-hop fixed this" if improved else ""
        print(
            f"[{item['id']:4s}] {item['hop_type']:6s} "
            f"single: hit={single_full_hit!s:5s} overlap={single_overlap:.2f}  |  "
            f"multi: hit={multi_full_hit!s:5s} overlap={multi_overlap:.2f} "
            f"(2nd hop taken={took_second_hop}){marker}"
        )

    n = len(eval_set)
    print("\n--- Summary ---")
    print(f"Single-hop hit rate:  {single_hits}/{n} ({single_hits/n:.0%})  "
          f"avg overlap {single_overlap_sum/n:.2f}")
    print(f"Multi-hop hit rate:   {multi_hits}/{n} ({multi_hits/n:.0%})  "
          f"avg overlap {multi_overlap_sum/n:.2f}")
    print(f"Improvement:          +{multi_hits - single_hits} questions fully answered "
          f"({(multi_hits - single_hits)/n:+.0%} hit rate)")

    multi_only = [r for r in rows if r["hop_type"] == "multi"]
    multi_single_hits = sum(r["single_hop_full_hit"] for r in multi_only)
    multi_multi_hits = sum(r["multi_hop_full_hit"] for r in multi_only)
    print(f"\nOn multi-hop-labeled questions only ({len(multi_only)} questions):")
    print(f"  Single-hop mode hit rate: {multi_single_hits}/{len(multi_only)} "
          f"({multi_single_hits/len(multi_only):.0%})")
    print(f"  Multi-hop mode hit rate:  {multi_multi_hits}/{len(multi_only)} "
          f"({multi_multi_hits/len(multi_only):.0%})")

    fieldnames = [
        "id", "hop_type", "question",
        "single_hop_full_hit", "single_hop_overlap",
        "multi_hop_full_hit", "multi_hop_overlap",
        "took_second_hop", "multi_hop_improved_over_single",
    ]
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nDetailed results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run()