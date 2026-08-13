"""
Fleet Compliance Multi-Hop Assistant — demo UI.

Two tabs:
  1. Chat — ask a question, see the answer plus which hop(s) were taken and
     which chunks were retrieved at each step.
  2. Eval Dashboard — loads eval/eval_results_top*.csv and
     eval/hop_comparison_results.csv (whichever exist) and shows, per
     question, whether it was answered correctly in single-hop vs
     multi-hop mode.

Run with:
    streamlit run app.py
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from agent import ask  # noqa: E402

EVAL_DIR = Path(__file__).parent / "eval"

st.set_page_config(page_title="Fleet Compliance Multi-Hop Assistant", layout="wide")
st.title("Fleet Compliance Multi-Hop Assistant")
st.caption("Agentic multi-hop RAG over FMCSA trucking regulations")

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Eval Dashboard"])

# ---------------------------------------------------------------------------
# TAB 1: Chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask a question")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        question = st.text_input(
            "Question",
            value="If a Class 8 truck is hauling hazardous materials, "
            "what inspection interval and driver qualification rules apply?",
            label_visibility="collapsed",
        )
    with col_b:
        top_k = st.number_input("top_k", min_value=1, max_value=10, value=3)

    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving and reasoning..."):
            result = ask(question, top_k=top_k)

        hop_label = "🔁 Multi-hop (2 retrievals)" if result["hop_count"] == 2 else "➡️ Single-hop (1 retrieval)"
        st.markdown(f"**{hop_label}**")

        st.markdown("### Answer")
        st.write(result["answer"])

        with st.expander(f"Retrieved chunks ({len(result['retrieved_chunks'])})", expanded=False):
            for c in result["retrieved_chunks"]:
                st.markdown(f"**[{c['part']} {c['section']}]** {c['title']}")
                st.caption(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
                st.divider()

        if result.get("reformulated_query"):
            st.markdown("### Second-hop details")
            st.write(f"**Reformulated query:** {result['reformulated_query']}")
            st.write(f"**Missing domain targeted:** {result.get('missing_domain') or 'none'}")

# ---------------------------------------------------------------------------
# TAB 2: Eval Dashboard
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Single-hop vs multi-hop comparison")

    hop_comparison_path = EVAL_DIR / "hop_comparison_results.csv"
    if hop_comparison_path.exists():
        df = pd.read_csv(hop_comparison_path)

        col1, col2, col3 = st.columns(3)
        col1.metric("Single-hop hit rate",
                     f"{df['single_hop_full_hit'].mean():.0%}")
        col2.metric("Multi-hop hit rate",
                     f"{df['multi_hop_full_hit'].mean():.0%}")
        improved = df["multi_hop_improved_over_single"].sum()
        col3.metric("Questions multi-hop fixed", int(improved))

        st.markdown("#### Per-question breakdown")
        display_df = df[[
            "id", "hop_type", "question",
            "single_hop_full_hit", "multi_hop_full_hit",
            "took_second_hop", "multi_hop_improved_over_single",
        ]].rename(columns={
            "hop_type": "labeled_as",
            "single_hop_full_hit": "single-hop: correct?",
            "multi_hop_full_hit": "multi-hop: correct?",
            "took_second_hop": "took 2nd hop?",
            "multi_hop_improved_over_single": "multi-hop fixed it?",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No comparison data yet. Run `python eval/compare_hops.py` first, "
            "then reload this page."
        )

    st.divider()
    st.subheader("Retrieval eval across top_k values")

    topk_files = sorted(EVAL_DIR.glob("eval_results_top*.csv"))
    if topk_files:
        summaries = []
        for f in topk_files:
            k = f.stem.replace("eval_results_top", "")
            df_k = pd.read_csv(f)
            multi_df = df_k[df_k["hop_type"] == "multi"]
            summaries.append({
                "top_k": k,
                "overall_hit_rate": df_k["full_hit"].mean(),
                "multi_hop_hit_rate": multi_df["full_hit"].mean() if len(multi_df) else None,
                "hop_decision_accuracy": df_k["hop_decision_correct"].mean(),
                "multi_hop_decision_accuracy": multi_df["hop_decision_correct"].mean() if len(multi_df) else None,
            })
        summary_df = pd.DataFrame(summaries)
        st.dataframe(
            summary_df.style.format({
                "overall_hit_rate": "{:.0%}",
                "multi_hop_hit_rate": "{:.0%}",
                "hop_decision_accuracy": "{:.0%}",
                "multi_hop_decision_accuracy": "{:.0%}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Hit rate by top_k")
        chart_df = summary_df.set_index("top_k")[["overall_hit_rate", "multi_hop_hit_rate"]]
        st.line_chart(chart_df)
    else:
        st.info(
            "No top_k sweep data yet. Run `python eval/run_eval.py` first, "
            "then reload this page."
        )

    st.divider()
    st.subheader("Per-question hop behavior (most recent single eval run)")
    single_run_files = sorted(EVAL_DIR.glob("eval_results_top*.csv"))
    if single_run_files:
        latest = single_run_files[-1]
        df_latest = pd.read_csv(latest)
        st.caption(f"Source: {latest.name}")
        st.dataframe(
            df_latest[["id", "hop_type", "full_hit", "agent_hop_count", "hop_decision_correct"]]
            .rename(columns={
                "hop_type": "labeled_as",
                "agent_hop_count": "hops_taken",
            }),
            use_container_width=True, hide_index=True,
        )