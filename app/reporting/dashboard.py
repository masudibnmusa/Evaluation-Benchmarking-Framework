"""Streamlit dashboard.   Run:  streamlit run app/reporting/dashboard.py"""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.comparison.ab_test_analyzer import compare_runs
from app.reporting.failure_analyzer import group_failures, worst_cases
from app.tracking.history_tracker import score_history
from app.tracking.results_store import ResultsStore

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide")
st.title("LLM Eval Dashboard")

store = ResultsStore()
runs = store.list_runs()
if not runs:
    st.info("No runs yet. Run `python -m app.main run ...` first.")
    st.stop()

run_ids = [r["run_id"] for r in reversed(runs)]
run = store.load(st.sidebar.selectbox("Run", run_ids))
s = run.summary

tab_overview, tab_failures, tab_trends, tab_compare = st.tabs(["Overview", "Failures", "Trends", "Compare runs"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean score", f"{s['mean_score']:.3f}")
    c2.metric("Pass rate", f"{s['pass_rate']:.1%}")
    c3.metric("Mean latency", f"{s['mean_latency_s']:.2f}s")
    c4.metric("Errors", s["n_errors"])
    st.caption(f"model `{run.model}` | prompt `{run.prompt_version}` | dataset `{run.dataset_hash}`")

    for key in ("by_category", "by_difficulty"):
        if s.get(key):
            st.subheader(key.replace("_", " ").title())
            df = pd.DataFrame(s[key]).T
            st.dataframe(df)
            st.bar_chart(df["mean_score"])

with tab_failures:
    groups = group_failures(run)
    st.write(f"**{groups['n_failures']}** failing cases of {s['n_cases']}")
    c1, c2 = st.columns(2)
    c1.write("By category")
    c1.json(groups["by_category"])
    c2.write("By scorer")
    c2.json(groups["by_scorer"])
    for r in worst_cases(run, 15):
        with st.expander(f"{r.case.id}  (score {r.final_score:.2f})"):
            st.write("**Input**")
            st.code(r.case.input, language=None)
            st.write("**Expected**")
            st.code(r.case.expected or "-", language=None)
            st.write("**Output**")
            st.code(r.output.output or r.output.error or "", language=None)
            for sc in r.scores:
                st.write(f"- {sc.scorer}: {sc.score:.2f} - {sc.reason}")

with tab_trends:
    hist = score_history(store, dataset_hash=run.dataset_hash)
    st.caption("Only runs on the same dataset version are shown.")
    if len(hist) > 1:
        st.line_chart(hist.set_index("created_at")[["mean_score", "pass_rate"]])
    st.dataframe(hist)

with tab_compare:
    others = [r for r in run_ids if r != run.run_id]
    if not others:
        st.info("Need at least two runs to compare.")
    else:
        other = store.load(st.selectbox("Compare against (A)", others))
        st.caption(f"Testing whether the selected run (B) differs from A. Dataset match: {other.dataset_hash == run.dataset_hash}")
        try:
            st.json(asdict(compare_runs(other, run)))
        except ValueError as e:
            st.error(str(e))