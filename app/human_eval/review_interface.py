"""Streamlit rating UI.   Run:  streamlit run app/human_eval/review_interface.py"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.config import RESULTS_DIR
from app.human_eval.inter_rater_agreement import rater_agreement
from app.tracking.results_store import ResultsStore

RATINGS_DIR = RESULTS_DIR / "human_ratings"


def load_ratings(run_id: str) -> list[dict]:
    path = RATINGS_DIR / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_rating(run_id: str, rater: str, case_id: str, rating: int, comment: str) -> None:
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id, "rater": rater, "case_id": case_id, "rating": rating,
        "comment": comment, "ts": datetime.now(timezone.utc).isoformat(),
    }
    with (RATINGS_DIR / f"{run_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    st.set_page_config(page_title="Human review", layout="wide")
    st.title("Human review")

    store = ResultsStore()
    runs = store.list_runs()
    if not runs:
        st.info("No runs yet. Run an evaluation first.")
        return

    run_id = st.sidebar.selectbox("Run", [r["run_id"] for r in reversed(runs)])
    rater = st.sidebar.text_input("Your name")
    run = store.load(run_id)
    ratings = load_ratings(run_id)

    if len({r["rater"] for r in ratings}) >= 2:
        with st.sidebar.expander("Inter-rater agreement"):
            for (a, b), v in rater_agreement(ratings).items():
                st.write(f"{a} vs {b}: kappa={v['kappa']:.2f}, weighted={v['weighted_kappa']:.2f} (n={v['n']})")

    if not rater:
        st.warning("Enter your name in the sidebar to start rating.")
        return

    done = {r["case_id"] for r in ratings if r["rater"] == rater}
    todo = [r for r in run.results if r.case.id not in done]
    st.progress(len(done) / max(1, len(run.results)), text=f"{len(done)}/{len(run.results)} reviewed")
    if not todo:
        st.success("All cases reviewed.")
        return

    item = todo[0]
    st.subheader(f"Case {item.case.id}")
    st.markdown("**Input**")
    st.code(item.case.input, language=None)
    if item.case.expected:
        st.markdown("**Expected**")
        st.code(item.case.expected, language=None)
    if item.case.criteria:
        st.markdown("**Criteria**\n" + "\n".join(f"- {c}" for c in item.case.criteria))
    st.markdown("**Model output**")
    st.code(item.output.output or item.output.error or "", language=None)

    rating = st.slider("Quality (1 = poor, 5 = excellent)", 1, 5, 3, key=f"rating_{item.case.id}")
    comment = st.text_area("Comment (optional)", key=f"comment_{item.case.id}")
    if st.button("Submit and next"):
        save_rating(run_id, rater, item.case.id, rating, comment)
        st.rerun()


main()