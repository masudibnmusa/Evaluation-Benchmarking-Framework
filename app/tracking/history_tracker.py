from __future__ import annotations

import pandas as pd

from app.tracking.results_store import ResultsStore


def score_history(store: ResultsStore | None = None, dataset_hash: str | None = None,
                  model: str | None = None) -> pd.DataFrame:
    """Mean score / pass rate per run over time. Filter by dataset_hash to keep it apples-to-apples."""
    store = store or ResultsStore()
    rows = []
    for meta in store.list_runs():
        if dataset_hash and meta["dataset_hash"] != dataset_hash:
            continue
        if model and meta["model"] != model:
            continue
        rows.append({
            "run_id": meta["run_id"],
            "created_at": meta["created_at"],
            "model": meta["model"],
            "prompt_version": meta["prompt_version"],
            "mean_score": meta["summary"]["mean_score"],
            "pass_rate": meta["summary"]["pass_rate"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at").reset_index(drop=True)
    return df