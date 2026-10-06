from __future__ import annotations

from dataclasses import replace

import pandas as pd

from app.config import ModelConfig, RunnerConfig
from app.datasets.schemas import RunRecord, TestCase
from app.pipeline import evaluate
from app.scorers.composite_scorer import CompositeScorer


def compare_models(cases: list[TestCase], models: list[str], prompt_template: str, prompt_version: str,
                   scorer: CompositeScorer, base_cfg: ModelConfig | None = None,
                   runner_cfg: RunnerConfig | None = None) -> dict[str, RunRecord]:
    """Run the same test set and prompt across several models."""
    base_cfg = base_cfg or ModelConfig(name=models[0])
    return {
        name: evaluate(cases, replace(base_cfg, name=name), prompt_template, prompt_version, scorer, runner_cfg)
        for name in models
    }


def comparison_table(runs: dict[str, RunRecord]) -> pd.DataFrame:
    rows = []
    for label, run in runs.items():
        s = run.summary
        rows.append({
            "variant": label,
            "mean_score": round(s["mean_score"], 3),
            "pass_rate": round(s["pass_rate"], 3),
            "errors": s["n_errors"],
            "mean_latency_s": round(s["mean_latency_s"], 2),
            "input_tokens": s["input_tokens"],
            "output_tokens": s["output_tokens"],
        })
    return pd.DataFrame(rows).set_index("variant")