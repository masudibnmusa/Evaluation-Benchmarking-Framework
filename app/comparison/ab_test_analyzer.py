from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.datasets.schemas import RunRecord


@dataclass
class ABResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float  # b - a
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


def paired_bootstrap(a, b, n_resamples: int = 5000, alpha: float = 0.05, seed: int = 0) -> ABResult:
    """Paired bootstrap on per-case score differences (B minus A)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("Need two equal-length score lists with at least 2 cases")

    diffs = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot = diffs[idx].mean(axis=1)

    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return ABResult(
        n=int(diffs.size), mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(diffs.mean()),
        ci_low=float(low), ci_high=float(high), p_value=float(p), significant=bool(low > 0 or high < 0),
    )


def compare_runs(run_a: RunRecord, run_b: RunRecord, **kwargs) -> ABResult:
    """Align two runs by case id and test whether B differs from A."""
    a = {r.case.id: r.final_score for r in run_a.results}
    b = {r.case.id: r.final_score for r in run_b.results}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("Runs share no case ids")
    return paired_bootstrap([a[i] for i in common], [b[i] for i in common], **kwargs)