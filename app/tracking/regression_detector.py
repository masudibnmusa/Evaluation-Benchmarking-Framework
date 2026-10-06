from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from app.config import DEFAULT_CASE_DROP, DEFAULT_MAX_AGGREGATE_DROP, DEFAULT_MAX_REGRESSED_FRACTION
from app.datasets.schemas import RunRecord


@dataclass
class RegressionReport:
    current_id: str
    baseline_id: str
    dataset_match: bool
    current_score: float
    baseline_score: float
    n_compared: int
    regressed: list[dict] = field(default_factory=list)
    improved: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.reasons

    @property
    def delta(self) -> float:
        return self.current_score - self.baseline_score

    def to_text(self) -> str:
        lines = [
            f"Regression check: {'PASS' if self.passed else 'FAIL'}",
            f"  baseline {self.baseline_id}: {self.baseline_score:.3f}",
            f"  current  {self.current_id}: {self.current_score:.3f}  (delta {self.delta:+.3f})",
            f"  cases compared: {self.n_compared}, regressed: {len(self.regressed)}, improved: {len(self.improved)}",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        for row in self.regressed[:10]:
            lines.append(f"  regressed: {row['id']}  {row['baseline']:.2f} -> {row['current']:.2f}")
        return "\n".join(lines)


def detect(current: RunRecord, baseline: RunRecord,
           max_aggregate_drop: float = DEFAULT_MAX_AGGREGATE_DROP,
           case_drop: float = DEFAULT_CASE_DROP,
           max_regressed_fraction: float = DEFAULT_MAX_REGRESSED_FRACTION,
           require_same_dataset: bool = True) -> RegressionReport:
    """Gate on aggregate score and on the *fraction* of regressed cases,
    not on any single case flipping (model output is noisy)."""
    cur = {r.case.id: r for r in current.results}
    base = {r.case.id: r for r in baseline.results}
    common = sorted(set(cur) & set(base))
    if not common:
        raise ValueError("Current run and baseline share no case ids")

    cur_mean = mean(cur[i].final_score for i in common)
    base_mean = mean(base[i].final_score for i in common)

    regressed, improved = [], []
    for i in common:
        d = cur[i].final_score - base[i].final_score
        row = {"id": i, "baseline": base[i].final_score, "current": cur[i].final_score, "delta": d}
        if d <= -case_drop:
            regressed.append(row)
        elif d >= case_drop:
            improved.append(row)

    reasons = []
    dataset_match = current.dataset_hash == baseline.dataset_hash
    if require_same_dataset and not dataset_match:
        reasons.append("dataset hash differs from baseline (scores are not comparable)")
    if base_mean - cur_mean > max_aggregate_drop:
        reasons.append(f"mean score dropped {base_mean - cur_mean:.3f} (limit {max_aggregate_drop})")
    frac = len(regressed) / len(common)
    if frac > max_regressed_fraction:
        reasons.append(f"{frac:.0%} of cases regressed (limit {max_regressed_fraction:.0%})")

    return RegressionReport(
        current_id=current.run_id, baseline_id=baseline.run_id, dataset_match=dataset_match,
        current_score=cur_mean, baseline_score=base_mean, n_compared=len(common),
        regressed=regressed, improved=improved, reasons=reasons,
    )