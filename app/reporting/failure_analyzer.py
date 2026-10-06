from __future__ import annotations

from collections import Counter

from app.datasets.schemas import CaseResult, RunRecord


def get_failures(run: RunRecord) -> list[CaseResult]:
    return [r for r in run.results if not r.passed]


def worst_cases(run: RunRecord, k: int = 5) -> list[CaseResult]:
    return sorted(get_failures(run), key=lambda r: r.final_score)[:k]


def group_failures(run: RunRecord) -> dict:
    """Where are failures concentrated: by category, by scorer, by error type?"""
    failures = get_failures(run)
    return {
        "n_failures": len(failures),
        "by_category": dict(Counter(r.case.metadata.get("category", "unspecified") for r in failures)),
        "by_scorer": dict(Counter(s.scorer for r in failures for s in r.scores if not s.passed)),
        "by_error": dict(Counter(r.output.error.split(":")[0] for r in failures if r.output.error)),
    }