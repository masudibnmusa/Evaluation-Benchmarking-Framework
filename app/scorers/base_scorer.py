from __future__ import annotations

from abc import ABC, abstractmethod

from app.datasets.schemas import ScoreResult, TestCase


class BaseScorer(ABC):
    name = "base"
    needs_expected = False  # composite skips this scorer for cases without `expected`

    @abstractmethod
    def score(self, case: TestCase, output: str) -> ScoreResult:
        """Score one model output (plain text) for one test case."""

    def _result(self, score: float, reason: str = "", threshold: float = 0.5) -> ScoreResult:
        score = max(0.0, min(1.0, float(score)))
        return ScoreResult(self.name, score, score >= threshold, reason)