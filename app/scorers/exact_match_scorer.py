from app.datasets.schemas import ScoreResult, TestCase
from app.scorers.base_scorer import BaseScorer


class ExactMatchScorer(BaseScorer):
    name = "exact_match"
    needs_expected = True

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def _norm(self, s: str) -> str:
        s = " ".join(s.split())  # collapse whitespace
        return s if self.case_sensitive else s.lower()

    def score(self, case: TestCase, output: str) -> ScoreResult:
        ok = self._norm(output) == self._norm(case.expected or "")
        reason = "match" if ok else f"expected {case.expected!r}, got {output[:80]!r}"
        return self._result(1.0 if ok else 0.0, reason)