from rapidfuzz import fuzz

from app.datasets.schemas import ScoreResult, TestCase
from app.scorers.base_scorer import BaseScorer


class FuzzyMatchScorer(BaseScorer):
    name = "fuzzy_match"
    needs_expected = True

    def __init__(self, threshold: float = 0.8, mode: str = "ratio"):
        """mode: 'ratio' (Levenshtein-based), 'partial_ratio', or 'token_set_ratio'."""
        self.threshold = threshold
        self.fn = getattr(fuzz, mode)

    def score(self, case: TestCase, output: str) -> ScoreResult:
        sim = self.fn(output.strip().lower(), (case.expected or "").strip().lower()) / 100.0
        return self._result(sim, f"similarity={sim:.2f}", self.threshold)