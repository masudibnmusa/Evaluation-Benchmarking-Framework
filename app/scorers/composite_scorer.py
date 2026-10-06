from __future__ import annotations

from app.datasets.schemas import ModelOutput, ScoreResult, TestCase
from app.scorers.base_scorer import BaseScorer


class CompositeScorer:
    """Weighted combination of several scorers -> one final score per case."""

    def __init__(self, components: list[tuple[BaseScorer, float]], pass_score: float = 0.5):
        if not components:
            raise ValueError("CompositeScorer needs at least one scorer")
        self.components = components
        self.pass_score = pass_score

    def describe(self) -> list[dict]:
        return [{"scorer": s.name, "weight": w} for s, w in self.components]

    def score_case(self, case: TestCase, model_output: ModelOutput) -> tuple[list[ScoreResult], float, bool]:
        if model_output.error:
            return [ScoreResult("error", 0.0, False, model_output.error)], 0.0, False

        results: list[ScoreResult] = []
        weighted = total = 0.0
        for scorer, weight in self.components:
            if scorer.needs_expected and case.expected is None:
                continue  # can't apply this scorer to this case
            r = scorer.score(case, model_output.output)
            results.append(r)
            weighted += r.score * weight
            total += weight

        if total == 0:
            return [ScoreResult("none", 0.0, False, "no applicable scorers")], 0.0, False
        final = weighted / total
        return results, final, final >= self.pass_score