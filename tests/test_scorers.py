from app.datasets.schemas import ModelOutput, TestCase
from app.scorers.base_scorer import BaseScorer
from app.scorers.composite_scorer import CompositeScorer
from app.scorers.exact_match_scorer import ExactMatchScorer
from app.scorers.fuzzy_match_scorer import FuzzyMatchScorer
from app.scorers.schema_validator_scorer import SchemaValidatorScorer


def test_exact_match_ignores_case_and_whitespace():
    case = TestCase(id="1", input="q", expected="Hello  World")
    assert ExactMatchScorer().score(case, "hello world").passed


def test_exact_match_fails_on_difference():
    case = TestCase(id="1", input="q", expected="4")
    assert not ExactMatchScorer().score(case, "5").passed


def test_fuzzy_match_tolerates_typos():
    case = TestCase(id="1", input="q", expected="hello world")
    assert FuzzyMatchScorer(threshold=0.8).score(case, "hello wrld").passed


def test_schema_validator():
    scorer = SchemaValidatorScorer(schema={"type": "object", "required": ["a"]})
    case = TestCase(id="1", input="q")
    assert scorer.score(case, '{"a": 1}').passed
    assert not scorer.score(case, "not json").passed
    assert scorer.score(case, '```json\n{"a": 1}\n```').passed


class Const(BaseScorer):
    name = "const"

    def __init__(self, value):
        self.value = value

    def score(self, case, output):
        return self._result(self.value)


def test_composite_weighted_average():
    comp = CompositeScorer([(Const(1.0), 3), (Const(0.0), 1)], pass_score=0.5)
    _, final, passed = comp.score_case(TestCase(id="1", input="q"), ModelOutput("1", "x", "m"))
    assert final == 0.75 and passed


def test_composite_scores_errors_as_zero():
    comp = CompositeScorer([(Const(1.0), 1)])
    _, final, passed = comp.score_case(TestCase(id="1", input="q"), ModelOutput("1", "", "m", error="boom"))
    assert final == 0.0 and not passed