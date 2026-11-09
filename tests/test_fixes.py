import json

import pytest

from app.comparison.prompt_comparator import compare_prompts
from app.config import ModelConfig
from app.datasets.dataset_loader import load_dataset
from app.datasets.schemas import ModelOutput, TestCase
from app.pipeline import build_composite, summarize, score_outputs
from app.scorers.base_scorer import BaseScorer
from app.scorers.composite_scorer import CompositeScorer
from app.scorers.exact_match_scorer import ExactMatchScorer
from app.scorers.schema_validator_scorer import SchemaValidatorScorer
from app.datasets.schemas import CaseResult, RunRecord
from app.tracking.regression_detector import detect


def make_run(run_id, scores):
    results = [CaseResult(TestCase(id=f"c{i}", input="q"), ModelOutput(f"c{i}", "a", "m"), [], s, s >= 0.5)
               for i, s in enumerate(scores)]
    return RunRecord(run_id, "now", "abc", "m", "v1", results=results)


def test_numeric_expected_in_json_dataset(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"id": 1, "input": "2+2?", "expected": 4}) + "\n")
    case = load_dataset(p)[0]
    assert case.expected == "4"
    assert ExactMatchScorer().score(case, "4").passed


def test_csv_missing_column_gives_clear_error(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("question,expected\nhi,there\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_dataset(p)


class Boom(BaseScorer):
    name = "boom"

    def score(self, case, output):
        raise RuntimeError("kaput")


def test_crashing_scorer_does_not_abort_run():
    scores, final, passed = CompositeScorer([(Boom(), 1)]).score_case(
        TestCase(id="1", input="q"), ModelOutput("1", "x", "m"))
    assert final == 0.0 and not passed and "kaput" in scores[0].reason


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        CompositeScorer([(ExactMatchScorer(), 0)])


def test_build_composite_tolerates_spaces_and_trailing_comma():
    comp = build_composite(" exact_match : 2 , fuzzy_match,")
    assert [(d["scorer"], d["weight"]) for d in comp.describe()] == [("exact_match", 2.0), ("fuzzy_match", 1.0)]


def test_schema_validator_handles_fence_with_text_and_string_schema():
    case = TestCase(id="1", input="q", metadata={"schema": '{"type": "object", "required": ["a"]}'})
    assert SchemaValidatorScorer().score(case, 'Sure:\n```json\n{"a": 1}\n```\nDone').passed


def test_schema_validator_invalid_schema_is_a_failure_not_a_crash():
    r = SchemaValidatorScorer(schema={"type": "not-a-type"}).score(TestCase(id="1", input="q"), "{}")
    assert not r.passed


def test_case_drop_boundary_float_noise():
    base = make_run("b", [0.7] * 10)
    cur = make_run("c", [0.4] + [0.7] * 9)  # drop of 0.29999... must count as 0.3
    assert len(detect(cur, base).regressed) == 1


def test_mean_latency_ignores_errored_cases():
    cases = [TestCase(id="a", input="q"), TestCase(id="b", input="q")]
    outs = [ModelOutput("a", "x", "m", latency_s=2.0), ModelOutput("b", "", "m", error="boom")]
    s = summarize(score_outputs(cases, outs, build_composite("fuzzy_match")))
    assert s["mean_latency_s"] == 2.0


class FakeClient:
    def complete(self, prompt):
        return "ok", 1, 1, 0.1


def test_compare_prompts_same_stem_do_not_overwrite(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "v1.txt").write_text("A {input}")
    (tmp_path / "b" / "v1.txt").write_text("B {input}")
    import app.pipeline as pipeline
    real = pipeline.TaskRunner
    monkeypatch.setattr(pipeline, "TaskRunner", lambda m, r: real(m, r, client=FakeClient()))
    runs = compare_prompts([TestCase(id="1", input="q", expected="ok")],
                           [str(tmp_path / "a" / "v1.txt"), str(tmp_path / "b" / "v1.txt")],
                           ModelConfig(name="m"), build_composite("exact_match"))
    assert len(runs) == 2
