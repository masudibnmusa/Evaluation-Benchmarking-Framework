from app.datasets.schemas import CaseResult, ModelOutput, RunRecord, TestCase
from app.tracking.regression_detector import detect


def make_run(run_id, scores, dataset_hash="abc"):
    results = []
    for i, s in enumerate(scores):
        case = TestCase(id=f"c{i}", input="q")
        results.append(CaseResult(case, ModelOutput(f"c{i}", "a", "m"), [], s, s >= 0.5))
    return RunRecord(run_id, "now", dataset_hash, "m", "v1", results=results)


def test_identical_runs_pass():
    base = make_run("base", [1, 1, 1, 1])
    assert detect(make_run("cur", [1, 1, 1, 1]), base).passed


def test_score_drop_fails():
    base = make_run("base", [1, 1, 1, 1])
    report = detect(make_run("cur", [1, 1, 1, 0]), base)
    assert not report.passed
    assert len(report.regressed) == 1


def test_dataset_mismatch_fails_by_default():
    base = make_run("base", [1, 1], dataset_hash="aaa")
    cur = make_run("cur", [1, 1], dataset_hash="bbb")
    assert not detect(cur, base).passed
    assert detect(cur, base, require_same_dataset=False).passed