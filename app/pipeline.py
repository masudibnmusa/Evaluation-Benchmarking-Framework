"""Glue: run -> score -> summarize. Shared by the CLI, comparators and CI scripts."""
from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import DEFAULT_PASS_SCORE, JUDGE_MODEL, ModelConfig, RunnerConfig
from app.datasets.dataset_loader import dataset_hash
from app.datasets.schemas import CaseResult, ModelOutput, RunRecord, TestCase
from app.runners.task_runner import TaskRunner
from app.scorers.composite_scorer import CompositeScorer


def load_prompt(prompt: str) -> tuple[str, str]:
    """`prompt` is a file path or literal text. Returns (template, version_label)."""
    try:
        is_file = Path(prompt).is_file()
    except OSError:  # literal text too long to be a filename
        is_file = False

    if is_file:
        template, version = Path(prompt).read_text(encoding="utf-8"), Path(prompt).stem
    else:
        template, version = prompt, "inline-" + hashlib.sha256(prompt.encode()).hexdigest()[:8]
    if "{input}" not in template:
        template += "\n\n{input}"
    return template, version


def _make_scorer(name: str, judge_model: str):
    if name == "exact_match":
        from app.scorers.exact_match_scorer import ExactMatchScorer
        return ExactMatchScorer()
    if name == "fuzzy_match":
        from app.scorers.fuzzy_match_scorer import FuzzyMatchScorer
        return FuzzyMatchScorer()
    if name in ("schema", "schema_validator"):
        from app.scorers.schema_validator_scorer import SchemaValidatorScorer
        return SchemaValidatorScorer()
    if name in ("semantic", "semantic_similarity"):
        from app.scorers.semantic_similarity_scorer import SemanticSimilarityScorer
        return SemanticSimilarityScorer()
    if name == "llm_judge":
        from app.scorers.llm_judge_scorer import LLMJudgeScorer
        return LLMJudgeScorer(model=judge_model)
    raise ValueError(f"Unknown scorer: {name}")


def build_composite(spec: str, judge_model: str = JUDGE_MODEL, pass_score: float = DEFAULT_PASS_SCORE) -> CompositeScorer:
    """spec examples: 'exact_match'   or   'exact_match:0.5,llm_judge:0.5'"""
    components = []
    for item in spec.split(","):
        name, _, weight = item.strip().partition(":")
        components.append((_make_scorer(name, judge_model), float(weight) if weight else 1.0))
    return CompositeScorer(components, pass_score=pass_score)


def score_outputs(cases: list[TestCase], outputs: list[ModelOutput], scorer: CompositeScorer) -> list[CaseResult]:
    """Separate from generation so saved outputs can be re-scored without calling the model again."""
    by_id = {o.case_id: o for o in outputs}
    results = []
    for case in cases:
        out = by_id[case.id]
        scores, final, passed = scorer.score_case(case, out)
        results.append(CaseResult(case, out, scores, final, passed))
    return results


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list[CaseResult]) -> dict:
    summary = {
        "n_cases": len(results),
        "mean_score": _mean(r.final_score for r in results),
        "pass_rate": _mean(1.0 if r.passed else 0.0 for r in results),
        "n_errors": sum(1 for r in results if r.output.error),
        "mean_latency_s": _mean(r.output.latency_s for r in results),
        "input_tokens": sum(r.output.input_tokens for r in results),
        "output_tokens": sum(r.output.output_tokens for r in results),
    }
    for key in ("category", "difficulty"):
        groups: dict[str, list[CaseResult]] = defaultdict(list)
        for r in results:
            groups[str(r.case.metadata.get(key, "unspecified"))].append(r)
        summary[f"by_{key}"] = {
            g: {
                "n": len(rs),
                "mean_score": _mean(r.final_score for r in rs),
                "pass_rate": _mean(1.0 if r.passed else 0.0 for r in rs),
            }
            for g, rs in groups.items()
        }
    return summary


def evaluate(cases: list[TestCase], model_cfg: ModelConfig, prompt_template: str, prompt_version: str,
             scorer: CompositeScorer, runner_cfg: RunnerConfig | None = None) -> RunRecord:
    runner_cfg = runner_cfg or RunnerConfig()
    outputs = TaskRunner(model_cfg, runner_cfg).run(cases, prompt_template)
    results = score_outputs(cases, outputs, scorer)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    return RunRecord(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_hash=dataset_hash(cases),
        model=model_cfg.name,
        prompt_version=prompt_version,
        params={
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
            "system": model_cfg.system,
            "prompt_template": prompt_template,
            "scorers": scorer.describe(),
            "pass_score": scorer.pass_score,
        },
        summary=summarize(results),
        results=results,
    )