from __future__ import annotations

from app.config import ModelConfig, RunnerConfig
from app.datasets.schemas import RunRecord, TestCase
from app.pipeline import evaluate, load_prompt
from app.scorers.composite_scorer import CompositeScorer


def compare_prompts(cases: list[TestCase], prompt_specs: list[str], model_cfg: ModelConfig,
                    scorer: CompositeScorer, runner_cfg: RunnerConfig | None = None) -> dict[str, RunRecord]:
    """Run the same test set and model across several prompt variants (file paths or literal text)."""
    runs = {}
    for spec in prompt_specs:
        template, version = load_prompt(spec)
        runs[version] = evaluate(cases, model_cfg, template, version, scorer, runner_cfg)
    return runs