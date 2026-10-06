from __future__ import annotations

import json

from app.config import JUDGE_MODEL, ModelConfig
from app.datasets.schemas import ScoreResult, TestCase
from app.runners.task_runner import AnthropicClient, LLMClient
from app.scorers.base_scorer import BaseScorer

JUDGE_PROMPT = """You are a strict, impartial evaluator. Judge the response only against the rubric. \
Do not reward length or style for its own sake.

<task_input>
{input}
</task_input>

<response>
{output}
</response>
{reference_block}
Rubric:
{rubric}

Rate the response from 1 (very poor) to 5 (excellent) against the rubric.
Reply with ONLY a JSON object: {{"reasoning": "<one or two sentences>", "score": <integer 1-5>}}"""

DEFAULT_RUBRIC = [
    "Accurately and completely addresses the task input",
    "Is clear, well organized and free of errors",
]


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge reply: {text[:100]!r}")
    return json.loads(text[start : end + 1])


class LLMJudgeScorer(BaseScorer):
    name = "llm_judge"

    def __init__(self, model: str = JUDGE_MODEL, pass_threshold: float = 0.75, client: LLMClient | None = None):
        self.pass_threshold = pass_threshold  # 0.75 == 4/5 on the judge's scale
        self.client = client or AnthropicClient(ModelConfig(name=model, temperature=0.0, max_tokens=300))

    def score(self, case: TestCase, output: str) -> ScoreResult:
        rubric = "\n".join(f"- {c}" for c in (case.criteria or DEFAULT_RUBRIC))
        reference = (
            f"\n<reference_answer>\n{case.expected}\n</reference_answer>\n" if case.expected else ""
        )
        prompt = JUDGE_PROMPT.format(
            input=case.input, output=output, reference_block=reference, rubric=rubric
        )
        try:
            text, *_ = self.client.complete(prompt)
            data = _extract_json(text)
            raw = min(max(int(data["score"]), 1), 5)
        except Exception as exc:  # noqa: BLE001
            return self._result(0.0, f"judge error: {exc}")
        return self._result((raw - 1) / 4, f"{raw}/5 - {data.get('reasoning', '')}", self.pass_threshold)