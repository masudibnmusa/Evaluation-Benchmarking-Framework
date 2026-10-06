from __future__ import annotations

import time
from typing import Protocol

from app.config import ANTHROPIC_API_KEY, ModelConfig, RunnerConfig
from app.datasets.schemas import ModelOutput, TestCase
from app.runners.batch_executor import run_batch
from app.runners.retry_handler import call_with_retries
from app.utils.logger import get_logger

log = get_logger(__name__)


class LLMClient(Protocol):
    def complete(self, prompt: str) -> tuple[str, int, int, float]:
        """Return (text, input_tokens, output_tokens, latency_seconds)."""


class AnthropicClient:
    def __init__(self, model_cfg: ModelConfig, runner_cfg: RunnerConfig | None = None):
        import anthropic

        runner_cfg = runner_cfg or RunnerConfig()
        self.cfg = model_cfg
        self.max_retries = runner_cfg.max_retries
        # SDK retries disabled: retry_handler owns retry behavior
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=runner_cfg.timeout_s, max_retries=0)

    def complete(self, prompt: str) -> tuple[str, int, int, float]:
        kwargs = dict(
            model=self.cfg.name,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.cfg.system:
            kwargs["system"] = self.cfg.system

        start = time.perf_counter()
        resp = call_with_retries(self.client.messages.create, max_retries=self.max_retries, **kwargs)
        latency = time.perf_counter() - start
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens, latency


class TaskRunner:
    """Runs every test case through the model under test."""

    def __init__(self, model_cfg: ModelConfig, runner_cfg: RunnerConfig | None = None,
                 client: LLMClient | None = None):
        self.model_cfg = model_cfg
        self.runner_cfg = runner_cfg or RunnerConfig()
        self.client = client or AnthropicClient(model_cfg, self.runner_cfg)

    def _run_one(self, case: TestCase, template: str) -> ModelOutput:
        # .replace instead of .format so braces in prompts (e.g. JSON examples) are safe
        prompt = template.replace("{input}", case.input)
        try:
            text, tok_in, tok_out, latency = self.client.complete(prompt)
            return ModelOutput(case.id, text, self.model_cfg.name, latency, tok_in, tok_out)
        except Exception as exc:  # noqa: BLE001
            log.warning("Case %s failed: %s", case.id, exc)
            return ModelOutput(case.id, "", self.model_cfg.name, error=f"{type(exc).__name__}: {exc}")

    def run(self, cases: list[TestCase], template: str) -> list[ModelOutput]:
        log.info("Running %d cases on %s", len(cases), self.model_cfg.name)
        return run_batch(
            cases,
            lambda c: self._run_one(c, template),
            concurrency=self.runner_cfg.concurrency,
            requests_per_second=self.runner_cfg.requests_per_second,
        )