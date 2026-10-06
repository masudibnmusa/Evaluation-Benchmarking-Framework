"""Central configuration: paths, API keys, model/runner settings, default thresholds."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TEST_SETS_DIR = DATA_DIR / "test_sets"
RESULTS_DIR = DATA_DIR / "results"
BASELINES_DIR = DATA_DIR / "baselines"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_MODEL = os.getenv("EVAL_DEFAULT_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-4-6")

# Default thresholds
DEFAULT_PASS_SCORE = 0.5            # a case passes if its final score >= this
DEFAULT_MAX_AGGREGATE_DROP = 0.02   # max allowed drop in mean score vs baseline
DEFAULT_CASE_DROP = 0.3             # a single case "regressed" if its score fell by this much
DEFAULT_MAX_REGRESSED_FRACTION = 0.1


@dataclass
class ModelConfig:
    name: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 1024
    system: str | None = None


@dataclass
class RunnerConfig:
    concurrency: int = 8
    max_retries: int = 3
    timeout_s: float = 60.0
    requests_per_second: float = 0.0  # 0 = unlimited