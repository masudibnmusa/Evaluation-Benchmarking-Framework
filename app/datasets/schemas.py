from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TestCase:
    __test__ = False  # stop pytest from trying to collect this class

    id: str
    input: str
    expected: Optional[str] = None
    criteria: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        return cls(
            id=str(d["id"]),
            input=d["input"],
            expected=d.get("expected"),
            criteria=list(d.get("criteria") or []),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class ModelOutput:
    case_id: str
    output: str
    model: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


@dataclass
class ScoreResult:
    scorer: str
    score: float  # always normalized to 0..1
    passed: bool
    reason: str = ""


@dataclass
class CaseResult:
    case: TestCase
    output: ModelOutput
    scores: list[ScoreResult]
    final_score: float
    passed: bool

    @classmethod
    def from_dict(cls, d: dict) -> "CaseResult":
        return cls(
            case=TestCase.from_dict(d["case"]),
            output=ModelOutput(**d["output"]),
            scores=[ScoreResult(**s) for s in d["scores"]],
            final_score=d["final_score"],
            passed=d["passed"],
        )


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    dataset_hash: str
    model: str
    prompt_version: str
    params: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    results: list[CaseResult] = field(default_factory=list)