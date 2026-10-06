from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from app.datasets.dataset_loader import load_dataset
from app.datasets.schemas import TestCase


class DatasetBuilder:
    """Small helper for curating test cases and saving them as JSONL.

    Example:
        b = DatasetBuilder()
        b.add("2+2?", expected="4", category="math", difficulty="easy")
        b.save("data/test_sets/math.jsonl")
    """

    def __init__(self, cases: list[TestCase] | None = None):
        self.cases: list[TestCase] = list(cases or [])

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetBuilder":
        return cls(load_dataset(path))

    def add(self, input: str, expected: str | None = None, criteria: list[str] | None = None,
            id: str | None = None, **metadata) -> TestCase:
        case_id = id or f"case_{len(self.cases) + 1:03d}"
        if any(c.id == case_id for c in self.cases):
            raise ValueError(f"Duplicate case id: {case_id}")
        case = TestCase(id=case_id, input=input, expected=expected,
                        criteria=criteria or [], metadata=metadata)
        self.cases.append(case)
        return case

    def summary(self) -> dict:
        return {
            "n_cases": len(self.cases),
            "by_category": dict(Counter(c.metadata.get("category", "unspecified") for c in self.cases)),
            "by_difficulty": dict(Counter(c.metadata.get("difficulty", "unspecified") for c in self.cases)),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for c in self.cases:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        return path