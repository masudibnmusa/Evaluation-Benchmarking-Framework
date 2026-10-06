from __future__ import annotations

import json
import re

import jsonschema

from app.datasets.schemas import ScoreResult, TestCase
from app.scorers.base_scorer import BaseScorer


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):  # tolerate ```json fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text)


class SchemaValidatorScorer(BaseScorer):
    """Rule-based checks: JSON schema (constructor or case.metadata['schema']), regex, max length."""

    name = "schema_validator"

    def __init__(self, schema: dict | None = None, pattern: str | None = None, max_chars: int | None = None):
        self.schema = schema
        self.pattern = pattern
        self.max_chars = max_chars

    def score(self, case: TestCase, output: str) -> ScoreResult:
        schema = self.schema or case.metadata.get("schema")
        checks, failures = 0, []

        if schema is not None:
            checks += 1
            try:
                jsonschema.validate(_parse_json(output), schema)
            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                failures.append(f"schema: {str(e).splitlines()[0]}")

        if self.pattern:
            checks += 1
            if not re.search(self.pattern, output):
                failures.append(f"pattern {self.pattern!r} not found")

        if self.max_chars:
            checks += 1
            if len(output) > self.max_chars:
                failures.append(f"length {len(output)} > {self.max_chars}")

        if checks == 0:
            return self._result(1.0, "no checks configured")
        return self._result((checks - len(failures)) / checks, "; ".join(failures) or "all checks passed", threshold=1.0)