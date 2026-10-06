from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from app.datasets.schemas import TestCase


def load_dataset(path: str | Path) -> list[TestCase]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data["cases"]
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = [_csv_row_to_dict(r) for r in csv.DictReader(f)]
    else:
        raise ValueError(f"Unsupported dataset format: {suffix} (use .jsonl, .json or .csv)")

    cases = [TestCase.from_dict(r) for r in rows]
    _validate(cases)
    return cases


def _csv_row_to_dict(row: dict) -> dict:
    """CSV columns: id, input, expected, criteria (pipe-separated); any other column -> metadata."""
    row = dict(row)
    return {
        "id": row.pop("id"),
        "input": row.pop("input"),
        "expected": row.pop("expected", None) or None,
        "criteria": [c.strip() for c in (row.pop("criteria", "") or "").split("|") if c.strip()],
        "metadata": {k: v for k, v in row.items() if v},
    }


def _validate(cases: list[TestCase]) -> None:
    if not cases:
        raise ValueError("Dataset is empty")
    seen = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"Duplicate case id: {c.id}")
        seen.add(c.id)
        if not c.input:
            raise ValueError(f"Case {c.id} has an empty input")


def dataset_hash(cases: list[TestCase]) -> str:
    """Stable fingerprint so results are only compared against the same test set."""
    payload = json.dumps([asdict(c) for c in sorted(cases, key=lambda c: c.id)], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]