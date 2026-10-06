from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from app.config import BASELINES_DIR, RESULTS_DIR
from app.datasets.schemas import CaseResult, RunRecord


class ResultsStore:
    """One JSONL file per run: line 1 = run metadata + summary, then one line per case
    (including the raw model output, so you can re-score later without re-calling the model)."""

    def __init__(self, results_dir: Path = RESULTS_DIR, baselines_dir: Path = BASELINES_DIR):
        self.results_dir = Path(results_dir)
        self.baselines_dir = Path(baselines_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write(path: Path, run: RunRecord) -> None:
        header = {k: v for k, v in asdict(run).items() if k != "results"}
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "run", **header}, ensure_ascii=False) + "\n")
            for r in run.results:
                f.write(json.dumps({"type": "case", **asdict(r)}, ensure_ascii=False) + "\n")

    def save(self, run: RunRecord) -> Path:
        path = self.results_dir / f"{run.run_id}.jsonl"
        self._write(path, run)
        return path

    @staticmethod
    def load_path(path: str | Path) -> RunRecord:
        with Path(path).open(encoding="utf-8") as f:
            header = json.loads(f.readline())
            results = [CaseResult.from_dict(json.loads(line)) for line in f if line.strip()]
        header.pop("type", None)
        return RunRecord(results=results, **header)

    def load(self, run_id: str = "latest") -> RunRecord:
        if run_id == "latest":
            files = sorted(self.results_dir.glob("*.jsonl"))
            if not files:
                raise FileNotFoundError(f"No saved runs in {self.results_dir}")
            return self.load_path(files[-1])
        return self.load_path(self.results_dir / f"{run_id}.jsonl")

    def list_runs(self) -> list[dict]:
        """Metadata + summary for every saved run, oldest first (cheap: reads only line 1)."""
        runs = []
        for path in sorted(self.results_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                header = json.loads(f.readline())
            header["path"] = str(path)
            runs.append(header)
        return runs

    def set_baseline(self, run_id: str, name: str) -> Path:
        run = self.load(run_id)
        path = self.baselines_dir / f"{name}.jsonl"
        self._write(path, run)
        return path

    def load_baseline(self, name_or_path: str) -> RunRecord:
        p = Path(name_or_path)
        if p.is_file():
            return self.load_path(p)
        return self.load_path(self.baselines_dir / f"{name_or_path}.jsonl")