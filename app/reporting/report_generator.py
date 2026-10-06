from __future__ import annotations

from pathlib import Path

from app.datasets.schemas import RunRecord
from app.reporting.failure_analyzer import worst_cases
from app.tracking.regression_detector import RegressionReport


def _clip(text: str, n: int = 200) -> str:
    text = (text or "").replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


def generate_report(run: RunRecord, regression: RegressionReport | None = None, top_failures: int = 5) -> str:
    s = run.summary
    scorers = ", ".join(f"{c['scorer']} ({c['weight']})" for c in run.params.get("scorers", []))
    lines = [
        f"# Eval report: {run.run_id}",
        "",
        f"- Model: `{run.model}`",
        f"- Prompt version: `{run.prompt_version}`",
        f"- Dataset hash: `{run.dataset_hash}`",
        f"- Scorers: {scorers}",
        f"- Cases: {s['n_cases']} | errors: {s['n_errors']}",
        "",
        "## Overall",
        "",
        f"- Mean score: **{s['mean_score']:.3f}**",
        f"- Pass rate: **{s['pass_rate']:.1%}**",
        f"- Mean latency: {s['mean_latency_s']:.2f}s",
        f"- Tokens: {s['input_tokens']} in / {s['output_tokens']} out",
    ]

    for key in ("by_category", "by_difficulty"):
        groups = s.get(key) or {}
        if len(groups) > 1:
            lines += ["", f"## {key.replace('_', ' ').title()}", "",
                      "| Group | N | Mean score | Pass rate |", "|---|---|---|---|"]
            for g, v in sorted(groups.items()):
                lines.append(f"| {g} | {v['n']} | {v['mean_score']:.3f} | {v['pass_rate']:.1%} |")

    failures = worst_cases(run, top_failures)
    if failures:
        lines += ["", f"## Worst failures (top {len(failures)})"]
        for r in failures:
            lines += [
                "", f"### {r.case.id} (score {r.final_score:.2f})",
                f"- Input: {_clip(r.case.input)}",
                f"- Expected: {_clip(r.case.expected or '-')}",
                f"- Output: {_clip(r.output.output or r.output.error or '')}",
            ]
            lines += [f"- {sc.scorer}: {sc.reason}" for sc in r.scores if not sc.passed]

    if regression:
        lines += ["", "## Regression check", "", "```", regression.to_text(), "```"]
    return "\n".join(lines)


def save_report(run: RunRecord, path: str | Path, regression: RegressionReport | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_report(run, regression), encoding="utf-8")
    return path