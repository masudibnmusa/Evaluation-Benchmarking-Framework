"""CLI entry point.   python -m app.main --help"""
from __future__ import annotations

import argparse
import sys

from app.comparison.ab_test_analyzer import compare_runs
from app.comparison.model_comparator import compare_models, comparison_table
from app.comparison.prompt_comparator import compare_prompts
from app.config import (
    DEFAULT_CASE_DROP, DEFAULT_MAX_AGGREGATE_DROP, DEFAULT_MAX_REGRESSED_FRACTION, DEFAULT_MODEL,
    DEFAULT_PASS_SCORE, JUDGE_MODEL, ModelConfig, RunnerConfig,
)
from app.datasets.dataset_loader import load_dataset
from app.pipeline import build_composite, evaluate, load_prompt
from app.reporting.report_generator import generate_report
from app.tracking.regression_detector import detect
from app.tracking.results_store import ResultsStore
from app.utils.logger import get_logger

log = get_logger("app.main")


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--dataset", required=True, help="path to .jsonl / .json / .csv test set")
    sp.add_argument("--scorers", default="exact_match",
                    help="comma list with optional weights, e.g. exact_match:0.5,llm_judge:0.5")
    sp.add_argument("--judge-model", default=JUDGE_MODEL)
    sp.add_argument("--pass-score", type=float, default=DEFAULT_PASS_SCORE)
    sp.add_argument("--concurrency", type=int, default=8)
    sp.add_argument("--no-save", action="store_true", help="do not write results to data/results/")


def _add_gate(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--max-drop", type=float, default=DEFAULT_MAX_AGGREGATE_DROP)
    sp.add_argument("--case-drop", type=float, default=DEFAULT_CASE_DROP)
    sp.add_argument("--max-regressed-fraction", type=float, default=DEFAULT_MAX_REGRESSED_FRACTION)
    sp.add_argument("--allow-dataset-change", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-eval", description="LLM evaluation framework")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate one model + prompt on a dataset")
    _add_common(run)
    run.add_argument("--prompt", required=True, help="prompt file or literal text; use {input} as the placeholder")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--baseline", help="baseline name or path; if set, also run the regression check")
    _add_gate(run)

    cm = sub.add_parser("compare-models", help="same dataset + prompt across several models")
    _add_common(cm)
    cm.add_argument("--prompt", required=True)
    cm.add_argument("--models", required=True, help="comma-separated model names")

    cp = sub.add_parser("compare-prompts", help="same dataset + model across several prompts")
    _add_common(cp)
    cp.add_argument("--prompts", required=True, help="comma-separated prompt files")
    cp.add_argument("--model", default=DEFAULT_MODEL)

    cmp_ = sub.add_parser("compare", help="regression check: saved run vs baseline (exit 1 on regression)")
    cmp_.add_argument("--run", default="latest")
    cmp_.add_argument("--baseline", required=True, help="baseline name or path")
    _add_gate(cmp_)

    sb = sub.add_parser("set-baseline", help="promote a saved run to be a baseline")
    sb.add_argument("--run", default="latest")
    sb.add_argument("--name", required=True)
    return p


def _setup(args):
    cases = load_dataset(args.dataset)
    scorer = build_composite(args.scorers, args.judge_model, args.pass_score)
    return cases, scorer, RunnerConfig(concurrency=args.concurrency)


def cmd_run(args) -> int:
    cases, scorer, runner_cfg = _setup(args)
    template, version = load_prompt(args.prompt)
    cfg = ModelConfig(name=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    run = evaluate(cases, cfg, template, version, scorer, runner_cfg)

    store = ResultsStore()
    if not args.no_save:
        log.info("Saved %s", store.save(run))

    report, code = None, 0
    if args.baseline:
        report = detect(
            run, store.load_baseline(args.baseline), args.max_drop, args.case_drop,
            args.max_regressed_fraction, require_same_dataset=not args.allow_dataset_change,
        )
        code = 0 if report.passed else 1
    print(generate_report(run, report))
    return code


def _print_pair_test(runs: dict) -> None:
    labels = list(runs)
    if len(labels) == 2:
        r = compare_runs(runs[labels[0]], runs[labels[1]])
        print(f"\n{labels[1]} vs {labels[0]}: diff={r.mean_diff:+.3f} "
              f"95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}] p={r.p_value:.3f} "
              f"{'(significant)' if r.significant else '(not significant)'}")


def cmd_compare_models(args) -> int:
    cases, scorer, runner_cfg = _setup(args)
    template, version = load_prompt(args.prompt)
    runs = compare_models(cases, args.models.split(","), template, version, scorer, runner_cfg=runner_cfg)
    if not args.no_save:
        store = ResultsStore()
        for run in runs.values():
            store.save(run)
    print(comparison_table(runs).to_string())
    _print_pair_test(runs)
    return 0


def cmd_compare_prompts(args) -> int:
    cases, scorer, runner_cfg = _setup(args)
    runs = compare_prompts(cases, args.prompts.split(","), ModelConfig(name=args.model), scorer, runner_cfg)
    if not args.no_save:
        store = ResultsStore()
        for run in runs.values():
            store.save(run)
    print(comparison_table(runs).to_string())
    _print_pair_test(runs)
    return 0


def cmd_compare(args) -> int:
    store = ResultsStore()
    report = detect(
        store.load(args.run), store.load_baseline(args.baseline), args.max_drop, args.case_drop,
        args.max_regressed_fraction, require_same_dataset=not args.allow_dataset_change,
    )
    print(report.to_text())
    return 0 if report.passed else 1


def cmd_set_baseline(args) -> int:
    print(f"Baseline saved: {ResultsStore().set_baseline(args.run, args.name)}")
    return 0


COMMANDS = {
    "run": cmd_run,
    "compare-models": cmd_compare_models,
    "compare-prompts": cmd_compare_prompts,
    "compare": cmd_compare,
    "set-baseline": cmd_set_baseline,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())