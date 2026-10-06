# LLM Eval Framework

A toolkit for systematically testing and comparing LLM outputs on a specific task. It turns "I tweaked the prompt and it feels better" into "I tweaked the prompt and here's the data showing it's actually better."

Run test cases through one or more models or prompts, score the outputs with multiple methods, track results over time, and catch regressions before they ship.

---

## Features

- **Dataset management**: load test cases from JSON, JSONL, or CSV, with metadata for category and difficulty breakdowns.
- **Task runner**: parallel execution with rate limiting, retries, and timeout handling.
- **Multiple scorers**: exact match, fuzzy match, schema validation, semantic similarity, LLM-as-judge, and human review.
- **Composite scoring**: combine several scorers into one per-example and aggregate score.
- **Comparison**: run the same test set across models or prompt variants and compare side by side, with significance testing.
- **Regression tracking**: store every run and flag drops against a baseline.
- **Reporting**: CLI reports, a Streamlit dashboard, and failure analysis.
- **CI gate** (optional): fail the build when scores regress.

## How It Works

```
Test dataset (inputs + expected output / criteria)
            ↓
    task_runner.py        run each case through the model/prompt under test
            ↓
    scorers/              exact match / LLM judge / semantic similarity / human eval
            ↓
    composite_scorer.py   per-example + aggregate scores
            ↓
    results_store.py      save this run
            ↓
    regression_detector.py   compare vs baseline / previous runs
            ↓
    report_generator.py / dashboard.py   scores, failures, trends
            ↓
    [optional] ci/gate_on_regression.py  block deploy on regression
```

## Project Structure

```
llm-eval-framework/
├── app/
│   ├── main.py                          # CLI entry point
│   ├── config.py                        # API keys, model configs, thresholds
│   ├── datasets/
│   │   ├── dataset_loader.py            # Load test cases (JSON/CSV/JSONL)
│   │   ├── dataset_builder.py           # Create/curate test cases
│   │   └── schemas.py                   # Test case format
│   ├── runners/
│   │   ├── task_runner.py               # Execute cases against target model/pipeline
│   │   ├── batch_executor.py            # Parallel execution + rate limiting
│   │   └── retry_handler.py             # API failures and timeouts
│   ├── scorers/
│   │   ├── base_scorer.py               # Abstract scorer interface
│   │   ├── exact_match_scorer.py
│   │   ├── fuzzy_match_scorer.py
│   │   ├── schema_validator_scorer.py
│   │   ├── semantic_similarity_scorer.py
│   │   ├── llm_judge_scorer.py
│   │   └── composite_scorer.py
│   ├── comparison/
│   │   ├── model_comparator.py
│   │   ├── prompt_comparator.py
│   │   └── ab_test_analyzer.py          # Statistical significance
│   ├── tracking/
│   │   ├── results_store.py
│   │   ├── regression_detector.py
│   │   └── history_tracker.py
│   ├── human_eval/
│   │   ├── review_interface.py
│   │   └── inter_rater_agreement.py
│   ├── reporting/
│   │   ├── report_generator.py
│   │   ├── dashboard.py                 # Streamlit
│   │   └── failure_analyzer.py
│   └── utils/
│       └── logger.py
├── ci/
│   ├── run_eval_suite.py
│   └── gate_on_regression.py
├── data/
│   ├── test_sets/                       # Curated test cases per task
│   ├── results/                         # Historical runs (JSONL per run)
│   └── baselines/                       # Reference scores
├── tests/
├── .env.example
├── requirements.txt
└── run_eval.sh
```

## Installation

```bash
git clone https://github.com/masudibnmusa/Evaluation-Benchmarking-Framework.git
cd llm-eval-framework

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # then add your API keys
```

**Requirements:** Python 3.10+ and an API key for each model provider you want to evaluate (and for the judge model, if you use LLM-as-judge).

## Quick Start

> The commands below show the intended CLI. Adjust flags to match your implementation as it develops.

```bash
# 1. Run an evaluation
python -m app.main run \
  --dataset data/test_sets/my_task.jsonl \
  --model claude-sonnet-4-6 \
  --prompt prompts/v1.txt \
  --scorers exact_match,llm_judge

# 2. Compare against the saved baseline
python -m app.main compare --run latest --baseline data/baselines/my_task.json

# 3. Open the dashboard
streamlit run app/reporting/dashboard.py
```

Or use the wrapper script:

```bash
./run_eval.sh
```

## Test Case Format

Each test case is one JSON object (one per line in JSONL):

```json
{
  "id": "case_001",
  "input": "Summarize: The quick brown fox...",
  "expected": "A fox jumps over a dog.",
  "criteria": ["Mentions the fox", "Under 20 words"],
  "metadata": {
    "category": "summarization",
    "difficulty": "easy",
    "tags": ["short-input"]
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Unique, stable identifier |
| `input` | yes | What is sent to the model |
| `expected` | depends | Reference answer (exact match, fuzzy, semantic similarity) |
| `criteria` | depends | Rubric items (LLM-as-judge, human review) |
| `metadata` | no | Category, difficulty, tags for breakdowns |

## Scoring Methods

Pick scorers based on the task type.

| Scorer | Best for | Notes |
|--------|----------|-------|
| **Exact match** | Classification, extraction | Normalize case and whitespace as needed |
| **Fuzzy match** | Near-correct strings | Levenshtein or partial match with a threshold |
| **Schema validator** | Structured output (JSON) | Validates schema, types, constraints; regex and length checks |
| **Semantic similarity** | Paraphrase-tolerant tasks | Embedding distance to a reference answer |
| **LLM-as-judge** | Summarization, writing quality | Scored against a rubric; validate against human labels |
| **Human review** | Nuanced quality | Rater UI plus inter-rater agreement |
| **Composite** | Everything | Weighted combination of the above |

### Custom scorers

Subclass `BaseScorer` and implement `score`:

```python
from app.scorers.base_scorer import BaseScorer

class ContainsKeywordScorer(BaseScorer):
    name = "contains_keyword"

    def score(self, case, output) -> float:
        return 1.0 if case.metadata["keyword"] in output else 0.0
```

### Using LLM-as-judge well

Judges have known biases (verbosity, position, self-preference). To keep them trustworthy:

- Use a strong judge model at temperature 0, with an explicit rubric.
- Label a small set by hand and measure judge-vs-human agreement.
- Randomize answer order in pairwise comparisons.
- Consider running the judge multiple times and aggregating.

## Comparing Models and Prompts

```bash
# Same test set, multiple models
python -m app.main compare-models \
  --dataset data/test_sets/my_task.jsonl \
  --models claude-sonnet-4-6,claude-haiku-4-5

# Same test set, multiple prompt variants
python -m app.main compare-prompts \
  --dataset data/test_sets/my_task.jsonl \
  --prompts prompts/v1.txt,prompts/v2.txt
```

Results are shown side by side, broken down by category and difficulty. The A/B analyzer reports whether differences are statistically significant, since on small test sets a few points of difference is often just noise.

## Regression Tracking

Every run is stored in `data/results/` as JSONL. Each record includes:

- the raw model output (stored separately from scores, so you can re-score without re-calling the model)
- model name, parameters, and prompt version
- a hash of the dataset (so runs are only compared against the same test set)
- per-scorer scores, latency, and token usage

The regression detector compares a run against a baseline and flags drops. Because model output is non-deterministic, detection is based on tolerances and aggregate thresholds rather than any single case flipping.

```bash
# Promote a run to be the new baseline
python -m app.main set-baseline --run <run_id> --name my_task
```

## Reporting

- **CLI report**: overall metrics, per-category breakdown, top failures.
- **Dashboard** (Streamlit): scores, trends over time, run comparisons, failure browser, and the human review interface.
- **Failure analyzer**: groups failing cases to surface common patterns.

## CI Integration

```bash
python ci/run_eval_suite.py
python ci/gate_on_regression.py --max-drop 0.02
```

Example GitHub Actions step:

```yaml
- name: Run eval suite
  run: |
    python ci/run_eval_suite.py
    python ci/gate_on_regression.py --max-drop 0.02
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The gate exits non-zero when aggregate scores fall below the baseline by more than the allowed threshold.

## Configuration

Secrets live in `.env`:

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Run settings live in `app/config.py` (or a YAML experiment file):

```yaml
experiment: my_task_v2
dataset: data/test_sets/my_task.jsonl
model:
  name: claude-sonnet-4-6
  temperature: 0
  max_tokens: 1024
prompt: prompts/v2.txt
scorers:
  - name: exact_match
    weight: 0.5
  - name: llm_judge
    weight: 0.5
    judge_model: claude-sonnet-4-6
runner:
  concurrency: 8
  max_retries: 3
  timeout_seconds: 60
thresholds:
  max_aggregate_drop: 0.02
```

## Design Principles

1. **Separate generation from scoring.** Raw outputs are saved so scoring can be re-run cheaply.
2. **Everything is traceable.** Every score links to the exact prompt, model, parameters, and dataset version that produced it.
3. **Scorers are pluggable.** Adding a method never requires touching the runner.
4. **Respect noise.** Use significance tests and tolerances, not single-run comparisons.
5. **Reproducible by default.** Experiments are defined in config, not in someone's shell history.

## Testing

```bash
pytest tests/
```

## License

MIT