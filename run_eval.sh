#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .venv ]; then
  source .venv/bin/activate
fi

DATASET="${DATASET:-data/test_sets/my_task.jsonl}"
PROMPT="${PROMPT:-prompts/v1.txt}"
MODEL="${MODEL:-claude-sonnet-4-6}"
SCORERS="${SCORERS:-exact_match}"

python -m app.main run \
  --dataset "$DATASET" \
  --prompt "$PROMPT" \
  --model "$MODEL" \
  --scorers "$SCORERS" \
  "$@"