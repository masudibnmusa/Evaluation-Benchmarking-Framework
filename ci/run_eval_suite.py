"""CI step: run the eval suite and save results.
Takes the same flags as `python -m app.main run`, e.g.
  python ci/run_eval_suite.py --dataset data/test_sets/t.jsonl --prompt prompts/v1.txt --scorers exact_match
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))