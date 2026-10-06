"""CI gate: exit 1 if the latest run regressed vs the baseline.
  python ci/gate_on_regression.py --baseline my_task --max-drop 0.02
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["compare", *sys.argv[1:]]))