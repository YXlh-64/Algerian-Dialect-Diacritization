"""Dispatcher-compatible entry point for the standard CANINE-S experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.track2.canine_s_model.train_canine_s import main


if __name__ == "__main__":
    main()
