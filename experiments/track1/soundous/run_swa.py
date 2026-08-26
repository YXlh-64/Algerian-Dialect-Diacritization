"""BiLSTM-CNN-CRF + Stochastic Weight Averaging over cosine-restart cycle snapshots.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import os

from experiments.track1.soundousndous._bootstrap import load_everything, load_config
from training.track1.soundousndous.experiment_trainers import train_swa


def main():
    ctx = load_everything()
    cfg = load_config("swa")
    out_dir = os.path.join(ctx["paths"]["checkpoints_dir"], "exp_swa")

    train_swa(
        ctx["vocab_size"], ctx["num_classes"], ctx["pad_idx"], ctx["device"], ctx["model_kwargs"],
        ctx["train_loader"], ctx["dev_loader"], out_dir, **cfg,
    )


if __name__ == "__main__":
    main()
