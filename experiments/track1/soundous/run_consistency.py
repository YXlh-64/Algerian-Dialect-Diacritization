"""-- BiLSTM-CNN-CRF + consistency-regularized data augmentation (random char dropout + KL).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import os

from experiments.track1.soundous._bootstrap import load_everything, load_config
from training.track1.soundous.experiment_trainers import train_consistency


def main():
    ctx = load_everything()
    cfg = load_config("consistency")
    out_dir = os.path.join(ctx["paths"]["checkpoints_dir"], "exp_consistency")

    train_consistency(
        ctx["vocab_size"], ctx["num_classes"], ctx["pad_idx"], ctx["unk_idx"], ctx["device"],
        ctx["model_kwargs"], ctx["train_loader"], ctx["dev_loader"], out_dir, **cfg,
    )


if __name__ == "__main__":
    main()
