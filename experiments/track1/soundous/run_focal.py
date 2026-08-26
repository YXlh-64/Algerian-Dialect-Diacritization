"""BiLSTM-CNN-CRF + focal-style emission reweighting for rare diacritic classes.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import os

from experiments.track1.soundousndous._bootstrap import load_everything, load_config
from training.track1.soundousndous.experiment_trainers import train_focal


def main():
    ctx = load_everything()
    cfg = load_config("focal")
    out_dir = os.path.join(ctx["paths"]["checkpoints_dir"], "exp_focal")

    train_focal(
        ctx["vocab_size"], ctx["num_classes"], ctx["pad_idx"], ctx["device"], ctx["model_kwargs"],
        ctx["train_rows"], ctx["label2idx"], ctx["no_diac_idx"], ctx["train_loader"], ctx["dev_loader"],
        out_dir, **cfg,
    )


if __name__ == "__main__":
    main()
