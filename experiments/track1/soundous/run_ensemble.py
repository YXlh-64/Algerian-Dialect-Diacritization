"""Multi-seed ensemble of BiLSTM-CNN-CRF. Trains N independent seeds (each saved
normally under exp_ensemble/seed_<n>/); combining happens at inference time 
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import os

from experiments.track1.soundousndous._bootstrap import load_everything, load_config
from training.track1.soundousndous.experiment_trainers import train_multi_seed


def main():
    ctx = load_everything()
    cfg = load_config("ensemble")
    seeds = cfg.pop("seeds")
    out_dir = os.path.join(ctx["paths"]["checkpoints_dir"], "exp_ensemble")

    train_multi_seed(
        ctx["vocab_size"], ctx["num_classes"], ctx["pad_idx"], ctx["device"], ctx["model_kwargs"],
        seeds, ctx["train_rows"], ctx["char2idx"], ctx["label2idx"], ctx["no_diac_idx"],
        ctx["dev_loader"], out_dir, batch_size=ctx["batch_size"], **cfg,
    )


if __name__ == "__main__":
    main()
