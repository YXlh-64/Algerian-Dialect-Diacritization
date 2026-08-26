"""Trains BiLSTM-CNN, BiLSTM-CRF, and BiLSTM-CNN-CRF  as a controlled
ablation -- same capacity , only use_cnn/use_crf differ.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import os

from experiments.track1.soundous._bootstrap import load_everything, load_config
from training.track1.soundousndous.train_loop import train_model


def main():
    ctx = load_everything()
    cfg = load_config("base")

    for arch_name in ["bilstm_cnn", "bilstm_crf", "bilstm_cnn_crf"]:
        print("=" * 70, f"\nTraining {arch_name}\n", "=" * 70)
        out_dir = os.path.join(ctx["paths"]["checkpoints_dir"], arch_name)
        train_model(
            arch_name, ctx["model_kwargs"], ctx["train_loader"], ctx["dev_loader"], out_dir,
            ctx["device"], ctx["vocab_size"], ctx["num_classes"], ctx["pad_idx"], **cfg,
        )


if __name__ == "__main__":
    main()
