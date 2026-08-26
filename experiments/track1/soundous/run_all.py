"""Runs every training script in sequence"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time

import experiments.track1.soundous.run_base_architectures as run_base_architectures
import experiments.track1.soundous.run_focal as run_focal
import experiments.track1.soundous.run_multitask as run_multitask
import experiments.track1.soundous.run_attention as run_attention
import experiments.track1.soundous.run_consistency as run_consistency
import experiments.track1.soundous.run_swa as run_swa
import experiments.track1.soundous.run_ensemble as run_ensemble
import experiments.track1.soundous.run_curriculum as run_curriculum

RUNS = [
    ("base architectures (bilstm_cnn / bilstm_crf / bilstm_cnn_crf)", run_base_architectures.main),
    ("focal loss ", run_focal.main),
    ("multi-task head ", run_multitask.main),
    ("self-attention ", run_attention.main),
    ("consistency regularization ", run_consistency.main),
    ("SWA", run_swa.main),
    ("multi-seed ensemble", run_ensemble.main),
    ("curriculum learning", run_curriculum.main),
]


def main():
    for name, fn in RUNS:
        print("\n" + "#" * 80)
        print(f"# {name}")
        print("#" * 80)
        t0 = time.time()
        fn()
        print(f"# done in {time.time() - t0:.1f}s")

    print("\nAll experiments trained. Next: run "
          "evaluation/track1/soundous/evaluate_all_experiments.py to generate test-set submissions "
          "and the metrics report.")


if __name__ == "__main__":
    main()
