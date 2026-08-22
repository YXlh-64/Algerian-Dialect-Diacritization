import json
import subprocess
import sys
from pathlib import Path

import torch

from configs.track4.SmailRoumaissa.paths import find_data_paths
from utils.track4.SmailRoumaissa.device import get_device
from utils.track4.SmailRoumaissa.data import Vocab
from utils.track4.SmailRoumaissa.constants import NUM_CLASSES
from evaluation.track4.SmailRoumaissa.lexical_prior import LexicalPrior
from evaluation.track4.SmailRoumaissa.inference import run_inference


def run_prediction(model, vocab, config, lexical,
                   entropy_threshold=0.75, gate_temperature=0.5, max_strength=3.0,
                   out_root: str = "/kaggle/working"):
    device = get_device()
    paths = find_data_paths()
    missing = paths.missing()
    if missing:
        print("MISSING:", missing)
        return

    out_path = Path(out_root) / "model_output.txt"
    run_inference(
        model, vocab, config["base_temperature"], config["shadda_temperature"],
        paths.raw_test, str(out_path),
        lexical=lexical, entropy_threshold=entropy_threshold,
        gate_temperature=gate_temperature, max_strength=max_strength,
        device=device,
    )

    cmd = [
        sys.executable, paths.make_submission_py,
        "--ids", paths.raw_test_ids,
        "--input", paths.raw_test,
        "--pred", str(out_path),
        "--out", str(Path(out_root) / "submission.csv"),
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    else:
        import pandas as pd
        sub = pd.read_csv(str(Path(out_root) / "submission.csv"))
        print(sub.shape)
        print(sub.head())


if __name__ == "__main__":
    run_prediction(None, None, None, None)
