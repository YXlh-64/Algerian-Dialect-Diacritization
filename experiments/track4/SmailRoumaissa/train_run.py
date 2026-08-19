import itertools
from pathlib import Path

import torch

from configs.track4.SmailRoumaissa.model_config import ModelConfig
from configs.track4.SmailRoumaissa.training_config import TrainingConfig, LexicalFusionConfig
from configs.track4.SmailRoumaissa.paths import find_data_paths
from utils.track4.SmailRoumaissa.device import get_device
from utils.track4.SmailRoumaissa.data import Vocab
from evaluation.track4.SmailRoumaissa.lexical_prior import LexicalPrior
from evaluation.track4.SmailRoumaissa.inference import evaluate_lexical_on_dev
from training.track4.SmailRoumaissa.trainer import train_model


def run_training(out_root: str = "/kaggle/working/runs"):
    device = get_device()
    paths = find_data_paths()
    missing = paths.missing()
    if missing:
        print("MISSING:", missing)
        return

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    model, vocab, config = train_model(
        paths.train, paths.dev, paths.vocab, Path(out_root) / "crfcnn",
        model_cfg=model_cfg, train_cfg=train_cfg,
        device=device,
    )

    lexical = LexicalPrior().fit(paths.train)

    ENTROPY_THRESHOLD_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]
    GATE_TEMPERATURE_GRID = [0.15, 0.3, 0.5]
    MAX_STRENGTH_GRID = [1.0, 2.0, 3.0]

    best = {"score": -1.0, "entropy_threshold": None, "gate_temperature": None, "max_strength": None}
    results = []

    for entropy_threshold, gate_temperature, max_strength in itertools.product(
            ENTROPY_THRESHOLD_GRID, GATE_TEMPERATURE_GRID, MAX_STRENGTH_GRID):
        _, fused_f1 = evaluate_lexical_on_dev(
            model, vocab, paths.dev,
            config["base_temperature"], config["shadda_temperature"], lexical,
            entropy_threshold=entropy_threshold, gate_temperature=gate_temperature,
            max_strength=max_strength, device=device,
        )
        results.append((entropy_threshold, gate_temperature, max_strength, fused_f1))
        if fused_f1 > best["score"]:
            best.update(score=fused_f1, entropy_threshold=entropy_threshold,
                        gate_temperature=gate_temperature, max_strength=max_strength)

    print(f"\nbest on dev: threshold={best['entropy_threshold']}  gate_temp={best['gate_temperature']}  "
          f"max_strength={best['max_strength']}  ->  dev_micro_f1={best['score']:.5f}")

    return model, vocab, config, lexical, LexicalFusionConfig(
        entropy_threshold=best["entropy_threshold"],
        gate_temperature=best["gate_temperature"],
        max_strength=best["max_strength"],
    )


if __name__ == "__main__":
    run_training()
