"""Evaluate equal best/last snapshot averaging for DualRoPE-CRF-v7."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from utils.track4.Lyes.gated_fusion.config import load_gates
from experiments.track4.Lyes.campaign.common import (
    write_prediction_artifacts,
    write_step_manifest,
)
from experiments.track4.Lyes.campaign.ensemble import (
    apply_lexical_gate,
    predict_probability_ensemble,
    probabilities_to_predictions,
)
from utils.track4.Lyes.data import load_jsonl, load_raw_sentences
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


ARTIFACT_PREFIX = "DZIRIFORMER_DUALROPE_CRF_V7_SNAPSHOT"
SYSTEM_NAME = "DziriFormer-DualRoPE-CRF-v7-Snapshot"
EXPECTED_ROOT_KEYS = {"schema_version", "output_root", "snapshot"}
EXPECTED_SNAPSHOT_KEYS = {
    "checkpoints",
    "control_v2_correct",
    "minimum_correct_gain",
    "production_correct",
}


def load_snapshot_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != EXPECTED_ROOT_KEYS:
        raise ValueError("invalid architecture-v10 campaign keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported architecture-v10 schema")
    snapshot = config["snapshot"]
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != EXPECTED_SNAPSHOT_KEYS
    ):
        raise ValueError("invalid snapshot campaign configuration")
    checkpoints = snapshot["checkpoints"]
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != 2
        or not all(isinstance(value, str) for value in checkpoints)
    ):
        raise ValueError("snapshot requires exactly best and last checkpoints")
    if any(
        int(snapshot[key]) <= 0
        for key in (
            "control_v2_correct",
            "minimum_correct_gain",
            "production_correct",
        )
    ):
        raise ValueError("snapshot gates must be positive")
    return config


def run(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    config = load_snapshot_config(config_path)
    snapshot = config["snapshot"]
    checkpoint_paths = [Path(value) for value in snapshot["checkpoints"]]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "snapshot checkpoints are missing: {}".format(missing)
        )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid snapshot loader settings")
    device = select_device(device_name)
    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    dev_records = load_jsonl(
        Path("Data/dev_data/dev_Algerian-DIAC.jsonl")
    )
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))
    dev_probabilities, _ = predict_probability_ensemble(
        checkpoint_paths, dev_records, device, batch_size, num_workers
    )
    neural_predictions = probabilities_to_predictions(
        dev_records, dev_probabilities
    )
    v2_predictions = apply_lexical_gate(
        dev_records, dev_probabilities, prior, gates
    )
    neural_metrics = compute_paper_metrics(
        dev_records, neural_predictions
    )
    v2_metrics = compute_paper_metrics(dev_records, v2_predictions)
    control_correct = int(snapshot["control_v2_correct"])
    correct = int(v2_metrics["correct_letters"])
    gain = correct - control_correct
    architecture_accepted = gain >= int(snapshot["minimum_correct_gain"])
    competitive = correct > int(snapshot["production_correct"])

    output_dir = Path(str(config["output_root"])) / "01_snapshot"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "neural_paper_metrics.json", neural_metrics)
    write_json(output_dir / "v2_paper_metrics.json", v2_metrics)
    artifacts: Mapping[str, Any] = {}
    if competitive:
        test_records = load_raw_sentences(
            Path("Data/test_data/raw_sentences_test.txt"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
        )
        test_probabilities, _ = predict_probability_ensemble(
            checkpoint_paths, test_records, device, batch_size, num_workers
        )
        test_predictions = apply_lexical_gate(
            test_records, test_probabilities, prior, gates
        )
        artifacts = write_prediction_artifacts(
            output_dir / "artifacts",
            ARTIFACT_PREFIX,
            test_records,
            test_predictions,
            Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
            Path("Data/test_data/raw_sentences_test.txt"),
        )
    selection = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "aggregation": "equal arithmetic probability mean of best.pt and last.pt",
        "checkpoints": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in checkpoint_paths
        ],
        "neural_correct": neural_metrics["correct_letters"],
        "neural_accuracy": neural_metrics["accuracy"],
        "v2_correct": correct,
        "v2_accuracy": v2_metrics["accuracy"],
        "control_v2_correct": control_correct,
        "correct_gain": gain,
        "minimum_correct_gain": snapshot["minimum_correct_gain"],
        "architecture_accepted": architecture_accepted,
        "production_correct": snapshot["production_correct"],
        "competitive_submission": competitive,
        "recommended_submission_path": (
            artifacts.get("submission_path") if competitive else None
        ),
        "artifacts": dict(artifacts),
    }
    write_json(output_dir / "SELECTION.json", selection)
    write_step_manifest(output_dir / "MANIFEST.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/architecture_v10/campaign.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    run(
        config_path=args.config,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
