"""Evaluate BoundaryCRF-v8 as a controlled final-ensemble replacement."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, List, Mapping

from experiments.track4.Lyes.campaign.export import export_architecture_ensemble
from experiments.track4.Lyes.dual_rope_boundary_crf_v8 import load_v8_config
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


ARTIFACT_PREFIX = "DZIRI_FINAL_BOUNDARY_CRF_ENSEMBLE_V8"
SYSTEM_NAME = "DziriFinal-BoundaryCRF-Ensemble-v8"
DESCRIPTION = (
    "Four-group equal probability Track 4 ensemble replacing only the "
    "ordinary DualRoPE-CRF-v7 expert with the accepted seed-42 "
    "DualRoPE-BoundaryCRF-v8 expert. The other groups remain the three-seed "
    "DualRoPE-CE-v6 mean, three-seed HGL-v4 mean, and validated five-model "
    "legacy mean. Every group contributes exactly 1/4 and the unchanged "
    "confidence-gated V2 lexical fallback is applied after averaging."
)


def _existing_paths(values: List[str], label: str) -> List[Path]:
    paths = [Path(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "{} checkpoints are missing: {}".format(label, missing)
        )
    return paths


def run(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    config = load_v8_config(config_path)
    boundary = _existing_paths(
        [str(config["boundary_crf_checkpoint"])], "BoundaryCRF"
    )
    dual = _existing_paths(
        [str(value) for value in config["dual_seed_checkpoints"]],
        "DualRoPE CE",
    )
    hgl = _existing_paths(
        [str(value) for value in config["hgl_checkpoints"]], "HGL"
    )
    legacy = _existing_paths(
        [str(value) for value in config["legacy_ensemble_checkpoints"]],
        "legacy ensemble",
    )
    output_dir = Path(str(config["output_root"])) / (
        "02_boundary_crf_final_ensemble"
    )
    manifest = export_architecture_ensemble(
        checkpoint_groups=[boundary, dual, hgl, legacy],
        output_dir=output_dir,
        artifact_prefix=ARTIFACT_PREFIX,
        system_name=SYSTEM_NAME,
        device=select_device(device_name),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    correct = int(manifest["dev"]["v2"]["correct"])
    threshold = int(
        config["acceptance"]["boundary_ensemble_v2_must_exceed_correct"]
    )
    accepted = correct > threshold
    submission_path = Path(
        str(manifest["v2_artifacts"]["submission_path"])
    )
    canonical_path = Path(str(config["output_root"])) / (
        "SUBMIT_THIS_DZIRI_FINAL_BOUNDARY_CRF_V8.csv"
    )
    if accepted:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(submission_path, canonical_path)
        if sha256_file(canonical_path) != sha256_file(submission_path):
            raise RuntimeError("canonical submission copy hash mismatch")
    elif canonical_path.exists():
        raise RuntimeError(
            "stale canonical v8 submission exists for a rejected result"
        )
    decision = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "description": DESCRIPTION,
        "dev_neural_micro_f1": manifest["dev"]["neural"]["micro_f1"],
        "dev_neural_correct": manifest["dev"]["neural"]["correct"],
        "dev_v2_micro_f1": manifest["dev"]["v2"]["micro_f1"],
        "dev_v2_correct": correct,
        "acceptance_operator": ">",
        "acceptance_correct": threshold,
        "accepted": accepted,
        "recommended_submission_path": (
            str(canonical_path) if accepted else None
        ),
        "candidate_submission_path": str(submission_path),
        "submission_sha256": (
            sha256_file(submission_path) if accepted else None
        ),
        "do_not_submit_neural_path": str(
            manifest["neural_artifacts"]["submission_path"]
        ),
    }
    write_json(output_dir / "SELECTION.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/dual_rope_v8/campaign.json"),
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
