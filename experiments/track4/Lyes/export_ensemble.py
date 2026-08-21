"""Export deterministic DualRoPE v7 ensemble candidates and submit guidance."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from experiments.track4.Lyes.campaign.export import export_architecture_ensemble
from utils.track4.Lyes.utils import select_device, write_json


EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "output_root",
    "dual_seed_checkpoints",
    "crf_checkpoint",
    "hgl_checkpoints",
    "legacy_ensemble_checkpoints",
    "acceptance",
}
EXPECTED_ACCEPTANCE_KEYS = {
    "seed42_equal_system_v2_correct",
    "dual_multiseed_must_exceed_correct",
    "multiseed_must_exceed_correct",
    "crf_neural_must_exceed_correct",
    "crf_v2_must_exceed_correct",
    "crf_final_must_exceed_correct",
}


def load_v7_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != EXPECTED_CONFIG_KEYS:
        raise ValueError("invalid DualRoPE v7 campaign configuration keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported DualRoPE v7 campaign schema")
    dual = config["dual_seed_checkpoints"]
    if not isinstance(dual, dict) or set(dual) != {"42", "43", "44"}:
        raise ValueError("dual_seed_checkpoints must define 42, 43, and 44")
    for key in ("hgl_checkpoints", "legacy_ensemble_checkpoints"):
        paths = config[key]
        if not isinstance(paths, list) or not paths:
            raise ValueError("{} must be a nonempty list".format(key))
    acceptance = config["acceptance"]
    if (
        not isinstance(acceptance, dict)
        or set(acceptance) != EXPECTED_ACCEPTANCE_KEYS
    ):
        raise ValueError("invalid DualRoPE v7 acceptance gates")
    if any(int(value) <= 0 for value in acceptance.values()):
        raise ValueError("all DualRoPE v7 acceptance gates must be positive")
    return config


def _existing_paths(values: Sequence[str], label: str) -> List[Path]:
    paths = [Path(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "{} checkpoint files are missing: {}".format(label, missing)
        )
    return paths


def checkpoint_groups(
    config: Mapping[str, Any], stage: str
) -> List[List[Path]]:
    dual_config = config["dual_seed_checkpoints"]
    dual_seeds = ["42"] if stage == "seed42" else ["42", "43", "44"]
    dual = _existing_paths(
        [str(dual_config[seed]) for seed in dual_seeds],
        "DualRoPE",
    )
    hgl = _existing_paths(
        [str(value) for value in config["hgl_checkpoints"]],
        "HGL",
    )
    legacy = _existing_paths(
        [str(value) for value in config["legacy_ensemble_checkpoints"]],
        "legacy ensemble",
    )
    if stage == "dual_multiseed":
        return [dual]
    if stage == "crf_final":
        crf = _existing_paths(
            [str(config["crf_checkpoint"])], "DualRoPE CRF"
        )
        return [crf, dual, hgl, legacy]
    return [dual, hgl, legacy]


def _stage_contract(stage: str) -> Mapping[str, str]:
    if stage == "seed42":
        return {
            "directory": "01_seed42_equal_system",
            "artifact_prefix": "DZIRI_ENSEMBLE_DUALROPE_HGL_V7_SEED42",
            "system_name": "DziriEnsemble-DualRoPE-HGL-v7-seed42",
            "description": (
                "Equal probability-group Track 4 ensemble of seed-42 "
                "DziriFormer-DualRoPE-CE-v6, the three-seed HGL-v4 expert, "
                "and the validated five-model legacy expert. The unchanged "
                "confidence-gated V2 lexical fallback is applied after "
                "neural probability averaging."
            ),
        }
    if stage == "multiseed":
        return {
            "directory": "02_multiseed_equal_system",
            "artifact_prefix": "DZIRI_ENSEMBLE_DUALROPE_HGL_V7_MULTI",
            "system_name": "DziriEnsemble-DualRoPE-HGL-v7-multiseed",
            "description": (
                "Equal probability-group Track 4 ensemble of a three-seed "
                "DualRoPE-v6 expert, the three-seed HGL-v4 expert, and the "
                "validated five-model legacy expert. Seeds are averaged "
                "within their systems before equal system averaging; the "
                "unchanged confidence-gated V2 fallback is applied last."
            ),
        }
    if stage == "dual_multiseed":
        return {
            "directory": "02a_dual_multiseed",
            "artifact_prefix": "DZIRIFORMER_DUALROPE_CE_V7_MULTI",
            "system_name": "DziriFormer-DualRoPE-CE-v7-multiseed",
            "description": (
                "Equal probability ensemble of DualRoPE-CE-v6 seeds 42, "
                "43, and 44. The unchanged confidence-gated V2 lexical "
                "fallback is applied only after seed probability averaging."
            ),
        }
    if stage == "crf_final":
        return {
            "directory": "04_final_crf_ensemble",
            "artifact_prefix": "DZIRI_FINAL_DUALROPE_CRF_ENSEMBLE_V7",
            "system_name": "DziriFinal-DualRoPE-CRF-Ensemble-v7",
            "description": (
                "Four-group equal probability Track 4 ensemble: the "
                "DualRoPE-CRF-v7 seed-42 expert, the three-seed "
                "DualRoPE-CE-v6 expert, the three-seed HGL-v4 expert, and "
                "the validated five-model legacy expert. The unchanged "
                "confidence-gated V2 lexical fallback is applied after "
                "probability averaging. No manual ensemble weights."
            ),
        }
    raise ValueError(
        "stage must be seed42, dual_multiseed, multiseed, or crf_final"
    )


def run(
    config_path: Path,
    stage: str,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    config = load_v7_config(config_path)
    contract = _stage_contract(stage)
    groups = checkpoint_groups(config, stage)
    output_dir = Path(str(config["output_root"])) / contract["directory"]
    manifest = export_architecture_ensemble(
        checkpoint_groups=groups,
        output_dir=output_dir,
        artifact_prefix=contract["artifact_prefix"],
        system_name=contract["system_name"],
        device=select_device(device_name),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    correct = int(manifest["dev"]["v2"]["correct"])
    if stage == "seed42":
        gate_key = "seed42_equal_system_v2_correct"
    elif stage == "dual_multiseed":
        gate_key = "dual_multiseed_must_exceed_correct"
    elif stage == "crf_final":
        gate_key = "crf_final_must_exceed_correct"
    else:
        gate_key = "multiseed_must_exceed_correct"
    threshold = int(config["acceptance"][gate_key])
    accepted = correct >= threshold if stage == "seed42" else correct > threshold
    submission_path = str(manifest["v2_artifacts"]["submission_path"])
    decision = {
        "schema_version": 1,
        "stage": stage,
        "system_name": contract["system_name"],
        "description": contract["description"],
        "dev_neural_micro_f1": manifest["dev"]["neural"]["micro_f1"],
        "dev_neural_correct": manifest["dev"]["neural"]["correct"],
        "dev_v2_micro_f1": manifest["dev"]["v2"]["micro_f1"],
        "dev_v2_correct": correct,
        "acceptance_operator": ">=" if stage == "seed42" else ">",
        "acceptance_correct": threshold,
        "accepted": accepted,
        "submission_path": submission_path,
        "submission_sha256": manifest["v2_artifacts"]["submission_sha256"],
        "do_not_submit_path": str(
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
        default=Path("configs/track4/Lyes/campaign.json"),
    )
    parser.add_argument(
        "--stage",
        choices=("seed42", "dual_multiseed", "multiseed", "crf_final"),
        required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    run(
        config_path=args.config,
        stage=args.stage,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
