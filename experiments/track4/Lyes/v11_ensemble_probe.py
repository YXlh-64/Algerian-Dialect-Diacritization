"""Run the fixed equal-group v7 plus context-boundary-v11 probe."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

from experiments.track4.Lyes.campaign.export import export_architecture_ensemble
from experiments.track4.Lyes.export_ensemble import (
    checkpoint_groups,
    load_v7_config,
)
from utils.track4.Lyes.utils import select_device, write_json


CONFIG_KEYS = {
    "schema_version",
    "v7_campaign_config",
    "v11_checkpoint",
    "output_dir",
    "system_name",
    "artifact_prefix",
    "baseline_v2_correct",
    "minimum_meaningful_gain",
}


def load_probe_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("invalid v11 ensemble-probe configuration keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported v11 ensemble-probe schema")
    for key in (
        "v7_campaign_config",
        "v11_checkpoint",
        "output_dir",
        "system_name",
        "artifact_prefix",
    ):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty string")
    if re.fullmatch(r"[A-Z0-9_]+", config["artifact_prefix"]) is None:
        raise ValueError("artifact_prefix must contain only A-Z, 0-9, and _")
    if int(config["baseline_v2_correct"]) <= 0:
        raise ValueError("baseline_v2_correct must be positive")
    if int(config["minimum_meaningful_gain"]) <= 0:
        raise ValueError("minimum_meaningful_gain must be positive")
    return config


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
    config = load_probe_config(config_path)
    v7_config = load_v7_config(Path(config["v7_campaign_config"]))
    groups = checkpoint_groups(v7_config, "crf_final")
    v11_checkpoint = Path(config["v11_checkpoint"])
    if not v11_checkpoint.is_file():
        raise FileNotFoundError(v11_checkpoint)
    groups.append([v11_checkpoint])
    if len(groups) != 5:
        raise RuntimeError("probe must contain exactly five architecture groups")

    output_dir = Path(config["output_dir"])
    manifest = export_architecture_ensemble(
        checkpoint_groups=groups,
        output_dir=output_dir,
        artifact_prefix=config["artifact_prefix"],
        system_name=config["system_name"],
        device=select_device(device_name),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    baseline = int(config["baseline_v2_correct"])
    required_gain = int(config["minimum_meaningful_gain"])
    neural_correct = int(manifest["dev"]["neural"]["correct"])
    v2_correct = int(manifest["dev"]["v2"]["correct"])
    gain = v2_correct - baseline
    accepted = gain >= required_gain
    neural_path = str(manifest["neural_artifacts"]["submission_path"])
    v2_path = str(manifest["v2_artifacts"]["submission_path"])
    decision = {
        "schema_version": 1,
        "system_name": config["system_name"],
        "composition": (
            "Five equal architecture-level probability experts: "
            "DualRoPE-CRF-v7 seed 42; DualRoPE-CE-v6 seed mean 42/43/44; "
            "HGL-v4 seed mean 42/43/44; legacy Base/J16/GL/Mixed/Hier "
            "five-model mean; ContextLowRankBoundaryCRF-v11 seed 42. "
            "Each expert contributes 1/5. Unchanged V2 is applied after "
            "probability averaging."
        ),
        "neural_correct": neural_correct,
        "neural_micro_f1": manifest["dev"]["neural"]["micro_f1"],
        "v2_correct": v2_correct,
        "v2_micro_f1": manifest["dev"]["v2"]["micro_f1"],
        "baseline_v2_correct": baseline,
        "v2_gain_correct": gain,
        "minimum_meaningful_gain": required_gain,
        "acceptance_correct": baseline + required_gain,
        "acceptance_operator": ">=",
        "accepted": accepted,
        "decision": "keep" if accepted else "reject",
        "probe_submission_path": v2_path,
        "recommended_submission_path": v2_path if accepted else None,
        "submission_sha256": manifest["v2_artifacts"]["submission_sha256"],
        "do_not_submit_paths": (
            [neural_path] if accepted else [neural_path, v2_path]
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
        default=Path("configs/track4/Lyes/context_boundary_v11/ensemble_probe.json"),
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
