"""Select, prepare, and export the final train+dev architecture-v10 refit."""

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import predict_with_gated_fallback
from experiments.track4.Lyes.campaign.common import (
    write_prediction_artifacts,
    write_step_manifest,
)
from experiments.track4.Lyes.campaign.folds import write_records_jsonl
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.config import load_config, validate_config
from utils.track4.Lyes.data import SentenceRecord, load_jsonl, load_raw_sentences
from evaluation.track4.Lyes.infer import predict_records
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


CONTROL = {
    "slug": "dual_rope_crf_v7_control",
    "system_name": "DziriFormer-DualRoPE-CRF-v7-seed42",
    "config": "configs/track4/Lyes/model.json",
    "selection": "outputs/dual_rope_v7/03_crf_seed42/SELECTION.json",
    "neural_correct": 14816,
    "parameter_count": 9890096,
}
EXPERIMENT_CONFIGS = {
    "wordpos_crf": "configs/track4/Lyes/architecture_v10/wordpos_crf_v10.json",
    "factorized_crf": (
        "configs/track4/Lyes/architecture_v10/factorized_emission_crf_v10.json"
    ),
    "low_rank_boundary_crf": (
        "configs/track4/Lyes/architecture_v10/low_rank_boundary_crf_v10.json"
    ),
}
SELECTION_PATHS = {
    "wordpos_crf": (
        "outputs/architecture_v10/02_wordpos_crf/seed_42/SELECTION.json"
    ),
    "factorized_crf": (
        "outputs/architecture_v10/03_factorized_emission_crf/"
        "seed_42/SELECTION.json"
    ),
    "low_rank_boundary_crf": (
        "outputs/architecture_v10/04_low_rank_boundary_crf/"
        "seed_42/SELECTION.json"
    ),
}
FINAL_ROOT = Path("outputs/architecture_v10/05_final_train_dev_refit")
PREFIX = "DZIRI_FINAL_TRAIN_DEV_REFIT_V10"


def choose_candidate(
    experiment_selections: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = [dict(CONTROL)]
    for selection in experiment_selections:
        if bool(selection["architecture_accepted"]):
            slug = str(selection["experiment"])
            candidates.append(
                {
                    "slug": slug,
                    "system_name": str(selection["system_name"]),
                    "config": EXPERIMENT_CONFIGS[slug],
                    "selection": SELECTION_PATHS[slug],
                    "neural_correct": int(selection["neural_correct"]),
                    "parameter_count": int(selection["parameter_count"]),
                }
            )
    candidates.sort(
        key=lambda value: (
            -int(value["neural_correct"]),
            int(value["parameter_count"]),
            str(value["slug"]),
        )
    )
    return candidates[0]


def _load_completed_selections() -> Sequence[Mapping[str, Any]]:
    selections = []
    missing = []
    for slug, raw_path in SELECTION_PATHS.items():
        path = Path(raw_path)
        if not path.is_file():
            missing.append(slug)
            continue
        with path.open("r", encoding="utf-8") as stream:
            selection = json.load(stream)
        if selection.get("experiment") != slug:
            raise ValueError("architecture selection slug mismatch")
        selections.append(selection)
    if missing:
        raise FileNotFoundError(
            "refit selection requires completed experiments: {}".format(
                ", ".join(missing)
            )
        )
    return selections


def prepare() -> Mapping[str, Any]:
    selections = _load_completed_selections()
    selected = choose_candidate(selections)
    train_path = Path("Data/train_data/train_Algerian-DIAC.jsonl")
    dev_path = Path("Data/dev_data/dev_Algerian-DIAC.jsonl")
    train_records = load_jsonl(train_path)
    dev_records = load_jsonl(dev_path)
    train_ids = {record.sent_id for record in train_records}
    dev_ids = {record.sent_id for record in dev_records}
    overlap = sorted(train_ids.intersection(dev_ids))
    if overlap:
        raise ValueError(
            "train/dev sentence IDs overlap: {}".format(overlap[:3])
        )
    combined = tuple(train_records) + tuple(dev_records)
    data_dir = FINAL_ROOT / "data"
    combined_path = data_dir / "train_plus_dev.jsonl"
    write_records_jsonl(combined_path, combined)

    source_config = load_config(Path(str(selected["config"])))
    resolved = copy.deepcopy(source_config)
    batch_size = int(resolved["training"]["batch_size"])
    source_updates = (
        math.ceil(len(train_records) / batch_size)
        * int(resolved["training"]["epochs"])
    )
    combined_updates_per_epoch = math.ceil(len(combined) / batch_size)
    refit_epochs = max(
        1, int(round(source_updates / combined_updates_per_epoch))
    )
    resolved["data"]["train"] = str(combined_path)
    resolved["data"]["dev"] = str(dev_path)
    resolved["training"]["epochs"] = refit_epochs
    resolved["training"]["selection_mode"] = "last_epoch"
    resolved["training"]["early_stopping_patience"] = refit_epochs + 1
    resolved["training"]["num_workers"] = 0
    resolved["training"]["device"] = "mps"
    resolved["output_dir"] = str(FINAL_ROOT / "seed_42")
    validate_config(resolved)
    config_path = FINAL_ROOT / "refit_config.json"
    write_json(config_path, resolved)
    plan = {
        "schema_version": 1,
        "selected": selected,
        "selection_basis": (
            "highest accepted pre-refit neural correct count; ties use fewer "
            "parameters, then lexical slug order"
        ),
        "source_train": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "sentences": len(train_records),
        },
        "source_dev": {
            "path": str(dev_path),
            "sha256": sha256_file(dev_path),
            "sentences": len(dev_records),
        },
        "combined_train": {
            "path": str(combined_path),
            "sha256": sha256_file(combined_path),
            "sentences": len(combined),
        },
        "source_optimizer_updates": source_updates,
        "refit_epochs": refit_epochs,
        "refit_updates_per_epoch": combined_updates_per_epoch,
        "refit_optimizer_updates": (
            refit_epochs * combined_updates_per_epoch
        ),
        "checkpoint_selection": "last_epoch",
        "dev_metrics_valid": False,
        "refit_config": str(config_path),
        "training_command": (
            ".venv/bin/python -m training.track4.Lyes.train --config {} --device mps "
            "--num-workers 0".format(config_path)
        ),
    }
    write_json(FINAL_ROOT / "REFIT_PLAN.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def export(
    checkpoint_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    plan_path = FINAL_ROOT / "REFIT_PLAN.json"
    if not plan_path.is_file():
        raise FileNotFoundError("prepare the refit before export")
    with plan_path.open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("refit checkpoint is missing")
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    train_plus_dev = load_jsonl(
        Path(str(plan["combined_train"]["path"]))
    )
    prior = WordLabelPrior().fit(train_plus_dev)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))
    test_records = load_raw_sentences(
        Path("Data/test_data/raw_sentences_test.txt"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    neural = predict_records(
        model, test_records, vocab, device, batch_size, num_workers
    )
    v2, test_gate_statistics = predict_with_gated_fallback(
        model,
        test_records,
        vocab,
        prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    artifact_dir = FINAL_ROOT / "artifacts"
    neural_artifacts = write_prediction_artifacts(
        artifact_dir,
        PREFIX + "_NEURAL",
        test_records,
        neural,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    v2_artifacts = write_prediction_artifacts(
        artifact_dir,
        PREFIX + "_V2",
        test_records,
        v2,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    manifest = {
        "schema_version": 1,
        "system_name": "DziriFinal-TrainDev-Refit-v10",
        "description": (
            "Competition-only Track 4 refit of the selected architecture on "
            "the released train and dev sentences. It matches the original "
            "optimizer-update budget as closely as possible and selects the "
            "last epoch without dev-based checkpoint selection. The V2 "
            "lexical fallback is also fit on train+dev. No unbiased dev "
            "metric exists for this artifact."
        ),
        "selected_pre_refit_system": plan["selected"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "dev_metrics_valid": False,
        "test_v2_gate_statistics": test_gate_statistics.to_dict(),
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
        "recommended_submission_path": v2_artifacts["submission_path"],
        "do_not_use_for_paper_dev_table": True,
    }
    write_step_manifest(FINAL_ROOT / "MANIFEST.json", manifest)
    write_json(FINAL_ROOT / "SUBMISSION_GUIDE.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=FINAL_ROOT / "seed_42" / "best.pt",
    )
    export_parser.add_argument("--device", default="auto")
    export_parser.add_argument("--batch-size", type=int, default=128)
    export_parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        export(
            checkpoint_path=args.checkpoint,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
