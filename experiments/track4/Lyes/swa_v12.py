"""Controlled checkpoint-weight probe and true SWA-tail training.

The checkpoint probe is deliberately named separately from SWA: it averages
two existing model states without training.  The SWA mode continues one
selected checkpoint at its recorded learning rate, saves one model-only
snapshot per epoch, and evaluates the arithmetic prefix mean of tail weights.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch.optim import AdamW

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import predict_with_gated_fallback
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import load_jsonl
from evaluation.track4.Lyes.infer import predict_records
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from training.track4.Lyes.train import _make_loader, _move_batch
from utils.track4.Lyes.utils import (
    append_jsonl,
    save_checkpoint,
    seed_everything,
    select_device,
    sha256_file,
    write_json,
)


CONFIG_KEYS = {"schema_version", "output_root", "systems"}
SYSTEM_KEYS = {
    "system_name",
    "artifact_prefix",
    "source_checkpoint",
    "probe_checkpoints",
    "tail_epochs",
    "maximum_tail_epochs",
    "batch_size",
    "num_workers",
    "baseline",
}
BASELINE_KEYS = {
    "neural_correct",
    "v2_correct",
    "v2_word_accuracy",
    "v2_sentence_accuracy",
    "v2_shadda_accuracy",
    "v2_tanween_accuracy",
    "minimum_correct_gain",
}


def load_swa_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("invalid SWA-v12 configuration keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported SWA-v12 configuration schema")
    systems = config["systems"]
    if not isinstance(systems, dict) or not systems:
        raise ValueError("SWA-v12 requires at least one named system")
    for key, system in systems.items():
        if not isinstance(key, str) or not key:
            raise ValueError("SWA-v12 system keys must be nonempty strings")
        if not isinstance(system, dict) or set(system) != SYSTEM_KEYS:
            raise ValueError("invalid SWA-v12 system configuration")
        if not isinstance(system["probe_checkpoints"], list) or len(
            system["probe_checkpoints"]
        ) < 2:
            raise ValueError("checkpoint-weight probe requires 2+ checkpoints")
        if not all(
            isinstance(value, str) for value in system["probe_checkpoints"]
        ):
            raise ValueError("probe checkpoints must be path strings")
        if int(system["tail_epochs"]) <= 0 or int(
            system["maximum_tail_epochs"]
        ) < int(system["tail_epochs"]):
            raise ValueError("invalid SWA tail epoch range")
        if int(system["batch_size"]) <= 0 or int(system["num_workers"]) < 0:
            raise ValueError("invalid SWA loader settings")
        baseline = system["baseline"]
        if not isinstance(baseline, dict) or set(baseline) != BASELINE_KEYS:
            raise ValueError("invalid SWA baseline metrics")
        if int(baseline["minimum_correct_gain"]) <= 0:
            raise ValueError("minimum correct-letter gain must be positive")
    return config


def _validate_compatible_checkpoints(
    checkpoints: Sequence[Mapping[str, Any]],
) -> None:
    first = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        if checkpoint["model_config"] != first["model_config"]:
            raise ValueError("checkpoint model configurations differ")
        if checkpoint["vocab"] != first["vocab"]:
            raise ValueError("checkpoint vocabularies differ")
        if set(checkpoint["model_state_dict"]) != set(
            first["model_state_dict"]
        ):
            raise ValueError("checkpoint model-state keys differ")


def average_model_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Return a deterministic equal arithmetic mean of compatible weights."""
    if len(state_dicts) < 2:
        raise ValueError("weight averaging requires at least two states")
    keys = tuple(state_dicts[0])
    if any(tuple(state) != keys for state in state_dicts[1:]):
        raise ValueError("state dictionaries must have identical ordered keys")
    averaged: Dict[str, torch.Tensor] = {}
    for key in keys:
        values = [state[key].detach().to("cpu") for state in state_dicts]
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("tensor shape mismatch for {}".format(key))
        if reference.is_floating_point() or reference.is_complex():
            accumulator_dtype = (
                torch.complex128 if reference.is_complex() else torch.float64
            )
            accumulator = torch.zeros_like(
                reference, dtype=accumulator_dtype
            )
            for value in values:
                accumulator.add_(value.to(accumulator_dtype))
            averaged[key] = (accumulator / len(values)).to(reference.dtype)
        else:
            if any(not torch.equal(reference, value) for value in values[1:]):
                raise ValueError("non-floating state differs for {}".format(key))
            averaged[key] = reference.clone()
    return averaged


def update_prefix_average(
    previous_average: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
    member_count: int,
) -> Dict[str, torch.Tensor]:
    """Add member ``member_count`` to an existing ``member_count-1`` mean."""
    if member_count < 2:
        raise ValueError("prefix update requires member_count >= 2")
    if tuple(previous_average) != tuple(current_state):
        raise ValueError("prefix-average states must have identical keys")
    updated: Dict[str, torch.Tensor] = {}
    for key, current_value in current_state.items():
        previous_value = previous_average[key]
        if previous_value.shape != current_value.shape:
            raise ValueError("prefix-average tensor shape mismatch")
        if current_value.is_floating_point() or current_value.is_complex():
            accumulator_dtype = (
                torch.complex128
                if current_value.is_complex()
                else torch.float64
            )
            previous64 = previous_value.to(accumulator_dtype)
            current64 = current_value.to(accumulator_dtype)
            updated[key] = (
                previous64
                + (current64 - previous64) / float(member_count)
            ).to(current_value.dtype)
        else:
            if not torch.equal(previous_value, current_value):
                raise ValueError(
                    "non-floating prefix-average state changed for {}".format(
                        key
                    )
                )
            updated[key] = current_value.clone()
    return updated


def _averaged_checkpoint(
    source: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    result = {
        "schema_version": 1,
        "model_config": copy.deepcopy(source["model_config"]),
        "model_state_dict": {
            key: value.detach().to("cpu").clone()
            for key, value in state_dict.items()
        },
        "vocab": copy.deepcopy(source["vocab"]),
        "experiment_config": copy.deepcopy(
            source.get("experiment_config", {})
        ),
        "epoch": int(metadata.get("epoch", source.get("epoch", 0))),
        "dev_metrics": {},
        "averaging_metadata": copy.deepcopy(dict(metadata)),
    }
    return result


def _evaluate_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    dev_records = load_jsonl(Path("Data/dev_data/dev_Algerian-DIAC.jsonl"))
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))
    neural_predictions = predict_records(
        model, dev_records, vocab, device, batch_size, num_workers
    )
    v2_predictions, _ = predict_with_gated_fallback(
        model,
        dev_records,
        vocab,
        prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    return (
        compute_paper_metrics(dev_records, neural_predictions),
        compute_paper_metrics(dev_records, v2_predictions),
    )


def _selection(
    system: Mapping[str, Any],
    neural: Mapping[str, Any],
    v2: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline = system["baseline"]
    correct_gain = int(v2["correct_letters"]) - int(baseline["v2_correct"])
    regressions = {
        "word_accuracy": float(v2["word_accuracy"])
        < float(baseline["v2_word_accuracy"]),
        "sentence_accuracy": float(v2["sentence_accuracy"])
        < float(baseline["v2_sentence_accuracy"]),
        "shadda_accuracy": float(v2["shadda"]["accuracy"])
        < float(baseline["v2_shadda_accuracy"]),
        "tanween_accuracy": float(v2["tanween"]["accuracy"])
        < float(baseline["v2_tanween_accuracy"]),
    }
    return {
        "neural_correct": int(neural["correct_letters"]),
        "v2_correct": int(v2["correct_letters"]),
        "v2_correct_gain": correct_gain,
        "minimum_correct_gain": int(baseline["minimum_correct_gain"]),
        "regressions": regressions,
        "accepted": correct_gain >= int(baseline["minimum_correct_gain"])
        and not any(regressions.values()),
    }


def run_checkpoint_weight_probe(
    config: Mapping[str, Any],
    system_key: str,
    device_name: str,
) -> Dict[str, Any]:
    system = config["systems"][system_key]
    paths = [Path(value) for value in system["probe_checkpoints"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("probe checkpoints missing: {}".format(missing))
    checkpoints = [torch.load(path, map_location="cpu") for path in paths]
    _validate_compatible_checkpoints(checkpoints)
    averaged_state = average_model_state_dicts(
        [checkpoint["model_state_dict"] for checkpoint in checkpoints]
    )
    output_dir = (
        Path(str(config["output_root"]))
        / "00_checkpoint_weight_probe"
        / system_key
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    averaged_path = output_dir / "checkpoint_weight_average.pt"
    metadata = {
        "kind": "checkpoint_weight_average_not_swa",
        "sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "epoch": max(int(checkpoint.get("epoch", 0)) for checkpoint in checkpoints),
    }
    save_checkpoint(
        averaged_path,
        _averaged_checkpoint(checkpoints[0], averaged_state, metadata),
    )
    device = select_device(device_name)
    neural, v2 = _evaluate_checkpoint(
        averaged_path,
        device,
        int(system["batch_size"]),
        int(system["num_workers"]),
    )
    write_json(output_dir / "neural_metrics.json", neural)
    write_json(output_dir / "v2_metrics.json", v2)
    result = {
        "schema_version": 1,
        "system_name": system["system_name"].replace(
            "SWA-tail", "CheckpointWeightProbe"
        ),
        "method": "equal arithmetic mean of existing checkpoint weights; not SWA",
        "checkpoint": str(averaged_path),
        "checkpoint_sha256": sha256_file(averaged_path),
        "selection": _selection(system, neural, v2),
    }
    write_json(output_dir / "SELECTION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    cpu_rng_state = state["torch_cpu"]
    if not isinstance(cpu_rng_state, torch.Tensor):
        raise TypeError("saved CPU RNG state must be a tensor")
    torch.set_rng_state(
        cpu_rng_state.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )
    if (
        device.type == "mps"
        and "torch_mps" in state
        and hasattr(torch.mps, "set_rng_state")
    ):
        mps_rng_state = state["torch_mps"]
        if not isinstance(mps_rng_state, torch.Tensor):
            raise TypeError("saved MPS RNG state must be a tensor")
        torch.mps.set_rng_state(
            mps_rng_state.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
        )


def _has_batch_norm(model: torch.nn.Module) -> bool:
    return any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )


def _train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    device: torch.device,
    gradient_clip_norm: float,
    shadda_loss_weight: float,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        input_ids, targets, attention_mask = _move_batch(batch, device)
        outputs = model(input_ids, attention_mask)
        loss = model.compute_loss(outputs, targets, shadda_loss_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def run_swa_tail(
    config: Mapping[str, Any],
    system_key: str,
    device_name: str,
    target_tail_epochs: int,
) -> Dict[str, Any]:
    """Run/resume the controlled fixed-LR tail and prefix weight averages."""
    system = config["systems"][system_key]
    if target_tail_epochs < int(system["tail_epochs"]) or target_tail_epochs > int(
        system["maximum_tail_epochs"]
    ):
        raise ValueError("target tail epochs are outside the configured range")
    device = select_device(device_name)
    source_path = Path(str(system["source_checkpoint"]))
    source_hash = sha256_file(source_path)
    source = load_checkpoint(source_path, device)
    source_seed = int(source.get("experiment_config", {}).get("seed", 42))
    seed_everything(source_seed)
    model, vocab = build_model_from_checkpoint(source, device)
    if _has_batch_norm(model):
        raise ValueError("SWA-v12 requires explicit BatchNorm refresh support")
    experiment = source.get("experiment_config", {})
    training = experiment.get("training", {})
    train_records = tuple(
        load_jsonl(Path("Data/train_data/train_Algerian-DIAC.jsonl"))
    )
    loader, sampler = _make_loader(
        train_records,
        vocab,
        batch_size=int(system["batch_size"]),
        shuffle=True,
        seed=source_seed,
        num_workers=int(system["num_workers"]),
        pin_memory=False,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3.0e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    optimizer.load_state_dict(source["optimizer_state_dict"])
    fixed_lr = float(optimizer.param_groups[0]["lr"])
    for group in optimizer.param_groups:
        group["lr"] = fixed_lr

    output_dir = Path(str(config["output_root"])) / "01_swa_tail" / system_key
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    resume_path = output_dir / "resume.pt"
    swa_state: MutableMapping[str, torch.Tensor] = {}
    completed = 0
    if resume_path.is_file():
        # RNG generator states must remain CPU ByteTensors. Loading the whole
        # resume payload onto MPS also moves ``torch_cpu`` to MPS, which makes
        # ``torch.set_rng_state`` reject an otherwise valid resume checkpoint.
        # ``load_state_dict`` moves model and optimizer tensors to their owning
        # parameter devices after this CPU load.
        resume = torch.load(resume_path, map_location="cpu")
        if resume["source_sha256"] != source_hash:
            raise ValueError("SWA resume source checkpoint changed")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = fixed_lr
        swa_state = {
            key: value.to("cpu") for key, value in resume["swa_state_dict"].items()
        }
        completed = int(resume["tail_epoch"])
        if completed > target_tail_epochs:
            raise ValueError(
                "resume already exceeds the requested target tail epochs"
            )
        _restore_rng_state(resume["rng_state"], device)
    else:
        metrics_path.write_text("", encoding="utf-8")

    started_at = time.time()
    previous_selection_path = output_dir / "SELECTION.json"
    if previous_selection_path.is_file():
        previous_selection = json.loads(
            previous_selection_path.read_text(encoding="utf-8")
        )
        best_selection = dict(previous_selection.get("best_selection", {}))
        best_correct = int(best_selection.get("v2_correct", -1))
    else:
        best_selection = {}
        best_correct = -1
    for tail_epoch in range(completed + 1, target_tail_epochs + 1):
        sampler.set_epoch(int(source.get("epoch", 0)) + tail_epoch)
        _train_epoch(
            model,
            loader,
            optimizer,
            device,
            float(training.get("gradient_clip_norm", 1.0)),
            float(training.get("shadda_loss_weight", 1.0)),
        )
        current = {
            key: value.detach().to("cpu") for key, value in model.state_dict().items()
        }
        if not swa_state:
            swa_state = {key: value.clone() for key, value in current.items()}
        else:
            swa_state = update_prefix_average(
                swa_state, current, tail_epoch
            )
        snapshot_path = snapshot_dir / "tail_epoch_{:02d}.pt".format(tail_epoch)
        save_checkpoint(
            snapshot_path,
            _averaged_checkpoint(
                source,
                current,
                {
                    "kind": "swa_tail_member",
                    "source_sha256": source_hash,
                    "tail_epoch": tail_epoch,
                    "fixed_learning_rate": fixed_lr,
                },
            ),
        )
        prefix_path = output_dir / "current_prefix_average.pt"
        save_checkpoint(
            prefix_path,
            _averaged_checkpoint(
                source,
                swa_state,
                {
                    "kind": "true_swa_prefix_weight_average",
                    "source_sha256": source_hash,
                    "tail_epoch": tail_epoch,
                    "member_count": tail_epoch,
                    "fixed_learning_rate": fixed_lr,
                },
            ),
        )
        neural, v2 = _evaluate_checkpoint(
            prefix_path,
            device,
            int(system["batch_size"]),
            int(system["num_workers"]),
        )
        selection = _selection(system, neural, v2)
        append_jsonl(
            metrics_path,
            {
                "tail_epoch": tail_epoch,
                "fixed_learning_rate": fixed_lr,
                "elapsed_seconds": time.time() - started_at,
                "neural": neural,
                "v2": v2,
                "selection": selection,
            },
        )
        if int(v2["correct_letters"]) > best_correct:
            best_correct = int(v2["correct_letters"])
            best_selection = dict(selection)
            save_checkpoint(output_dir / "best_swa.pt", torch.load(prefix_path))
        save_checkpoint(
            resume_path,
            {
                "schema_version": 1,
                "source_sha256": source_hash,
                "tail_epoch": tail_epoch,
                "fixed_learning_rate": fixed_lr,
                "model_state_dict": {
                    key: value.detach().to("cpu")
                    for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "swa_state_dict": dict(swa_state),
                "rng_state": _capture_rng_state(device),
            },
        )
    result = {
        "schema_version": 1,
        "system_name": system["system_name"],
        "method": "fixed-LR continuation with arithmetic prefix weight means",
        "source_checkpoint": str(source_path),
        "source_sha256": source_hash,
        "fixed_learning_rate": fixed_lr,
        "tail_epochs_completed": target_tail_epochs,
        "best_selection": best_selection,
        "best_checkpoint": str(output_dir / "best_swa.pt"),
    }
    write_json(output_dir / "SELECTION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/swa_v12/campaign.json"),
    )
    parser.add_argument("--system", required=True)
    parser.add_argument(
        "--mode", choices=("checkpoint-weight-probe", "swa-tail"), required=True
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tail-epochs", type=int)
    args = parser.parse_args()
    config = load_swa_config(args.config)
    if args.system not in config["systems"]:
        parser.error("unknown --system")
    if args.mode == "checkpoint-weight-probe":
        run_checkpoint_weight_probe(config, args.system, args.device)
        return
    target = args.target_tail_epochs
    if target is None:
        target = int(config["systems"][args.system]["tail_epochs"])
    run_swa_tail(config, args.system, args.device, target)


if __name__ == "__main__":
    main()
