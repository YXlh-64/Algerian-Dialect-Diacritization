"""Reusable training, validation, checkpointing, and device orchestration."""

from __future__ import annotations

import gc
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.track1.bilstm_cnn_crf.evaluate_bilstm_cnn_crf import (
    score_record_predictions,
)
from models.track1.bilstm_cnn_crf.bilstm_cnn_crf_model import BiLSTMDiacritizer
from training.track1.bilstm_cnn_crf.data import (
    DataSettings,
    effective_number_weights,
    make_loader,
    move_batch,
)

MODEL_INITIALIZATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class TrainingContext:
    config: Any
    data: DataSettings
    output_dir: Path


def initialize_model(
    spec: dict[str, Any], device: torch.device, context: TrainingContext
) -> BiLSTMDiacritizer:
    """Initialize deterministically without racing PyTorch's global CPU RNG."""
    with MODEL_INITIALIZATION_LOCK:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec["seed"])
            model = BiLSTMDiacritizer(
                len(context.data.vocabulary),
                context.data.num_labels,
                spec["use_cnn"],
                spec["use_crf"],
                context.config,
                context.data.pad_id,
            )
        model = model.to(device)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(spec["seed"])
    return model


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=bool(enabled and device.type == "cuda"))


def train_epoch(
    model: BiLSTMDiacritizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    class_weights: torch.Tensor,
    device: torch.device,
    context: TrainingContext,
) -> float:
    model.train()
    running_loss = 0.0
    examples = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, context.config.amp):
            loss, _ = model.loss(batch, class_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), context.config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(batch["indices"])
        running_loss += float(loss.detach()) * batch_size
        examples += batch_size
    return running_loss / max(examples, 1)


@torch.no_grad()
def predict_records(
    model: BiLSTMDiacritizer,
    records: list[dict[str, Any]],
    device: torch.device,
    context: TrainingContext,
) -> list[dict[str, Any]]:
    model.eval()
    loader = make_loader(
        records,
        context.data,
        batch_size=context.config.eval_batch_size,
        training=False,
        seed=context.config.seed,
        device=device,
    )
    outputs: list[dict[str, Any] | None] = [None] * len(records)
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with autocast_context(device, context.config.amp):
            emissions = model.emissions(batch)
        paths = model.decode(emissions.float(), batch["mask"])
        log_probabilities = torch.log_softmax(emissions.float(), dim=-1).cpu().numpy()
        for row, record_index in enumerate(raw_batch["indices"]):
            length = int(raw_batch["lengths"][row])
            outputs[record_index] = {
                "sent_id": records[record_index]["sent_id"],
                "chars": records[record_index]["chars"],
                "log_probs": log_probabilities[row, :length].copy(),
                "prediction": np.asarray(paths[row], dtype=np.int64),
            }
    return [output for output in outputs if output is not None]


def transition_snapshot(
    model: BiLSTMDiacritizer,
) -> dict[str, np.ndarray] | None:
    if model.crf is None:
        return None
    return {
        "start": model.crf.start_transitions.detach().float().cpu().numpy().copy(),
        "end": model.crf.end_transitions.detach().float().cpu().numpy().copy(),
        "transitions": model.crf.transitions.detach().float().cpu().numpy().copy(),
    }


def run_training(
    model: BiLSTMDiacritizer,
    records: list[dict[str, Any]],
    *,
    epochs: int,
    device: torch.device,
    seed: int,
    spec_name: str,
    context: TrainingContext,
    validation_records: list[dict[str, Any]] | None = None,
    use_early_stopping: bool = False,
) -> dict[str, Any]:
    """Run the shared optimizer/scheduler loop for selection or full refit.

    History recording, scheduler order, and early-stopping semantics match the
    original validated notebook: every evaluated epoch is recorded, including
    the stale epoch that triggers stopping.
    """
    class_weights = effective_number_weights(
        records,
        num_labels=context.data.num_labels,
        beta=context.config.effective_beta,
        cap=context.config.max_class_weight,
    ).to(device)
    train_loader = make_loader(
        records,
        context.data,
        batch_size=context.config.batch_size,
        training=True,
        seed=seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=context.config.learning_rate,
        weight_decay=context.config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=context.config.min_learning_rate,
    )
    scaler = make_grad_scaler(device, context.config.amp)
    best_score, best_epoch, best_state = -math.inf, 0, None
    stale_epochs, history = 0, []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, train_loader, optimizer, scaler, class_weights, device, context
        )
        learning_rate = optimizer.param_groups[0]["lr"]

        if validation_records is None:
            history.append({"epoch": epoch, "train_loss": loss})
            print(
                f"{device} | {spec_name} full | "
                f"epoch {epoch:02d}/{epochs:02d} | loss={loss:.4f}"
            )
            scheduler.step()
            continue

        dev_outputs = predict_records(model, validation_records, device, context)
        metrics = score_record_predictions(
            validation_records, [output["prediction"] for output in dev_outputs]
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "macro_f1_16": metrics["macro_f1_16"],
                "macro_f1_supported": metrics["macro_f1_supported"],
                "accuracy": metrics["accuracy"],
                "learning_rate": learning_rate,
            }
        )
        print(
            f"{device} | {spec_name} | epoch {epoch:02d} | loss={loss:.4f} | "
            f"macroF1-16={metrics['macro_f1_16']:.5f} | "
            f"supported={metrics['macro_f1_supported']:.5f} | "
            f"acc={metrics['accuracy']:.5f}"
        )
        if metrics["macro_f1_16"] > best_score + 1e-5:
            best_score = metrics["macro_f1_16"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if use_early_stopping and stale_epochs >= context.config.patience:
                print(f"Early stopping {spec_name} at epoch {epoch}.")
                break

    return {
        "history": history,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }


def fit_with_validation(
    spec: dict[str, Any],
    training_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    device: torch.device,
    context: TrainingContext,
) -> dict[str, Any]:
    model = initialize_model(spec, device, context)
    started = time.time()
    training = run_training(
        model,
        training_records,
        epochs=context.config.epochs,
        device=device,
        seed=spec["seed"],
        spec_name=spec["name"],
        context=context,
        validation_records=validation_records,
        use_early_stopping=True,
    )
    best_state = training["best_state"]
    best_epoch = training["best_epoch"]
    if best_state is None:
        raise RuntimeError(f"{spec['name']} did not produce a validation checkpoint")

    model.load_state_dict(best_state)
    model.to(device)
    best_outputs = predict_records(model, validation_records, device, context)
    best_metrics = score_record_predictions(
        validation_records, [output["prediction"] for output in best_outputs]
    )
    transition = transition_snapshot(model)
    checkpoint_path = context.output_dir / f"{spec['name']}_selection.pt"
    torch.save(
        {
            "state_dict": best_state,
            "spec": spec,
            "config": asdict(context.config),
            "best_epoch": best_epoch,
            "dev_metrics": {
                key: value
                for key, value in best_metrics.items()
                if key not in {"per_class_f1", "support"}
            },
        },
        checkpoint_path,
    )
    print(
        f"{spec['name']}: best_epoch={best_epoch}, "
        f"macroF1-16={best_metrics['macro_f1_16']:.5f}, "
        f"elapsed={(time.time() - started) / 60:.1f} min"
    )
    model.cpu()
    del model
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return {
        "spec": spec,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "history": training["history"],
        "outputs": best_outputs,
        "metrics": best_metrics,
        "transition": transition,
        "checkpoint_path": str(checkpoint_path),
    }


def fit_full_data(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    epochs: int,
    device: torch.device,
    context: TrainingContext,
) -> tuple[BiLSTMDiacritizer, list[dict[str, float]]]:
    model = initialize_model(spec, device, context)
    training = run_training(
        model,
        records,
        epochs=epochs,
        device=device,
        seed=spec["seed"],
        spec_name=spec["name"],
        context=context,
    )
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "spec": spec,
            "config": asdict(context.config),
            "epochs": epochs,
        },
        context.output_dir / f"{spec['name']}_full.pt",
    )
    return model, training["history"]


def run_on_training_devices(
    items: list[Any], worker, stage_name: str, devices: list[torch.device]
) -> list[Any]:
    """Keep one persistent thread per GPU and process its seeds sequentially."""
    indexed_items = list(enumerate(items))
    assignments = [
        indexed_items[device_index :: len(devices)]
        for device_index in range(len(devices))
    ]

    def run_device_queue(
        device: torch.device, queue: list[tuple[int, Any]]
    ) -> list[tuple[int, Any]]:
        completed = []
        for item_index, item in queue:
            print(
                f"\n{'=' * 90}\n"
                f"{stage_name}: item {item_index + 1}/{len(items)} on {device}"
            )
            completed.append((item_index, worker(item, device)))
        return completed

    ordered_results: list[Any | None] = [None] * len(items)
    if len(devices) == 1:
        device_results = run_device_queue(devices[0], assignments[0])
        for item_index, result in device_results:
            ordered_results[item_index] = result
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(run_device_queue, device, queue)
                for device, queue in zip(devices, assignments)
                if queue
            ]
            for future in as_completed(futures):
                for item_index, result in future.result():
                    ordered_results[item_index] = result

    if any(result is None for result in ordered_results):
        raise RuntimeError(f"{stage_name} did not return every scheduled result")
    return list(ordered_results)
