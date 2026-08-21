"""Train and validate a reproducible Track 4 model."""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, OneCycleLR
from torch.utils.data import DataLoader

from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    LengthBucketBatchSampler,
    SentenceRecord,
    load_jsonl,
    load_vocab,
    validate_vocabulary_coverage,
)
from evaluation.track4.Lyes.metrics import MetricAccumulator
from models.track4.Lyes.dual_stream_crf_head import (
    CharDiacritizer,
    ModelConfig,
    build_guided_label_hints,
)
from training.track4.Lyes.rdrop import symmetric_crf_marginal_kl, symmetric_emission_kl
from utils.track4.Lyes.utils import (
    append_jsonl,
    cosine_with_warmup_multiplier,
    save_checkpoint,
    seed_everything,
    select_device,
    write_json,
)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def _move_batch(
    batch: Mapping[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    return input_ids, targets, attention_mask


def _make_loader(
    records: Tuple[SentenceRecord, ...],
    vocab: Mapping[str, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, LengthBucketBatchSampler]:
    dataset = CharacterDataset(records)
    sampler = LengthBucketBatchSampler(
        lengths=[len(record.chars) + 2 for record in records],
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return loader, sampler


@torch.no_grad()
def evaluate_model(
    model: CharDiacritizer,
    loader: DataLoader,
    device: torch.device,
    shadda_loss_weight: float,
) -> Dict[str, Any]:
    model.eval()
    metrics = MetricAccumulator()
    for batch in loader:
        input_ids, targets, attention_mask = _move_batch(batch, device)
        outputs = model(input_ids, attention_mask)
        loss = model.compute_loss(outputs, targets, shadda_loss_weight)
        predictions = model.decode_outputs(outputs)
        metrics.update(predictions, targets, float(loss.item()))
    return metrics.compute()


def _checkpoint(
    model: CharDiacritizer,
    optimizer: AdamW,
    scheduler: LRScheduler,
    vocab: Mapping[str, int],
    config: Mapping[str, Any],
    epoch: int,
    dev_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
        "vocab": dict(vocab),
        "experiment_config": copy.deepcopy(dict(config)),
        "epoch": epoch,
        "dev_metrics": copy.deepcopy(dict(dev_metrics)),
    }


def _build_scheduler(
    optimizer: AdamW,
    training_config: Mapping[str, Any],
    total_updates: int,
) -> LRScheduler:
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    scheduler_name = str(training_config["scheduler"])
    if scheduler_name == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=float(training_config["learning_rate"]),
            total_steps=total_updates,
            pct_start=float(training_config["one_cycle_pct_start"]),
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=float(training_config["one_cycle_div_factor"]),
            final_div_factor=float(
                training_config["one_cycle_final_div_factor"]
            ),
        )
    if scheduler_name != "cosine_warmup":
        raise ValueError(
            "unsupported learning-rate scheduler: {}".format(scheduler_name)
        )
    warmup_steps = int(
        round(total_updates * float(training_config["warmup_fraction"]))
    )
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_multiplier(
            step, total_updates, warmup_steps
        ),
    )


def train(config: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    training_config = config["training"]
    selection_mode = str(training_config["selection_mode"])
    dev_evaluation_mode = str(training_config["dev_evaluation_mode"])
    rdrop_coefficient = float(training_config["rdrop_coefficient"])
    rdrop_distribution = str(training_config["rdrop_distribution"])
    device = select_device(str(training_config["device"]))
    use_amp = bool(training_config["amp"]) and device.type == "cuda"

    train_records = tuple(load_jsonl(Path(config["data"]["train"])))
    dev_records = tuple(load_jsonl(Path(config["data"]["dev"])))
    vocab = load_vocab(Path(config["data"]["vocab"]))
    validate_vocabulary_coverage(train_records, vocab)
    validate_vocabulary_coverage(dev_records, vocab)

    model_config = ModelConfig.from_mapping(
        config["model"],
        vocab_size=len(vocab),
        pad_id=vocab["<PAD>"],
        space_id=vocab[" "],
        bos_id=vocab["<BOS>"],
        eos_id=vocab["<EOS>"],
    )
    maximum_observed_length = max(
        len(record.chars) + 2 for record in train_records + dev_records
    )
    if maximum_observed_length > model_config.max_length:
        raise ValueError(
            "observed sequence length {} exceeds model max_length {}".format(
                maximum_observed_length, model_config.max_length
            )
        )

    pin_memory = device.type == "cuda"
    train_loader, train_sampler = _make_loader(
        train_records,
        vocab,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(training_config["num_workers"]),
        pin_memory=pin_memory,
    )
    dev_loader, _ = _make_loader(
        dev_records,
        vocab,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(training_config["num_workers"]),
        pin_memory=pin_memory,
    )

    model = CharDiacritizer(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = updates_per_epoch * int(training_config["epochs"])
    scheduler = _build_scheduler(
        optimizer, training_config, total_updates
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    write_json(output_dir / "resolved_config.json", dict(config))

    best_micro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    optimizer_updates = 0
    dev_evaluations = 0
    started_at = time.time()

    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        train_metrics = MetricAccumulator()
        training_nll_total = 0.0
        training_consistency_total = 0.0
        training_batch_count = 0
        accumulated_batches = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids, targets, attention_mask = _move_batch(batch, device)
            label_hints = None
            if model.config.guided_label_training:
                label_hints = build_guided_label_hints(
                    targets,
                    model.config.guided_mask_steps,
                    schedule=model.config.guided_schedule,
                    epoch=epoch,
                    total_epochs=int(training_config["epochs"]),
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(
                    input_ids,
                    attention_mask,
                    label_hints=label_hints,
                )
                if rdrop_coefficient > 0.0:
                    outputs_second = model(
                        input_ids,
                        attention_mask,
                        label_hints=label_hints,
                    )
                    nll_first = model.compute_loss(
                        outputs,
                        targets,
                        float(training_config["shadda_loss_weight"]),
                    )
                    nll_second = model.compute_loss(
                        outputs_second,
                        targets,
                        float(training_config["shadda_loss_weight"]),
                    )
                    mean_nll = 0.5 * (nll_first + nll_second)
                    if rdrop_distribution == "crf_marginal":
                        consistency = symmetric_crf_marginal_kl(
                            model, outputs, outputs_second
                        )
                    else:
                        consistency = symmetric_emission_kl(
                            outputs, outputs_second
                        )
                    loss = mean_nll + rdrop_coefficient * consistency
                else:
                    loss = model.compute_loss(
                        outputs,
                        targets,
                        float(training_config["shadda_loss_weight"]),
                    )
                    mean_nll = loss
                    consistency = loss.new_zeros(())
            scaler.scale(loss / accumulation_steps).backward()
            accumulated_batches += 1
            predictions = outputs["logits"].detach().argmax(dim=-1)
            train_metrics.update(predictions, targets, float(loss.item()))
            training_nll_total += float(mean_nll.detach().item())
            training_consistency_total += float(
                consistency.detach().item()
            )
            training_batch_count += 1

            is_last_batch = batch_index == len(train_loader)
            if accumulated_batches == accumulation_steps or is_last_batch:
                scaler.unscale_(optimizer)
                if is_last_batch and accumulated_batches < accumulation_steps:
                    correction = accumulation_steps / accumulated_batches
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training_config["gradient_clip_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                optimizer_updates += 1

        evaluate_dev = (
            dev_evaluation_mode == "each_epoch"
            or epoch == int(training_config["epochs"])
        )
        dev_metrics = None
        if evaluate_dev:
            dev_metrics = evaluate_model(
                model,
                dev_loader,
                device,
                float(training_config["shadda_loss_weight"]),
            )
            dev_evaluations += 1
        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_updates": optimizer_updates,
            "elapsed_seconds": time.time() - started_at,
            "train": train_metrics.compute(),
            "dev": dev_metrics,
            "rdrop": {
                "coefficient": rdrop_coefficient,
                "distribution": rdrop_distribution,
                "mean_nll": training_nll_total / training_batch_count,
                "mean_consistency": (
                    training_consistency_total / training_batch_count
                ),
            },
        }
        append_jsonl(metrics_path, epoch_metrics)
        if dev_metrics is None:
            print(
                "epoch={:03d} train_f1={:.6f} dev=deferred lr={:.8f}".format(
                    epoch,
                    epoch_metrics["train"]["micro_f1"],
                    optimizer.param_groups[0]["lr"],
                ),
                flush=True,
            )
        else:
            print(
                "epoch={:03d} train_f1={:.6f} dev_f1={:.6f} dev_loss={:.6f} lr={:.8f}".format(
                    epoch,
                    epoch_metrics["train"]["micro_f1"],
                    dev_metrics["micro_f1"],
                    dev_metrics["loss"],
                    optimizer.param_groups[0]["lr"],
                ),
                flush=True,
            )

        checkpoint = _checkpoint(
            model,
            optimizer,
            scheduler,
            vocab,
            config,
            epoch,
            {} if dev_metrics is None else dev_metrics,
        )
        save_checkpoint(output_dir / "last.pt", checkpoint)

        if selection_mode == "last_epoch" and dev_metrics is not None:
            best_micro_f1 = float(dev_metrics["micro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", checkpoint)
        elif (
            selection_mode == "dev_best"
            and dev_metrics is not None
            and float(dev_metrics["micro_f1"]) > best_micro_f1
        ):
            best_micro_f1 = float(dev_metrics["micro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", checkpoint)
        elif selection_mode == "dev_best" and dev_metrics is not None:
            epochs_without_improvement += 1

        if (
            selection_mode == "dev_best"
            and epochs_without_improvement
            >= int(training_config["early_stopping_patience"])
        ):
            print(
                "early stopping after {} epochs without improvement".format(
                    epochs_without_improvement
                ),
                flush=True,
            )
            break
        if device.type == "mps":
            torch.mps.empty_cache()

    summary = {
        "best_epoch": best_epoch,
        "best_dev_micro_f1": best_micro_f1,
        "selection_mode": selection_mode,
        "selection_metric_valid": selection_mode == "dev_best",
        "dev_evaluation_mode": dev_evaluation_mode,
        "dev_evaluations": dev_evaluations,
        "rdrop_coefficient": rdrop_coefficient,
        "rdrop_distribution": rdrop_distribution,
        "epochs_completed": epoch,
        "optimizer_updates": optimizer_updates,
        "elapsed_seconds": time.time() - started_at,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_sentences": len(train_records),
        "dev_sentences": len(dev_records),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--dev-data", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.train_data is not None:
        config["data"]["train"] = str(args.train_data)
    if args.dev_data is not None:
        config["data"]["dev"] = str(args.dev_data)
    if args.vocab is not None:
        config["data"]["vocab"] = str(args.vocab)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num-workers cannot be negative")
        config["training"]["num_workers"] = args.num_workers
    train(config)


if __name__ == "__main__":
    main()
