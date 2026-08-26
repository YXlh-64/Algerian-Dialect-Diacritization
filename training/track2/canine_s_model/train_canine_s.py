from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, CanineForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed

from utils.track2.canine_s_model.data_utils import CLASS_NAMES, LABEL_TO_MARKS, build_dataset, load_jsonl, resolve_dataset_dir, tokenize_and_align_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CANINE-S model for Algerian diacritization.")
    parser.add_argument("--data-dir", type=str, default=None, help="Root data directory containing train_data/ and dev_data/.")
    parser.add_argument("--config", type=str, default="utils/track2/canine_s_model/configs/canine_s_strategy_a.yaml", help="Path to YAML config.")
    parser.add_argument("--model-name", type=str, default="google/canine-s", help="HF CANINE checkpoint name.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--final-model-dir", type=str, default=None, help="Directory to save final model artifacts")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum tokenized sequence length.")
    parser.add_argument("--num-train-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        import yaml
        return yaml.safe_load(handle)


def _build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels = []
    pred_labels = []
    for pred_row, label_row in zip(preds, labels):
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)
    return {
        "micro_f1": f1_score(true_labels, pred_labels, average="micro"),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def run_training(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = resolve_dataset_dir(args.data_dir)
    config = load_config(args.config)

    train_path = data_dir / "train_data" / "train_Algerian-DIAC.jsonl"
    dev_path = data_dir / "dev_data" / "dev_Algerian-DIAC.jsonl"
    vocab_path = data_dir / "vocab.json"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(f"Expected train and dev files under {data_dir}")

    model_name = args.model_name or config.get("model_name", "google/canine-s")
    max_seq_len = args.max_seq_len or config.get("max_seq_len", 512)
    output_dir = Path(args.output_dir or config.get("checkpoint_dir", "working/checkpoints/canine_s"))
    final_model_dir = Path(args.final_model_dir or config.get("final_model_dir", "artifacts/canine_s"))

    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(train_path)
    dev_records = load_jsonl(dev_path)
    train_ds = build_dataset(train_records)
    dev_ds = build_dataset(dev_records)

    tokenizer = _build_tokenizer(model_name)
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    train_tokenized = train_ds.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, max_seq_len=max_seq_len),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    dev_tokenized = dev_ds.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, max_seq_len=max_seq_len),
        batched=True,
        remove_columns=dev_ds.column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = CanineForTokenClassification.from_pretrained(
        model_name,
        num_labels=16,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate or config.get("learning_rate", 2e-5),
        num_train_epochs=args.num_train_epochs or config.get("num_train_epochs", 10),
        per_device_train_batch_size=args.batch_size or config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=args.eval_batch_size or config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "micro_f1"),
        greater_is_better=config.get("greater_is_better", True),
        fp16=config.get("fp16", False),
        seed=args.seed,
        logging_steps=config.get("logging_steps", 50),
        save_total_limit=config.get("save_total_limit", 2),
        report_to=config.get("report_to", "none"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    with (final_model_dir / "label_to_marks.json").open("w", encoding="utf-8") as handle:
        json.dump(LABEL_TO_MARKS, handle, ensure_ascii=False, indent=2)

    with (final_model_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "model_name": model_name,
            "num_labels": 16,
            "max_seq_len": max_seq_len,
            "learning_rate": float(training_args.learning_rate),
            "num_train_epochs": int(training_args.num_train_epochs),
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "per_device_eval_batch_size": int(training_args.per_device_eval_batch_size),
            "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
            "weight_decay": float(training_args.weight_decay),
            "warmup_ratio": float(training_args.warmup_ratio),
            "seed": int(args.seed),
        }, handle, indent=2)

    metrics_out = {
        "dev_micro_f1": float(eval_metrics["eval_micro_f1"]),
        "dev_accuracy": float(eval_metrics["eval_accuracy"]),
    }
    with (final_model_dir / "dev_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_out, handle, indent=2)

    history = pd.DataFrame(trainer.state.log_history)
    train_logs = history.dropna(subset=["loss"]).copy()
    eval_logs = history.dropna(subset=["eval_micro_f1"]).copy()
    if not train_logs.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train_logs["epoch"], train_logs["loss"], marker="o")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        if not eval_logs.empty:
            axes[1].plot(eval_logs["epoch"], eval_logs["eval_micro_f1"], marker="o", color="green")
            axes[1].set_title("Dev Micro F1")
            axes[1].set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(final_model_dir / "training_history.png", dpi=150)

    return metrics_out


def main() -> None:
    args = parse_args()
    metrics = run_training(args)
    print("Training complete")
    print(f"Dev micro F1: {metrics.get('dev_micro_f1', 'N/A')}")
    print(f"Dev accuracy: {metrics.get('dev_accuracy', 'N/A')}")


if __name__ == "__main__":
    main()
