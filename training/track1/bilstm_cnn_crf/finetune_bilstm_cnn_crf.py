"""Train and submit the focused Track-1 P2 BiLSTM-CNN-CRF ensemble.

The entrypoint owns experiment orchestration only. Dataset/batching helpers,
the reusable training engine, model architecture, and ensemble evaluation live
in their corresponding Track-1 modules.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.track1.bilstm_cnn_crf.evaluate_bilstm_cnn_crf import (
    build_sentence_memory,
    build_word_log_priors,
    class_log_prior,
    decode_ensemble,
    score_record_predictions,
    tune_ensemble,
    vocalize,
    write_submission,
)
from models.track1.bilstm_cnn_crf.bilstm_cnn_crf_model import (
    BiLSTMDiacritizer,
    count_parameters,
)
from training.track1.bilstm_cnn_crf.data import DataSettings, seed_everything
from training.track1.bilstm_cnn_crf.engine import (
    TrainingContext,
    fit_full_data,
    fit_with_validation,
    initialize_model,
    predict_records,
    run_on_training_devices,
    transition_snapshot,
)
from utils.track1.data import NUM_LABELS, load_competition_data

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_REGISTRY = {
    "p2_ensemble": "five-seed character BiLSTM-CNN-CRF ensemble",
}
MODEL_SPECS = [
    {
        "name": f"p2_bilstm_cnn_crf_seed_{seed}",
        "use_cnn": True,
        "use_crf": True,
        "seed": seed,
    }
    for seed in (3407, 3408, 3409, 3410, 3411)
]


@dataclass
class RunConfig:
    profile: str = "competition"
    seed: int = 2026
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 30
    patience: int = 7
    embedding_dim: int = 128
    boundary_dim: int = 16
    model_dim: int = 256
    cnn_channels: int = 96
    cnn_kernels: tuple[int, ...] = (3, 5, 7)
    hidden_dim: int = 256
    lstm_layers: int = 3
    mlp_dim: int = 256
    dropout: float = 0.30
    learning_rate: float = 2e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    focal_gamma: float = 1.5
    effective_beta: float = 0.999
    max_class_weight: float = 8.0
    crf_aux_weight: float = 0.50
    sampler_max_weight: float = 5.0
    num_workers: int = 2
    max_gpus: int = 2
    parallel_gpu_training: bool = True
    amp: bool = True
    refit_on_full_data: bool = True
    exact_sentence_memory: bool = False
    output_dir: str = "/kaggle/working"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-model",
        default="p2_ensemble",
        choices=MODEL_REGISTRY,
        help="Track-1 architecture preset selected by run_pipeline.py",
    )
    parser.add_argument(
        "--profile",
        default="competition",
        choices=("competition", "smoke"),
        help="Use smoke for a short end-to-end wiring check",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default Kaggle/local artifact directory",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def build_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(profile=args.profile)
    if args.profile == "smoke":
        config.epochs = 2
        config.patience = 2
        config.embedding_dim = 48
        config.boundary_dim = 8
        config.model_dim = 64
        config.cnn_channels = 24
        config.hidden_dim = 64
        config.lstm_layers = 1
        config.mlp_dim = 64
        config.batch_size = 32
        config.eval_batch_size = 64
        config.refit_on_full_data = False
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    return config


def select_training_devices(config: RunConfig) -> list[torch.device]:
    device_count = torch.cuda.device_count()
    if device_count:
        devices = [
            torch.device(f"cuda:{index}")
            for index in range(min(config.max_gpus, device_count))
        ]
    else:
        devices = [torch.device("cpu")]
    return devices if config.parallel_gpu_training else devices[:1]


def resolve_output_dir(config: RunConfig) -> Path:
    output_dir = Path(config.output_dir)
    if config.output_dir.startswith("/kaggle") and not Path("/kaggle/working").exists():
        output_dir = Path.cwd() / "working/exports/track1/bilstm_cnn_crf/p2_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output_dir)
    return output_dir


def print_selection_summary(selection_results: list[dict[str, Any]]) -> None:
    print("\nSeed selection summary (ranked by dev macro-F1-16):")
    for result in sorted(
        selection_results,
        key=lambda item: item["metrics"]["macro_f1_16"],
        reverse=True,
    ):
        print(
            f"  {result['spec']['name']}: epoch={result['best_epoch']} | "
            f"macroF1-16={result['metrics']['macro_f1_16']:.5f} | "
            f"supported={result['metrics']['macro_f1_supported']:.5f} | "
            f"accuracy={result['metrics']['accuracy']:.5f}"
        )


def validate_submission(submission: pd.DataFrame, sample_submission_path: Path) -> None:
    sample_submission = pd.read_csv(sample_submission_path)
    if submission.columns.tolist() != ["Id", "Label"]:
        raise ValueError("submission columns must be ['Id', 'Label']")
    if len(submission) != len(sample_submission):
        raise ValueError("submission row count does not match the official sample")
    if submission["Id"].tolist() != sample_submission["Id"].tolist():
        raise ValueError("submission IDs are not in official sample order")
    if not submission["Label"].between(0, NUM_LABELS - 1).all():
        raise ValueError("submission contains an invalid label")
    if submission.isna().any().any():
        raise ValueError("submission contains missing values")


def run_experiment(config: RunConfig, active_model: str) -> None:
    devices = select_training_devices(config)
    output_dir = resolve_output_dir(config)
    data = load_competition_data()
    dual_gpu_active = len(devices) > 1
    settings = DataSettings(
        vocabulary=data.vocabulary,
        pad_id=data.vocabulary["<PAD>"],
        unk_id=data.vocabulary["<UNK>"],
        sampler_max_weight=config.sampler_max_weight,
        num_workers=config.num_workers,
        dual_gpu_active=dual_gpu_active,
    )
    context = TrainingContext(config=config, data=settings, output_dir=output_dir)

    print(
        f"model={active_model} | profile={config.profile} | "
        f"devices={[str(device) for device in devices]} | "
        f"parallel_seed_training={dual_gpu_active}"
    )
    for device in devices:
        if device.type == "cuda":
            print(f"{device}: {torch.cuda.get_device_name(device.index)}")
    print(
        f"DATA_ROOT: {data.root}\n"
        f"train={len(data.train_records):,} | dev={len(data.dev_records):,} | "
        f"test={len(data.test_records):,} | vocab={len(data.vocabulary)}"
    )

    seed_everything(config.seed)
    for spec in MODEL_SPECS:
        preview = BiLSTMDiacritizer(
            len(data.vocabulary),
            NUM_LABELS,
            spec["use_cnn"],
            spec["use_crf"],
            config,
            settings.pad_id,
        )
        print(f"{spec['name']}: {count_parameters(preview):,} parameters")
        del preview

    def selection_worker(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return fit_with_validation(
            spec, data.train_records, data.dev_records, device, context
        )

    selection_results = run_on_training_devices(
        MODEL_SPECS, selection_worker, "dev selection", devices
    )
    print_selection_summary(selection_results)

    dev_model_outputs = [result["outputs"] for result in selection_results]
    selection_transitions = [result["transition"] for result in selection_results]
    ensemble_config, ensemble_dev_predictions, _search = tune_ensemble(
        dev_model_outputs,
        selection_transitions,
        data.dev_records,
        data.train_records,
    )
    ensemble_dev_metrics = score_record_predictions(
        data.dev_records, ensemble_dev_predictions
    )
    print("Selected ensemble:")
    print(
        {
            **ensemble_config,
            "weights": {
                spec["name"]: float(weight)
                for spec, weight in zip(MODEL_SPECS, ensemble_config["weights"])
            },
        }
    )
    print(
        f"ensemble macroF1-16={ensemble_dev_metrics['macro_f1_16']:.5f} | "
        f"supported={ensemble_dev_metrics['macro_f1_supported']:.5f} | "
        f"accuracy={ensemble_dev_metrics['accuracy']:.5f}\n"
        f"CER={ensemble_dev_metrics['CER']:.5f} | "
        f"WER={ensemble_dev_metrics['WER']:.5f} | "
        f"DER={ensemble_dev_metrics['DER']:.5f} | "
        f"DER*={ensemble_dev_metrics['DER_star']:.5f} | "
        f"WER*={ensemble_dev_metrics['WER_star']:.5f}"
    )

    full_records = data.train_records + data.dev_records

    def final_inference_worker(
        result: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        spec = result["spec"]
        if config.refit_on_full_data:
            model, history = fit_full_data(
                spec, full_records, result["best_epoch"], device, context
            )
        else:
            model = initialize_model(spec, device, context)
            model.load_state_dict(result["best_state"])
            model.to(device)
            history = []
        outputs = predict_records(model, data.test_records, device, context)
        transition = transition_snapshot(model)
        model.cpu()
        del model
        gc.collect()
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return {
            "spec": spec,
            "history": history,
            "outputs": outputs,
            "transition": transition,
        }

    final_jobs = run_on_training_devices(
        selection_results,
        final_inference_worker,
        "full-data refit and test inference",
        devices,
    )
    final_predictions = decode_ensemble(
        [job["outputs"] for job in final_jobs],
        data.test_records,
        [job["transition"] for job in final_jobs],
        ensemble_config["weights"],
        build_word_log_priors(full_records),
        ensemble_config["lexical_strength"],
        class_log_prior(full_records),
        ensemble_config["frequency_strength"],
        ensemble_config["transition_strength"],
        build_sentence_memory(full_records),
        exact_sentence_memory=config.exact_sentence_memory,
    )
    if len(final_predictions) != len(data.test_records):
        raise ValueError("test prediction count mismatch")
    for record, prediction in zip(data.test_records, final_predictions):
        if len(prediction) != len(record["chars"]):
            raise ValueError(f"prediction length mismatch for {record['sent_id']}")
        if not np.all((0 <= prediction) & (prediction < NUM_LABELS)):
            raise ValueError(f"invalid prediction for {record['sent_id']}")
        if not all(
            predicted == 0
            for char, predicted in zip(record["chars"], prediction)
            if char == " "
        ):
            raise ValueError(f"nonzero space prediction for {record['sent_id']}")

    submission_path = output_dir / "submission.csv"
    submission = write_submission(data.test_records, final_predictions, submission_path)
    validate_submission(submission, data.sample_submission_path)
    print(f"Wrote {submission_path} with {len(submission):,} rows")

    vocalized_path = output_dir / "vocalized_predictions.txt"
    with vocalized_path.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(data.test_records, final_predictions):
            handle.write(vocalize(record["chars"], prediction) + "\n")

    run_summary = {
        "config": asdict(config),
        "models": [
            {
                "spec": result["spec"],
                "best_epoch": result["best_epoch"],
                "dev_macro_f1_16": result["metrics"]["macro_f1_16"],
                "dev_accuracy": result["metrics"]["accuracy"],
            }
            for result in selection_results
        ],
        "ensemble": {
            **ensemble_config,
            "weights": ensemble_config["weights"].tolist(),
            "dev_macro_f1_16": ensemble_dev_metrics["macro_f1_16"],
            "dev_macro_f1_supported": ensemble_dev_metrics["macro_f1_supported"],
            "dev_accuracy": ensemble_dev_metrics["accuracy"],
            "dev_CER": ensemble_dev_metrics["CER"],
            "dev_WER": ensemble_dev_metrics["WER"],
            "dev_DER": ensemble_dev_metrics["DER"],
            "dev_DER_star": ensemble_dev_metrics["DER_star"],
            "dev_WER_star": ensemble_dev_metrics["WER_star"],
        },
        "submission": str(submission_path),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {vocalized_path}")
    print(f"Wrote {summary_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_experiment(build_config(args), args.active_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
