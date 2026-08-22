"""Strict JSON experiment configuration with deterministic defaults."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data": {
        "train": "Data/train_data/train_Algerian-DIAC.jsonl",
        "dev": "Data/dev_data/dev_Algerian-DIAC.jsonl",
        "vocab": "Data/vocab.json",
    },
    "model": {
        "architecture": "conv_local_transformer",
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "ffn_dim": 1024,
        "dropout": 0.15,
        "max_length": 512,
        "attention_window": 64,
        "conv_kernels": [3, 5, 7],
        "factorized_head": True,
        "head_mode": None,
        "global_attention_every": 0,
        "guided_label_training": False,
        "guided_schedule": "none",
        "guided_mask_steps": 10,
        "word_num_layers": 2,
        "word_ffn_dim": 512,
        "max_word_length": 32,
        "word_position_features": False,
        "dual_local_num_layers": 6,
        "dual_global_num_layers": 4,
        "dual_refinement_num_layers": 2,
        "rope_base": 10000.0,
        "dual_local_shifted": False,
        "crf_boundary_rank": 2,
    },
    "training": {
        "epochs": 60,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "scheduler": "cosine_warmup",
        "warmup_fraction": 0.05,
        "one_cycle_pct_start": 0.3,
        "one_cycle_div_factor": 25.0,
        "one_cycle_final_div_factor": 10000.0,
        "gradient_clip_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "shadda_loss_weight": 1.0,
        "early_stopping_patience": 10,
        "selection_mode": "dev_best",
        "dev_evaluation_mode": "each_epoch",
        "rdrop_coefficient": 0.0,
        "rdrop_distribution": "emission",
        "num_workers": 2,
        "amp": True,
        "device": "auto",
    },
    "output_dir": "outputs/conv_local_seed42",
}


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError("unknown configuration key: {}".format(key))
        if isinstance(result[key], dict):
            if not isinstance(value, Mapping):
                raise ValueError("{} must be an object".format(key))
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        override = json.load(stream)
    if not isinstance(override, dict):
        raise ValueError("configuration root must be a JSON object")
    config = _deep_merge(DEFAULT_CONFIG, override)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    model = config["model"]
    training = config["training"]
    if model["architecture"] not in (
        "plain_transformer",
        "conv_local_transformer",
        "hierarchical_transformer",
        "dual_rope_transformer",
    ):
        raise ValueError("unsupported model architecture")
    if model["d_model"] <= 0 or model["d_model"] % model["num_heads"] != 0:
        raise ValueError("d_model must be positive and divisible by num_heads")
    if model["num_layers"] <= 0 or model["ffn_dim"] <= 0:
        raise ValueError("num_layers and ffn_dim must be positive")
    if not 0.0 <= model["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if model["max_length"] < 3:
        raise ValueError("max_length must include BOS, content, and EOS")
    if model["attention_window"] <= 0:
        raise ValueError("attention_window must be positive")
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in model["conv_kernels"]):
        raise ValueError("all convolution kernels must be positive and odd")
    if model["head_mode"] not in (
        None,
        "factorized",
        "direct",
        "gated_joint",
        "crf",
        "boundary_crf",
        "factorized_crf",
        "low_rank_boundary_crf",
        "context_low_rank_boundary_crf",
    ):
        raise ValueError(
            "head_mode must be null, factorized, direct, gated_joint, crf, "
            "boundary_crf, factorized_crf, low_rank_boundary_crf, or "
            "context_low_rank_boundary_crf"
        )
    if model["global_attention_every"] < 0:
        raise ValueError("global_attention_every cannot be negative")
    if model["guided_mask_steps"] <= 0:
        raise ValueError("guided_mask_steps must be positive")
    if model["guided_schedule"] not in (
        "none",
        "uniform",
        "linear_blank_curriculum",
    ):
        raise ValueError("unsupported guided_schedule")
    if bool(model["guided_label_training"]) != (
        model["guided_schedule"] != "none"
    ):
        raise ValueError(
            "guided_label_training must match guided_schedule != none"
        )
    if model["word_num_layers"] <= 0 or model["word_ffn_dim"] <= 0:
        raise ValueError("word_num_layers and word_ffn_dim must be positive")
    if model["max_word_length"] <= 0:
        raise ValueError("max_word_length must be positive")
    if not isinstance(model["word_position_features"], bool):
        raise ValueError("word_position_features must be boolean")
    if (
        model["dual_local_num_layers"] <= 0
        or model["dual_global_num_layers"] <= 0
        or model["dual_refinement_num_layers"] <= 0
    ):
        raise ValueError("all dual-stream layer counts must be positive")
    if model["rope_base"] <= 1.0:
        raise ValueError("rope_base must exceed 1")
    if isinstance(model["crf_boundary_rank"], bool) or not isinstance(
        model["crf_boundary_rank"], int
    ):
        raise ValueError("crf_boundary_rank must be an integer")
    if model["crf_boundary_rank"] <= 0:
        raise ValueError("crf_boundary_rank must be positive")
    if model["architecture"] == "dual_rope_transformer":
        total_dual_layers = (
            model["dual_local_num_layers"]
            + model["dual_global_num_layers"]
            + model["dual_refinement_num_layers"]
        )
        if model["num_layers"] != total_dual_layers:
            raise ValueError(
                "num_layers must equal the total dual-stream block count"
            )
        if (model["d_model"] // model["num_heads"]) % 2 != 0:
            raise ValueError(
                "dual_rope_transformer requires an even head dimension"
            )
        resolved_head = model["head_mode"]
        if resolved_head is None:
            resolved_head = (
                "factorized" if model["factorized_head"] else "direct"
            )
        if resolved_head not in (
            "direct",
            "crf",
            "boundary_crf",
            "factorized_crf",
            "low_rank_boundary_crf",
            "context_low_rank_boundary_crf",
        ):
            raise ValueError(
                "dual_rope_transformer requires a direct or CRF head"
            )
        if model["guided_label_training"]:
            raise ValueError(
                "dual_rope_transformer does not use guided label hints"
            )
    if training["epochs"] <= 0 or training["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if training["learning_rate"] <= 0.0:
        raise ValueError("learning_rate must be positive")
    if training["scheduler"] not in ("cosine_warmup", "one_cycle"):
        raise ValueError("unsupported learning-rate scheduler")
    if not 0.0 <= training["warmup_fraction"] < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 < training["one_cycle_pct_start"] < 1.0:
        raise ValueError("one_cycle_pct_start must be in (0, 1)")
    if (
        training["one_cycle_div_factor"] <= 0.0
        or training["one_cycle_final_div_factor"] <= 0.0
    ):
        raise ValueError("OneCycle division factors must be positive")
    if training["gradient_accumulation_steps"] <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if training["early_stopping_patience"] <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if training["selection_mode"] not in ("dev_best", "last_epoch"):
        raise ValueError("selection_mode must be dev_best or last_epoch")
    if training["dev_evaluation_mode"] not in ("each_epoch", "final_only"):
        raise ValueError(
            "dev_evaluation_mode must be each_epoch or final_only"
        )
    if (
        training["dev_evaluation_mode"] == "final_only"
        and training["selection_mode"] != "last_epoch"
    ):
        raise ValueError(
            "final_only dev evaluation requires last_epoch selection"
        )
    rdrop_coefficient = training["rdrop_coefficient"]
    if isinstance(rdrop_coefficient, bool) or not isinstance(
        rdrop_coefficient, (int, float)
    ):
        raise ValueError("rdrop_coefficient must be numeric")
    if float(rdrop_coefficient) < 0.0:
        raise ValueError("rdrop_coefficient cannot be negative")
    resolved_head = model["head_mode"]
    if resolved_head is None:
        resolved_head = (
            "factorized" if model["factorized_head"] else "direct"
        )
    if float(rdrop_coefficient) > 0.0 and resolved_head != "crf":
        raise ValueError(
            "positive rdrop_coefficient currently requires head_mode=crf"
        )
    if training["rdrop_distribution"] not in (
        "emission",
        "crf_marginal",
    ):
        raise ValueError(
            "rdrop_distribution must be emission or crf_marginal"
        )
