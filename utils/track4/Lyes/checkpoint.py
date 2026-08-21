"""Validated checkpoint loading and model reconstruction."""

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, ModelConfig


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    required = {
        "schema_version",
        "model_config",
        "model_state_dict",
        "vocab",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError("checkpoint is missing keys: {}".format(missing))
    if checkpoint["schema_version"] != 1:
        raise ValueError(
            "unsupported checkpoint schema {}".format(
                checkpoint["schema_version"]
            )
        )
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> Tuple[CharDiacritizer, Dict[str, int]]:
    vocab = {
        str(key): int(value) for key, value in checkpoint["vocab"].items()
    }
    raw_model_config = dict(checkpoint["model_config"])
    raw_model_config["conv_kernels"] = tuple(
        raw_model_config["conv_kernels"]
    )
    model_config = ModelConfig(**raw_model_config)
    if model_config.vocab_size != len(vocab):
        raise ValueError("checkpoint vocabulary/model size mismatch")
    if model_config.pad_id != vocab["<PAD>"]:
        raise ValueError("checkpoint padding ID mismatch")

    model = CharDiacritizer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, vocab
