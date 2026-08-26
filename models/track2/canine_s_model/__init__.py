"""Standard 16-class CANINE-S model used by the Track 2 Manel experiment."""

from .canine_s_model import (
    build_label_mapping,
    diacritize,
    load_model,
    save_model_artifacts,
)

__all__ = [
    "build_label_mapping",
    "diacritize",
    "load_model",
    "save_model_artifacts",
]
