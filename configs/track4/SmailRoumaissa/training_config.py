from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    max_epochs: int = 80
    patience: int = 15
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    char_dropout_prob: float = 0.10
    seed: int = 42


@dataclass
class LexicalFusionConfig:
    entropy_threshold: float = 0.75
    gate_temperature: float = 0.5
    max_strength: float = 3.0


@dataclass
class CalibrationConfig:
    base_temperature: float = 1.0
    shadda_temperature: float = 1.0
    max_iter: int = 200
