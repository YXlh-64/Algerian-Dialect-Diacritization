from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    vocab_size: int = 43
    pad_id: int = 0
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_dim: int = 1024
    kernels: Tuple[int, ...] = (3, 5, 7)
    max_len: int = 512
    dropout: float = 0.30
    rel_pos_buckets: int = 32
    rel_pos_max_distance: int = 128
    num_classes: int = 16
