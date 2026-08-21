"""Ines's dual-stream character encoder and CRF-head model implementation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from utils.track4.Lyes.labels import IGNORE_INDEX, NUM_BASE_LABELS, NUM_LABELS


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    space_id: int = 4
    bos_id: int = 2
    eos_id: int = 3
    architecture: str = "conv_local_transformer"
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.15
    max_length: int = 512
    attention_window: int = 64
    conv_kernels: Tuple[int, ...] = (3, 5, 7)
    factorized_head: bool = True
    head_mode: Optional[str] = None
    global_attention_every: int = 0
    guided_label_training: bool = False
    guided_schedule: str = "none"
    guided_mask_steps: int = 10
    word_num_layers: int = 2
    word_ffn_dim: int = 512
    max_word_length: int = 32
    word_position_features: bool = False
    dual_local_num_layers: int = 6
    dual_global_num_layers: int = 4
    dual_refinement_num_layers: int = 2
    rope_base: float = 10000.0
    dual_local_shifted: bool = False
    crf_boundary_rank: int = 2

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        vocab_size: int,
        pad_id: int,
        space_id: int = 4,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            pad_id=pad_id,
            space_id=space_id,
            bos_id=bos_id,
            eos_id=eos_id,
            architecture=str(values["architecture"]),
            d_model=int(values["d_model"]),
            num_layers=int(values["num_layers"]),
            num_heads=int(values["num_heads"]),
            ffn_dim=int(values["ffn_dim"]),
            dropout=float(values["dropout"]),
            max_length=int(values["max_length"]),
            attention_window=int(values["attention_window"]),
            conv_kernels=tuple(int(kernel) for kernel in values["conv_kernels"]),
            factorized_head=bool(values["factorized_head"]),
            head_mode=(
                None
                if values["head_mode"] is None
                else str(values["head_mode"])
            ),
            global_attention_every=int(values["global_attention_every"]),
            guided_label_training=bool(values["guided_label_training"]),
            guided_schedule=str(values["guided_schedule"]),
            guided_mask_steps=int(values["guided_mask_steps"]),
            word_num_layers=int(values["word_num_layers"]),
            word_ffn_dim=int(values["word_ffn_dim"]),
            max_word_length=int(values["max_word_length"]),
            word_position_features=bool(values["word_position_features"]),
            dual_local_num_layers=int(values["dual_local_num_layers"]),
            dual_global_num_layers=int(values["dual_global_num_layers"]),
            dual_refinement_num_layers=int(
                values["dual_refinement_num_layers"]
            ),
            rope_base=float(values["rope_base"]),
            dual_local_shifted=bool(values["dual_local_shifted"]),
            crf_boundary_rank=int(values["crf_boundary_rank"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["conv_kernels"] = list(self.conv_kernels)
        return result

    @property
    def resolved_head_mode(self) -> str:
        if self.head_mode is not None:
            return self.head_mode
        return "factorized" if self.factorized_head else "direct"


class MultiKernelConvFrontend(nn.Module):
    """Parallel depthwise convolutions that learn short dialectal patterns."""

    def __init__(
        self, d_model: int, kernels: Sequence[int], dropout: float
    ) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("at least one convolution kernel is required")
        self.input_norm = nn.LayerNorm(d_model)
        self.branches = nn.ModuleList(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=d_model,
                bias=False,
            )
            for kernel in kernels
        )
        self.projection = nn.Conv1d(
            d_model * len(kernels), d_model, kernel_size=1
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        residual = inputs
        normalized = self.input_norm(inputs).transpose(1, 2)
        convolved = torch.cat(
            [branch(normalized) for branch in self.branches], dim=1
        )
        projected = self.projection(F.gelu(convolved)).transpose(1, 2)
        output = self.output_norm(residual + self.dropout(projected))
        return output.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


class FullSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output, _ = self.attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        return output.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


class WindowSelfAttention(nn.Module):
    """Exact non-overlapping 1-D window attention with optional half shift."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        window_size: int,
        shifted: bool,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.shift = window_size // 2 if shifted else 0
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, length, d_model = inputs.shape
        left_padding = self.shift
        padded_length_without_right = length + left_padding
        right_padding = (
            self.window_size - padded_length_without_right % self.window_size
        ) % self.window_size

        padded_inputs = F.pad(
            inputs, (0, 0, left_padding, right_padding), value=0.0
        )
        padded_mask = F.pad(
            attention_mask, (left_padding, right_padding), value=False
        )
        number_of_windows = padded_inputs.size(1) // self.window_size

        window_inputs = padded_inputs.reshape(
            batch_size * number_of_windows, self.window_size, d_model
        )
        window_mask = padded_mask.reshape(
            batch_size * number_of_windows, self.window_size
        )
        nonempty = window_mask.any(dim=1)
        nonempty_indices = nonempty.nonzero(as_tuple=False).squeeze(1)

        attended = torch.zeros_like(window_inputs)
        if nonempty_indices.numel() > 0:
            active_inputs = window_inputs.index_select(0, nonempty_indices)
            active_mask = window_mask.index_select(0, nonempty_indices)
            active_output, _ = self.attention(
                active_inputs,
                active_inputs,
                active_inputs,
                key_padding_mask=~active_mask,
                need_weights=False,
            )
            attended = attended.index_copy(
                0, nonempty_indices, active_output
            )

        attended = attended.reshape(
            batch_size, number_of_windows * self.window_size, d_model
        )
        attended = attended[:, left_padding : left_padding + length]
        return attended.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


class RotaryEmbedding(nn.Module):
    """Deterministic rotary position encoding for one attention head."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be positive and even")
        if base <= 1.0:
            raise ValueError("RoPE base must exceed 1")
        inverse_frequency = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.head_dim = head_dim
        self.register_buffer(
            "inverse_frequency", inverse_frequency, persistent=False
        )

    def cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 2:
            raise ValueError("RoPE positions must have shape [batch, length]")
        angles = positions.to(self.inverse_frequency.dtype).unsqueeze(-1)
        angles = angles * self.inverse_frequency.view(1, 1, -1)
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    @staticmethod
    def rotate_pairs(inputs: torch.Tensor) -> torch.Tensor:
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(
        self, inputs: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if inputs.size(-1) != self.head_dim:
            raise ValueError("RoPE input head dimension does not match")
        cosine, sine = self.cos_sin(positions, inputs.dtype)
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        return inputs * cosine + self.rotate_pairs(inputs) * sine


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with RoPE applied to projected Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        head_dim = d_model // num_heads
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(head_dim, base=rope_base)

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = inputs.shape
        return inputs.reshape(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        query_states = self._split_heads(self.query_projection(queries))
        key_states = self._split_heads(self.key_projection(keys))
        value_states = self._split_heads(self.value_projection(values))
        query_states = self.rotary(query_states, query_positions)
        key_states = self.rotary(key_states, key_positions)

        allowed_keys = key_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=allowed_keys,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(
            queries.size(0), queries.size(1), self.d_model
        )
        output = self.output_projection(attended)
        return output.masked_fill(~query_mask.unsqueeze(-1), 0.0)


class RoPEFullSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.attention = RoPEMultiheadAttention(
            d_model, num_heads, dropout, rope_base
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention(
            inputs,
            inputs,
            inputs,
            attention_mask,
            attention_mask,
            positions,
            positions,
        )


class RoPEWindowSelfAttention(nn.Module):
    """Exact window attention with absolute positions retained inside RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        window_size: int,
        rope_base: float,
        shifted: bool = False,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.shift = window_size // 2 if shifted else 0
        self.attention = RoPEMultiheadAttention(
            d_model, num_heads, dropout, rope_base
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, d_model = inputs.shape
        left_padding = self.shift
        right_padding = (
            self.window_size - (length + left_padding) % self.window_size
        ) % self.window_size
        padded_inputs = F.pad(
            inputs, (0, 0, left_padding, right_padding), value=0.0
        )
        padded_mask = F.pad(
            attention_mask, (left_padding, right_padding), value=False
        )
        padded_positions = F.pad(
            positions, (left_padding, right_padding), value=0
        )
        number_of_windows = padded_inputs.size(1) // self.window_size
        window_inputs = padded_inputs.reshape(
            batch_size * number_of_windows, self.window_size, d_model
        )
        window_mask = padded_mask.reshape(
            batch_size * number_of_windows, self.window_size
        )
        window_positions = padded_positions.reshape(
            batch_size * number_of_windows, self.window_size
        )
        nonempty_indices = (
            window_mask.any(dim=1).nonzero(as_tuple=False).squeeze(1)
        )

        attended = torch.zeros_like(window_inputs)
        if nonempty_indices.numel() > 0:
            active_inputs = window_inputs.index_select(0, nonempty_indices)
            active_mask = window_mask.index_select(0, nonempty_indices)
            active_positions = window_positions.index_select(
                0, nonempty_indices
            )
            active_output = self.attention(
                active_inputs,
                active_inputs,
                active_inputs,
                active_mask,
                active_mask,
                active_positions,
                active_positions,
            )
            attended = attended.index_copy(
                0, nonempty_indices, active_output
            )

        attended = attended.reshape(
            batch_size, number_of_windows * self.window_size, d_model
        )
        attended = attended[:, left_padding : left_padding + length]
        return attended.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


class RoPETransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block using full or windowed RoPE attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_base: float,
        attention_window: Optional[int],
        shifted: bool = False,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        if attention_window is None:
            self.attention: nn.Module = RoPEFullSelfAttention(
                d_model, num_heads, dropout, rope_base
            )
        else:
            self.attention = RoPEWindowSelfAttention(
                d_model,
                num_heads,
                dropout,
                attention_window,
                rope_base,
                shifted=shifted,
            )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attention(
            self.attention_norm(inputs), attention_mask, positions
        )
        output = inputs + self.attention_dropout(attended)
        output = output.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        output = output + self.feed_forward(self.ffn_norm(output))
        return output.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


class DualRoPEEncoder(nn.Module):
    """Parallel local/global RoPE streams with learned cross-stream fusion."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.local_blocks = nn.ModuleList(
            RoPETransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                rope_base=config.rope_base,
                attention_window=config.attention_window,
                shifted=config.dual_local_shifted and bool(index % 2),
            )
            for index in range(config.dual_local_num_layers)
        )
        self.global_blocks = nn.ModuleList(
            RoPETransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                rope_base=config.rope_base,
                attention_window=None,
            )
            for _ in range(config.dual_global_num_layers)
        )
        self.cross_query_norm = nn.LayerNorm(config.d_model)
        self.cross_key_value_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = RoPEMultiheadAttention(
            config.d_model,
            config.num_heads,
            config.dropout,
            config.rope_base,
        )
        self.cross_dropout = nn.Dropout(config.dropout)
        self.cross_output_norm = nn.LayerNorm(config.d_model)
        self.fusion_gate = nn.Linear(config.d_model * 2, config.d_model)
        self.fusion_norm = nn.LayerNorm(config.d_model)
        self.refinement_blocks = nn.ModuleList(
            RoPETransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                rope_base=config.rope_base,
                attention_window=None,
            )
            for _ in range(config.dual_refinement_num_layers)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_states = inputs
        for block in self.local_blocks:
            local_states = block(local_states, attention_mask, positions)

        global_states = inputs
        for block in self.global_blocks:
            global_states = block(global_states, attention_mask, positions)

        normalized_global_states = self.cross_key_value_norm(global_states)
        cross_attended = self.cross_attention(
            self.cross_query_norm(local_states),
            normalized_global_states,
            normalized_global_states,
            attention_mask,
            attention_mask,
            positions,
            positions,
        )
        cross_states = self.cross_output_norm(
            local_states + self.cross_dropout(cross_attended)
        )
        combined = torch.cat((local_states, cross_states), dim=-1)
        fusion_gate = torch.sigmoid(self.fusion_gate(combined))
        fused = fusion_gate * local_states + (
            1.0 - fusion_gate
        ) * cross_states
        fused = self.fusion_norm(fused)
        fused = fused.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        for block in self.refinement_blocks:
            fused = block(fused, attention_mask, positions)
        return fused, fusion_gate.masked_fill(
            ~attention_mask.unsqueeze(-1), 0.0
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_window: Optional[int],
        shifted: bool,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        if attention_window is None:
            self.attention = FullSelfAttention(d_model, num_heads, dropout)
        else:
            self.attention = WindowSelfAttention(
                d_model,
                num_heads,
                dropout,
                attention_window,
                shifted=shifted,
            )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attended = self.attention(self.attention_norm(inputs), attention_mask)
        output = inputs + self.attention_dropout(attended)
        output = output.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        output = output + self.feed_forward(self.ffn_norm(output))
        return output.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


class WordContextEncoder(nn.Module):
    """Build global word context from character states without a word lexicon."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool_score = nn.Linear(d_model, 1)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_window=None,
                shifted=False,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.fusion_projection = nn.Linear(d_model * 2, d_model)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        character_states: torch.Tensor,
        content_mask: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, d_model = character_states.shape
        safe_word_ids = word_ids.clamp_min(0)
        scatter_indices = safe_word_ids.unsqueeze(-1).expand(
            batch_size, length, d_model
        )

        pool_weights = torch.sigmoid(self.pool_score(character_states))
        pool_weights = pool_weights * content_mask.unsqueeze(-1)
        weighted_states = character_states * pool_weights

        word_states = character_states.new_zeros(
            batch_size, length, d_model
        )
        word_weights = character_states.new_zeros(batch_size, length, 1)
        word_states.scatter_add_(1, scatter_indices, weighted_states)
        word_weights.scatter_add_(
            1, safe_word_ids.unsqueeze(-1), pool_weights
        )
        word_states = word_states / word_weights.clamp_min(1.0e-6)
        word_mask = word_weights.squeeze(-1).gt(0.0)

        for block in self.blocks:
            word_states = block(word_states, word_mask)
        word_states = self.final_norm(word_states)
        word_states = word_states.masked_fill(
            ~word_mask.unsqueeze(-1), 0.0
        )

        broadcast = word_states.gather(1, scatter_indices)
        broadcast = broadcast.masked_fill(
            ~content_mask.unsqueeze(-1), 0.0
        )
        combined = torch.cat((character_states, broadcast), dim=-1)
        candidate = torch.tanh(self.fusion_projection(combined))
        gate = torch.sigmoid(self.fusion_gate(combined))
        fused = character_states + gate * candidate
        return self.output_norm(fused)


def build_word_features(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    space_id: int,
    bos_id: int,
    eos_id: int,
    max_word_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return content mask, word IDs, forward positions, and reverse positions."""

    content_mask = (
        attention_mask
        & input_ids.ne(space_id)
        & input_ids.ne(bos_id)
        & input_ids.ne(eos_id)
    )
    previous_content = F.pad(content_mask[:, :-1], (1, 0), value=False)
    word_starts = content_mask & ~previous_content
    word_ids = word_starts.long().cumsum(dim=1) - 1
    word_ids = word_ids.masked_fill(~content_mask, -1)

    batch_size, length = input_ids.shape
    absolute_positions = torch.arange(
        length, device=input_ids.device
    ).unsqueeze(0).expand(batch_size, length)
    start_positions = torch.where(
        word_starts, absolute_positions, torch.zeros_like(absolute_positions)
    )
    latest_start = torch.cummax(start_positions, dim=1).values
    within_word = (absolute_positions - latest_start).masked_fill(
        ~content_mask, 0
    )
    within_word = within_word.clamp(max=max_word_length - 1)

    safe_word_ids = word_ids.clamp_min(0)
    word_lengths = torch.zeros(
        batch_size, length, dtype=torch.long, device=input_ids.device
    )
    word_lengths.scatter_add_(1, safe_word_ids, content_mask.long())
    character_word_lengths = word_lengths.gather(1, safe_word_ids)
    reverse_within_word = (
        character_word_lengths - within_word - 1
    ).masked_fill(~content_mask, 0)
    reverse_within_word = reverse_within_word.clamp(
        min=0, max=max_word_length - 1
    )
    return content_mask, word_ids, within_word, reverse_within_word


def build_guided_label_hints(
    targets: torch.Tensor,
    mask_steps: int,
    schedule: str = "uniform",
    epoch: int = 1,
    total_epochs: int = 1,
) -> torch.Tensor:
    """Sample 2SDiac-style discrete masking ratios; zero means blank hint."""

    if mask_steps <= 0:
        raise ValueError("mask_steps must be positive")
    valid = targets.ne(IGNORE_INDEX)
    safe_targets = targets.masked_fill(~valid, 0)
    batch_size = targets.size(0)
    if schedule not in ("uniform", "linear_blank_curriculum"):
        raise ValueError("unsupported guided hint schedule")
    if epoch <= 0 or total_epochs <= 0 or epoch > total_epochs:
        raise ValueError("invalid curriculum epoch")
    mask_levels = torch.randint(
        0,
        mask_steps + 1,
        (batch_size, 1),
        device=targets.device,
    )
    mask_probabilities = mask_levels.to(torch.float32) / mask_steps
    reveal = torch.rand(
        targets.shape, device=targets.device
    ).ge(mask_probabilities)
    reveal &= valid
    if schedule == "linear_blank_curriculum":
        blank_probability = (
            0.0
            if total_epochs == 1
            else float(epoch - 1) / float(total_epochs - 1)
        )
        force_blank = torch.rand(
            (batch_size, 1), device=targets.device
        ).lt(blank_probability)
        reveal &= ~force_blank
    return torch.where(reveal, safe_targets + 1, torch.zeros_like(targets))


class LinearChainCRF(nn.Module):
    """First-order linear-chain CRF over packed scored-letter positions."""

    def __init__(
        self,
        num_labels: int,
        boundary_conditioned: bool = False,
        boundary_rank: int = 0,
        context_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if num_labels <= 1:
            raise ValueError("CRF requires at least two labels")
        if boundary_rank < 0:
            raise ValueError("CRF boundary rank cannot be negative")
        if boundary_conditioned and boundary_rank > 0:
            raise ValueError(
                "full and low-rank boundary transitions are mutually exclusive"
            )
        if context_conditioned and boundary_conditioned:
            raise ValueError(
                "full and contextual boundary transitions are mutually exclusive"
            )
        if context_conditioned and boundary_rank == 0:
            raise ValueError(
                "context-conditioned transitions require a positive rank"
            )
        self.context_conditioned = context_conditioned
        self.num_labels = num_labels
        self.start_transitions = nn.Parameter(torch.zeros(num_labels))
        self.end_transitions = nn.Parameter(torch.zeros(num_labels))
        self.transitions = nn.Parameter(torch.zeros(num_labels, num_labels))
        if boundary_conditioned:
            self.boundary_transitions: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(num_labels, num_labels)
            )
        else:
            self.boundary_transitions = None
        if boundary_rank > 0:
            self.boundary_left: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(num_labels, boundary_rank)
            )
            self.boundary_right: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(boundary_rank, num_labels)
            )
            nn.init.normal_(self.boundary_left, mean=0.0, std=0.02)
        else:
            self.boundary_left = None
            self.boundary_right = None

    @property
    def has_boundary_conditioning(self) -> bool:
        return (
            self.boundary_transitions is not None
            or (
                self.boundary_left is not None
                and not self.context_conditioned
            )
        )

    def _boundary_transition_matrix(self) -> torch.Tensor:
        if self.boundary_transitions is not None:
            return self.boundary_transitions
        if self.boundary_left is None or self.boundary_right is None:
            raise RuntimeError("boundary transitions were not initialized")
        return self.transitions + self.boundary_left @ self.boundary_right

    def _validate(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        boundary_mask: Optional[torch.Tensor] = None,
        transition_gate: Optional[torch.Tensor] = None,
    ) -> None:
        if emissions.ndim != 3 or emissions.size(-1) != self.num_labels:
            raise ValueError(
                "CRF emissions must have shape [batch, length, labels]"
            )
        if mask.shape != emissions.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("CRF mask must be boolean [batch, length]")
        if not mask.any(dim=1).all():
            raise ValueError("every CRF sequence must contain a scored letter")
        if not self.has_boundary_conditioning:
            if boundary_mask is not None:
                raise ValueError(
                    "standard CRF does not accept a boundary mask"
                )
        else:
            if boundary_mask is None:
                raise ValueError(
                    "boundary-conditioned CRF requires a boundary mask"
                )
            if boundary_mask.shape != mask.shape:
                raise ValueError("CRF boundary mask must match mask shape")
            if boundary_mask.dtype != torch.bool:
                raise ValueError("CRF boundary mask must be boolean")
            if (boundary_mask & ~mask).any():
                raise ValueError(
                    "CRF boundaries may only mark scored letters"
                )
        if self.context_conditioned:
            if transition_gate is None:
                raise ValueError(
                    "context-conditioned CRF requires a transition gate"
                )
            if transition_gate.shape != mask.shape:
                raise ValueError("CRF transition gate must match mask shape")
            if not transition_gate.dtype.is_floating_point:
                raise ValueError("CRF transition gate must be floating point")
            if not torch.isfinite(transition_gate).all():
                raise ValueError("CRF transition gate must be finite")
            if transition_gate.lt(0.0).any() or transition_gate.gt(1.0).any():
                raise ValueError("CRF transition gate must be in [0, 1]")
        elif transition_gate is not None:
            raise ValueError(
                "non-contextual CRF does not accept a transition gate"
            )
        if targets is not None:
            if targets.shape != mask.shape:
                raise ValueError("CRF targets must match mask shape")
            active_targets = targets[mask]
            if active_targets.lt(0).any() or active_targets.ge(
                self.num_labels
            ).any():
                raise ValueError("active CRF targets are outside label range")

    def _transition_matrix(
        self,
        boundary: Optional[torch.Tensor],
        transition_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.context_conditioned:
            if transition_gate is None:
                raise ValueError("context transition gates are required")
            if self.boundary_left is None or self.boundary_right is None:
                raise RuntimeError("context transition residual is missing")
            residual = self.boundary_left @ self.boundary_right
            return self.transitions.unsqueeze(0) + (
                transition_gate[:, None, None] * residual.unsqueeze(0)
            )
        if not self.has_boundary_conditioning:
            return self.transitions.unsqueeze(0)
        if boundary is None:
            raise ValueError("boundary flags are required")
        return torch.where(
            boundary[:, None, None],
            self._boundary_transition_matrix().unsqueeze(0),
            self.transitions.unsqueeze(0),
        )

    def log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        transition_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate(
            emissions,
            mask,
            boundary_mask=boundary_mask,
            transition_gate=transition_gate,
        )
        batch_size, length, _ = emissions.shape
        alpha = emissions.new_zeros(batch_size, self.num_labels)
        started = torch.zeros(
            batch_size, dtype=torch.bool, device=emissions.device
        )
        for index in range(length):
            active = mask[:, index]
            emission = emissions[:, index]
            first_scores = self.start_transitions + emission
            boundary = (
                None
                if boundary_mask is None
                else boundary_mask[:, index]
            )
            gate = (
                None
                if transition_gate is None
                else transition_gate[:, index]
            )
            transition_scores = alpha.unsqueeze(
                2
            ) + self._transition_matrix(boundary, gate)
            next_scores = torch.logsumexp(
                transition_scores, dim=1
            ) + emission
            proposed = torch.where(
                started.unsqueeze(1), next_scores, first_scores
            )
            alpha = torch.where(active.unsqueeze(1), proposed, alpha)
            started = started | active
        return torch.logsumexp(
            alpha + self.end_transitions.unsqueeze(0), dim=1
        )

    def gold_score(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        transition_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate(
            emissions,
            mask,
            targets,
            boundary_mask=boundary_mask,
            transition_gate=transition_gate,
        )
        batch_size, length, _ = emissions.shape
        scores = emissions.new_zeros(batch_size)
        started = torch.zeros(
            batch_size, dtype=torch.bool, device=emissions.device
        )
        previous = torch.zeros(
            batch_size, dtype=torch.long, device=emissions.device
        )
        batch_indices = torch.arange(
            batch_size, device=emissions.device
        )
        for index in range(length):
            active = mask[:, index]
            current = targets[:, index].masked_fill(~active, 0)
            emission_score = emissions[batch_indices, index, current]
            first_score = self.start_transitions[current] + emission_score
            boundary = (
                None
                if boundary_mask is None
                else boundary_mask[:, index]
            )
            gate = (
                None
                if transition_gate is None
                else transition_gate[:, index]
            )
            transition_matrices = self._transition_matrix(boundary, gate)
            if transition_matrices.size(0) == 1 and batch_size > 1:
                transition_matrices = transition_matrices.expand(
                    batch_size, -1, -1
                )
            transition_score = transition_matrices[
                batch_indices, previous, current
            ] + emission_score
            contribution = torch.where(
                started, transition_score, first_score
            )
            scores = scores + contribution * active.to(scores.dtype)
            previous = torch.where(active, current, previous)
            started = started | active
        return scores + self.end_transitions[previous]

    def negative_log_likelihood(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        transition_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (
            self.log_partition(
                emissions, mask, boundary_mask, transition_gate
            )
            - self.gold_score(
                emissions,
                targets,
                mask,
                boundary_mask,
                transition_gate,
            )
        ).mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        transition_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate(
            emissions,
            mask,
            boundary_mask=boundary_mask,
            transition_gate=transition_gate,
        )
        batch_size, original_length, _ = emissions.shape
        lengths = mask.sum(dim=1)
        maximum_length = int(lengths.max().item())
        packed_positions = mask.long().cumsum(dim=1) - 1
        packed = emissions.new_zeros(
            batch_size, maximum_length, self.num_labels
        )
        batch_indices = torch.arange(
            batch_size, device=emissions.device
        ).unsqueeze(1).expand(batch_size, original_length)
        packed[
            batch_indices[mask], packed_positions[mask]
        ] = emissions[mask]
        packed_boundaries = torch.zeros(
            batch_size,
            maximum_length,
            dtype=torch.bool,
            device=emissions.device,
        )
        if boundary_mask is not None:
            packed_boundaries[
                batch_indices[mask], packed_positions[mask]
            ] = boundary_mask[mask]
        packed_gates: Optional[torch.Tensor] = None
        if transition_gate is not None:
            packed_gates = emissions.new_zeros(batch_size, maximum_length)
            packed_gates[
                batch_indices[mask], packed_positions[mask]
            ] = transition_gate[mask]

        scores = self.start_transitions.unsqueeze(0) + packed[:, 0]
        backpointers = []
        for index in range(1, maximum_length):
            candidates = (
                scores.unsqueeze(2)
                + self._transition_matrix(
                    packed_boundaries[:, index],
                    (
                        None
                        if packed_gates is None
                        else packed_gates[:, index]
                    ),
                )
            )
            best_scores, best_previous = candidates.max(dim=1)
            proposed = best_scores + packed[:, index]
            active = lengths.gt(index).unsqueeze(1)
            scores = torch.where(active, proposed, scores)
            backpointers.append(best_previous)

        current = (scores + self.end_transitions.unsqueeze(0)).argmax(
            dim=1
        )
        packed_decoded = torch.zeros(
            batch_size,
            maximum_length,
            dtype=torch.long,
            device=emissions.device,
        )
        for index in range(maximum_length - 1, -1, -1):
            active = lengths.gt(index)
            packed_decoded[:, index] = torch.where(
                active,
                current,
                packed_decoded[:, index],
            )
            if index > 0:
                previous = backpointers[index - 1].gather(
                    1, current.unsqueeze(1)
                ).squeeze(1)
                current = torch.where(active, previous, current)

        decoded = torch.zeros(
            batch_size,
            original_length,
            dtype=torch.long,
            device=emissions.device,
        )
        decoded[mask] = packed_decoded[
            batch_indices[mask], packed_positions[mask]
        ]
        return decoded

    @torch.no_grad()
    def k_best_segments(
        self,
        emissions: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return exact K-best paths for one contiguous word segment.

        Segment scores contain emissions and ordinary within-segment
        transitions only.  CRF start/end scores and cross-word transitions are
        intentionally excluded and are applied by the sentence word lattice.
        Ties are resolved lexicographically by the complete label path.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if emissions.ndim != 2 or emissions.size(1) != self.num_labels:
            raise ValueError(
                "segment emissions must have shape [length, labels]"
            )
        if emissions.size(0) <= 0:
            raise ValueError("segment emissions cannot be empty")
        if not torch.isfinite(emissions).all():
            raise ValueError("segment emissions must be finite")
        if self.has_boundary_conditioning or self.context_conditioned:
            raise ValueError(
                "segment K-best currently requires the standard linear CRF"
            )
        if not torch.isfinite(self.transitions).all():
            raise ValueError("CRF transitions must be finite")

        length = int(emissions.size(0))
        states: List[List[Tuple[float, Tuple[int, ...]]]] = []
        for label in range(self.num_labels):
            states.append(
                [(float(emissions[0, label].item()), (label,))]
            )

        for index in range(1, length):
            next_states: List[List[Tuple[float, Tuple[int, ...]]]] = []
            for current in range(self.num_labels):
                candidates: List[Tuple[float, Tuple[int, ...]]] = []
                emission_score = float(emissions[index, current].item())
                for previous in range(self.num_labels):
                    transition_score = float(
                        self.transitions[previous, current].item()
                    )
                    for score, path in states[previous]:
                        candidates.append(
                            (
                                score + transition_score + emission_score,
                                path + (current,),
                            )
                        )
                candidates.sort(key=lambda item: (-item[0], item[1]))
                next_states.append(candidates[:k])
            states = next_states

        completed = [candidate for label in states for candidate in label]
        completed.sort(key=lambda item: (-item[0], item[1]))
        selected = completed[:k]
        paths = torch.tensor(
            [path for _, path in selected],
            dtype=torch.long,
            device=emissions.device,
        )
        scores = emissions.new_tensor([score for score, _ in selected])
        return paths, scores

    def log_marginals(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        transition_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-letter normalized log marginals at original positions."""

        self._validate(
            emissions,
            mask,
            boundary_mask=boundary_mask,
            transition_gate=transition_gate,
        )
        batch_size, length, _ = emissions.shape

        # Forward values are carried through masked positions. At an active
        # position they represent the score of every label for the packed CRF
        # prefix ending at that position.
        alpha = emissions.new_zeros(batch_size, self.num_labels)
        started = torch.zeros(
            batch_size, dtype=torch.bool, device=emissions.device
        )
        forward_values = []
        for index in range(length):
            active = mask[:, index]
            emission = emissions[:, index]
            first = self.start_transitions.unsqueeze(0) + emission
            transition_matrix = self._transition_matrix(
                (
                    None
                    if boundary_mask is None
                    else boundary_mask[:, index]
                ),
                (
                    None
                    if transition_gate is None
                    else transition_gate[:, index]
                ),
            )
            continued = torch.logsumexp(
                alpha.unsqueeze(2) + transition_matrix, dim=1
            ) + emission
            proposed = torch.where(started.unsqueeze(1), continued, first)
            alpha = torch.where(active.unsqueeze(1), proposed, alpha)
            forward_values.append(alpha)
            started = started | active
        forward_tensor = torch.stack(forward_values, dim=1)
        log_normalizer = torch.logsumexp(
            alpha + self.end_transitions.unsqueeze(0), dim=1
        )

        # Reverse scan carries the emission, transition metadata, and beta of
        # the next active packed position across spaces and padding.
        next_beta = self.end_transitions.unsqueeze(0).expand(batch_size, -1)
        next_emission = emissions.new_zeros(batch_size, self.num_labels)
        has_next = torch.zeros(
            batch_size, dtype=torch.bool, device=emissions.device
        )
        if boundary_mask is None:
            next_boundary = None
        else:
            next_boundary = torch.zeros(
                batch_size, dtype=torch.bool, device=emissions.device
            )
        if transition_gate is None:
            next_gate = None
        else:
            next_gate = emissions.new_zeros(batch_size)

        backward_values = [emissions.new_empty(0)] * length
        for index in range(length - 1, -1, -1):
            active = mask[:, index]
            transition_matrix = self._transition_matrix(
                next_boundary, next_gate
            )
            continued = torch.logsumexp(
                transition_matrix
                + next_emission.unsqueeze(1)
                + next_beta.unsqueeze(1),
                dim=2,
            )
            current = torch.where(
                has_next.unsqueeze(1),
                continued,
                self.end_transitions.unsqueeze(0),
            )
            backward_values[index] = current
            next_beta = torch.where(active.unsqueeze(1), current, next_beta)
            next_emission = torch.where(
                active.unsqueeze(1), emissions[:, index], next_emission
            )
            if next_boundary is not None and boundary_mask is not None:
                next_boundary = torch.where(
                    active, boundary_mask[:, index], next_boundary
                )
            if next_gate is not None and transition_gate is not None:
                next_gate = torch.where(
                    active, transition_gate[:, index], next_gate
                )
            has_next = has_next | active
        backward_tensor = torch.stack(backward_values, dim=1)
        active_marginals = (
            forward_tensor
            + backward_tensor
            - log_normalizer[:, None, None]
        )
        uniform = -torch.log(emissions.new_tensor(float(self.num_labels)))
        return torch.where(
            mask.unsqueeze(-1),
            active_marginals,
            uniform.expand_as(active_marginals),
        )


class CharDiacritizer(nn.Module):
    """Character encoder with either direct or factorized 16-class decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture == "dual_rope_transformer":
            if config.resolved_head_mode not in (
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
            if config.guided_label_training:
                raise ValueError(
                    "dual_rope_transformer does not use guided label hints"
                )
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        if config.architecture == "dual_rope_transformer":
            self.position_embedding: Optional[nn.Embedding] = None
        else:
            self.position_embedding = nn.Embedding(
                config.max_length, config.d_model
            )
        if config.guided_label_training:
            self.hint_embedding: Optional[nn.Embedding] = nn.Embedding(
                NUM_LABELS + 1, config.d_model, padding_idx=0
            )
        else:
            self.hint_embedding = None
        if (
            config.architecture == "hierarchical_transformer"
            or config.word_position_features
        ):
            self.word_position_embedding: Optional[nn.Embedding] = nn.Embedding(
                config.max_word_length, config.d_model
            )
            self.reverse_word_position_embedding: Optional[
                nn.Embedding
            ] = nn.Embedding(config.max_word_length, config.d_model)
        else:
            self.word_position_embedding = None
            self.reverse_word_position_embedding = None
        if config.word_position_features:
            self.word_initial_embedding: Optional[nn.Embedding] = nn.Embedding(
                2, config.d_model
            )
            self.word_final_embedding: Optional[nn.Embedding] = nn.Embedding(
                2, config.d_model
            )
        else:
            self.word_initial_embedding = None
            self.word_final_embedding = None
        self.embedding_norm = nn.LayerNorm(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        if config.architecture in (
            "conv_local_transformer",
            "hierarchical_transformer",
        ):
            self.conv_frontend: Optional[nn.Module] = MultiKernelConvFrontend(
                config.d_model, config.conv_kernels, config.dropout
            )
        elif config.architecture in (
            "plain_transformer",
            "dual_rope_transformer",
        ):
            self.conv_frontend = None
        else:
            raise ValueError(
                "unsupported architecture: {}".format(config.architecture)
            )

        if config.architecture == "dual_rope_transformer":
            self.blocks = nn.ModuleList()
            self.dual_rope_encoder: Optional[DualRoPEEncoder] = (
                DualRoPEEncoder(config)
            )
        else:
            blocks = []
            for index in range(config.num_layers):
                use_global = config.architecture == "plain_transformer" or (
                    config.global_attention_every > 0
                    and (index + 1) % config.global_attention_every == 0
                )
                blocks.append(
                    TransformerBlock(
                        d_model=config.d_model,
                        num_heads=config.num_heads,
                        ffn_dim=config.ffn_dim,
                        dropout=config.dropout,
                        attention_window=(
                            None if use_global else config.attention_window
                        ),
                        shifted=bool(index % 2),
                    )
                )
            self.blocks = nn.ModuleList(blocks)
            self.dual_rope_encoder = None

        if config.architecture == "hierarchical_transformer":
            self.word_context_encoder: Optional[
                WordContextEncoder
            ] = WordContextEncoder(
                d_model=config.d_model,
                num_heads=config.num_heads,
                ffn_dim=config.word_ffn_dim,
                num_layers=config.word_num_layers,
                dropout=config.dropout,
            )
        else:
            self.word_context_encoder = None
        self.final_norm = nn.LayerNorm(config.d_model)

        head_mode = config.resolved_head_mode
        if head_mode in ("factorized", "gated_joint", "factorized_crf"):
            self.base_head: Optional[nn.Linear] = nn.Linear(
                config.d_model, NUM_BASE_LABELS
            )
            self.shadda_head: Optional[nn.Linear] = nn.Linear(
                config.d_model, 2
            )
        else:
            self.base_head = None
            self.shadda_head = None
        if head_mode in (
            "direct",
            "gated_joint",
            "crf",
            "boundary_crf",
            "low_rank_boundary_crf",
            "context_low_rank_boundary_crf",
        ):
            self.label_head = nn.Linear(config.d_model, NUM_LABELS)
        else:
            self.label_head = None
        if head_mode == "gated_joint":
            self.head_gate: Optional[nn.Linear] = nn.Linear(
                config.d_model, 1
            )
        else:
            self.head_gate = None
        if head_mode in (
            "crf",
            "boundary_crf",
            "factorized_crf",
            "low_rank_boundary_crf",
            "context_low_rank_boundary_crf",
        ):
            self.crf: Optional[LinearChainCRF] = LinearChainCRF(
                NUM_LABELS,
                boundary_conditioned=head_mode == "boundary_crf",
                boundary_rank=(
                    config.crf_boundary_rank
                    if head_mode in (
                        "low_rank_boundary_crf",
                        "context_low_rank_boundary_crf",
                    )
                    else 0
                ),
                context_conditioned=(
                    head_mode == "context_low_rank_boundary_crf"
                ),
            )
        else:
            self.crf = None
        if head_mode == "context_low_rank_boundary_crf":
            self.crf_context_gate: Optional[nn.Linear] = nn.Linear(
                config.d_model + 1, 1
            )
        else:
            self.crf_context_gate = None

        self.apply(self._initialize_weights)
        if self.head_gate is not None:
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        if self.dual_rope_encoder is not None:
            nn.init.zeros_(self.dual_rope_encoder.fusion_gate.weight)
            nn.init.zeros_(self.dual_rope_encoder.fusion_gate.bias)
        if self.crf_context_gate is not None:
            nn.init.zeros_(self.crf_context_gate.weight)
            nn.init.zeros_(self.crf_context_gate.bias)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_hints: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden, fusion_gate = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label_hints=label_hints,
        )
        batch_size, length = input_ids.shape

        head_mode = self.config.resolved_head_mode
        if head_mode in ("factorized", "gated_joint", "factorized_crf"):
            if self.base_head is None or self.shadda_head is None:
                raise RuntimeError("factorized heads were not initialized")
            base_logits = self.base_head(hidden)
            shadda_logits = self.shadda_head(hidden)
            base_log_probabilities = F.log_softmax(base_logits, dim=-1)
            shadda_log_probabilities = F.log_softmax(shadda_logits, dim=-1)
            label_log_probabilities = (
                shadda_log_probabilities.unsqueeze(-1)
                + base_log_probabilities.unsqueeze(-2)
            ).reshape(batch_size, length, NUM_LABELS)
            if head_mode == "gated_joint":
                if self.label_head is None or self.head_gate is None:
                    raise RuntimeError("gated joint heads were not initialized")
                joint_logits = self.label_head(hidden)
                joint_log_probabilities = F.log_softmax(
                    joint_logits, dim=-1
                )
                gate_logits = self.head_gate(hidden)
                mixed_log_probabilities = torch.logaddexp(
                    F.logsigmoid(gate_logits) + joint_log_probabilities,
                    F.logsigmoid(-gate_logits) + label_log_probabilities,
                )
                return {
                    "logits": mixed_log_probabilities,
                    "base_logits": base_logits,
                    "shadda_logits": shadda_logits,
                    "joint_logits": joint_logits,
                    "head_gate": torch.sigmoid(gate_logits),
                }
            factorized_outputs = {
                "logits": label_log_probabilities,
                "base_logits": base_logits,
                "shadda_logits": shadda_logits,
            }
            if head_mode == "factorized_crf":
                factorized_outputs["crf_mask"] = (
                    attention_mask
                    & input_ids.ne(self.config.space_id)
                    & input_ids.ne(self.config.bos_id)
                    & input_ids.ne(self.config.eos_id)
                )
                if fusion_gate is not None:
                    factorized_outputs["fusion_gate"] = fusion_gate
            return factorized_outputs

        if self.label_head is None:
            raise RuntimeError("direct label head was not initialized")
        direct_outputs = {"logits": self.label_head(hidden)}
        if self.crf is not None:
            direct_outputs["crf_mask"] = (
                attention_mask
                & input_ids.ne(self.config.space_id)
                & input_ids.ne(self.config.bos_id)
                & input_ids.ne(self.config.eos_id)
            )
            if head_mode in (
                "boundary_crf",
                "low_rank_boundary_crf",
            ):
                previous_is_space = F.pad(
                    input_ids[:, :-1].eq(self.config.space_id),
                    (1, 0),
                    value=False,
                )
                direct_outputs["crf_boundary_mask"] = (
                    direct_outputs["crf_mask"] & previous_is_space
                )
            elif head_mode == "context_low_rank_boundary_crf":
                if self.crf_context_gate is None:
                    raise RuntimeError("CRF context gate was not initialized")
                previous_is_space = F.pad(
                    input_ids[:, :-1].eq(self.config.space_id),
                    (1, 0),
                    value=False,
                )
                gate_features = torch.cat(
                    [
                        hidden,
                        previous_is_space.unsqueeze(-1).to(hidden.dtype),
                    ],
                    dim=-1,
                )
                transition_gate = torch.sigmoid(
                    self.crf_context_gate(gate_features)
                ).squeeze(-1)
                transition_gate = transition_gate.masked_fill(
                    ~direct_outputs["crf_mask"], 0.0
                )
                direct_outputs["crf_transition_gate"] = transition_gate
                direct_outputs["crf_boundary_indicator"] = (
                    direct_outputs["crf_mask"] & previous_is_space
                )
        if fusion_gate is not None:
            direct_outputs["fusion_gate"] = fusion_gate
        return direct_outputs

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_hints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return final contextual states without changing checkpoint state.

        The second value is the optional DualRoPE fusion gate.  `forward`
        delegates to this method, so existing checkpoints and output heads keep
        exactly the same parameters and numerical path.
        """
        batch_size, length = input_ids.shape
        if length > self.config.max_length:
            raise ValueError(
                "sequence length {} exceeds configured max_length {}".format(
                    length, self.config.max_length
                )
            )
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, length)
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(positions)
        if self.hint_embedding is not None:
            if label_hints is None:
                label_hints = torch.zeros_like(input_ids)
            if label_hints.shape != input_ids.shape:
                raise ValueError("label_hints must match input_ids shape")
            if torch.any(label_hints.lt(0)) or torch.any(
                label_hints.gt(NUM_LABELS)
            ):
                raise ValueError("label_hints must be in [0, 16]")
            hidden = hidden + self.hint_embedding(label_hints)

        word_features: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        if (
            self.word_context_encoder is not None
            or self.config.word_position_features
        ):
            word_features = build_word_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                space_id=self.config.space_id,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
                max_word_length=self.config.max_word_length,
            )
            _, _, within_word, reverse_within_word = word_features
            content_mask = word_features[0]
            if (
                self.word_position_embedding is None
                or self.reverse_word_position_embedding is None
            ):
                raise RuntimeError("word-position embeddings are missing")
            word_position_features = (
                self.word_position_embedding(within_word)
                + self.reverse_word_position_embedding(reverse_within_word)
            )
            if self.config.word_position_features:
                if (
                    self.word_initial_embedding is None
                    or self.word_final_embedding is None
                ):
                    raise RuntimeError(
                        "word boundary embeddings are missing"
                    )
                word_position_features = (
                    word_position_features
                    + self.word_initial_embedding(
                        (content_mask & within_word.eq(0)).long()
                    )
                    + self.word_final_embedding(
                        (content_mask & reverse_within_word.eq(0)).long()
                    )
                )
            word_position_features = word_position_features.masked_fill(
                ~content_mask.unsqueeze(-1), 0.0
            )
            hidden = (
                hidden
                + word_position_features
            )
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        hidden = hidden.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        if self.conv_frontend is not None:
            hidden = self.conv_frontend(hidden, attention_mask)
        fusion_gate: Optional[torch.Tensor] = None
        if self.dual_rope_encoder is not None:
            hidden, fusion_gate = self.dual_rope_encoder(
                hidden, attention_mask, positions
            )
        else:
            for block in self.blocks:
                hidden = block(hidden, attention_mask)
        if self.word_context_encoder is not None:
            if word_features is None:
                raise RuntimeError("word features were not built")
            content_mask, word_ids, _, _ = word_features
            hidden = self.word_context_encoder(
                hidden, content_mask, word_ids
            )
        hidden = self.final_norm(hidden)
        hidden = hidden.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        return hidden, fusion_gate

    def compute_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
        shadda_loss_weight: float,
    ) -> torch.Tensor:
        if self.config.resolved_head_mode == "factorized":
            valid = targets.ne(IGNORE_INDEX)
            safe_targets = targets.masked_fill(~valid, 0)
            base_targets = (safe_targets % NUM_BASE_LABELS).masked_fill(
                ~valid, IGNORE_INDEX
            )
            shadda_targets = (
                safe_targets // NUM_BASE_LABELS
            ).masked_fill(~valid, IGNORE_INDEX)
            base_loss = F.cross_entropy(
                outputs["base_logits"].reshape(-1, NUM_BASE_LABELS),
                base_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            shadda_loss = F.cross_entropy(
                outputs["shadda_logits"].reshape(-1, 2),
                shadda_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            return base_loss + shadda_loss_weight * shadda_loss

        if self.config.resolved_head_mode in (
            "crf",
            "boundary_crf",
            "factorized_crf",
            "low_rank_boundary_crf",
            "context_low_rank_boundary_crf",
        ):
            if self.crf is None:
                raise RuntimeError("CRF was not initialized")
            return self.crf.negative_log_likelihood(
                outputs["logits"],
                targets,
                outputs["crf_mask"].bool(),
                outputs.get("crf_boundary_mask"),
                outputs.get("crf_transition_gate"),
            )

        return F.cross_entropy(
            outputs["logits"].reshape(-1, NUM_LABELS),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def decode_outputs(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.config.resolved_head_mode in (
            "crf",
            "boundary_crf",
            "factorized_crf",
            "low_rank_boundary_crf",
            "context_low_rank_boundary_crf",
        ):
            if self.crf is None:
                raise RuntimeError("CRF was not initialized")
            return self.crf.decode(
                outputs["logits"],
                outputs["crf_mask"].bool(),
                outputs.get("crf_boundary_mask"),
                outputs.get("crf_transition_gate"),
            )
        return outputs["logits"].argmax(dim=-1)

    def log_probabilities(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.config.resolved_head_mode in (
            "crf",
            "boundary_crf",
            "factorized_crf",
            "low_rank_boundary_crf",
            "context_low_rank_boundary_crf",
        ):
            if self.crf is None:
                raise RuntimeError("CRF was not initialized")
            return self.crf.log_marginals(
                outputs["logits"],
                outputs["crf_mask"].bool(),
                outputs.get("crf_boundary_mask"),
                outputs.get("crf_transition_gate"),
            )
        return F.log_softmax(outputs["logits"], dim=-1)

    def probabilities(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.log_probabilities(outputs).exp()

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.decode_outputs(self.forward(input_ids, attention_mask))
