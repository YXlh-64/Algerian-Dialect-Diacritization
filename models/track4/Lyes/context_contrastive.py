"""Shared-encoder sentence/isolated-word fusion for ContextContrastive-v15."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import IGNORE_INDEX, SentenceRecord
from utils.track4.Lyes.labels import NUM_LABELS
from utils.track4.Lyes.lexical_fusion import iter_words
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer


@dataclass(frozen=True)
class IsolatedWordPacking:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    sentence_indices: torch.Tensor
    sentence_positions: torch.Tensor
    word_rows: torch.Tensor
    word_positions: torch.Tensor


def pack_isolated_words(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    space_id: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> IsolatedWordPacking:
    """Pack each non-empty word as BOS + letters + EOS deterministically."""

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be aligned [B, L]")
    cpu_ids = input_ids.detach().cpu()
    cpu_mask = attention_mask.detach().cpu()
    words: List[Tuple[int, Tuple[int, ...]]] = []
    for sentence_index in range(input_ids.size(0)):
        valid_length = int(cpu_mask[sentence_index].sum().item())
        start: Optional[int] = None
        for position in range(1, max(1, valid_length - 1)):
            token = int(cpu_ids[sentence_index, position].item())
            if token == space_id:
                if start is not None:
                    words.append((sentence_index, tuple(range(start, position))))
                    start = None
            elif token not in (bos_id, eos_id, pad_id):
                if start is None:
                    start = position
        if start is not None:
            words.append((sentence_index, tuple(range(start, valid_length - 1))))
    if not words:
        empty = input_ids.new_empty((0,), dtype=torch.long)
        return IsolatedWordPacking(
            input_ids=input_ids.new_empty((0, 2)),
            attention_mask=attention_mask.new_empty((0, 2)),
            sentence_indices=empty,
            sentence_positions=empty,
            word_rows=empty,
            word_positions=empty,
        )
    max_word_length = max(len(positions) for _, positions in words)
    packed_ids_cpu = torch.full(
        (len(words), max_word_length + 2), pad_id, dtype=torch.long
    )
    packed_mask_cpu = torch.zeros(
        (len(words), max_word_length + 2), dtype=torch.bool
    )
    sentence_indices: List[int] = []
    sentence_positions: List[int] = []
    word_rows: List[int] = []
    word_positions: List[int] = []
    for row, (sentence_index, positions) in enumerate(words):
        length = len(positions)
        packed_ids_cpu[row, 0] = bos_id
        packed_ids_cpu[row, 1 : length + 1] = cpu_ids[
            sentence_index, list(positions)
        ]
        packed_ids_cpu[row, length + 1] = eos_id
        packed_mask_cpu[row, : length + 2] = True
        sentence_indices.extend([sentence_index] * length)
        sentence_positions.extend(positions)
        word_rows.extend([row] * length)
        word_positions.extend(range(1, length + 1))
    return IsolatedWordPacking(
        input_ids=packed_ids_cpu.to(input_ids.device),
        attention_mask=packed_mask_cpu.to(input_ids.device),
        sentence_indices=torch.tensor(sentence_indices, dtype=torch.long, device=input_ids.device),
        sentence_positions=torch.tensor(sentence_positions, dtype=torch.long, device=input_ids.device),
        word_rows=torch.tensor(word_rows, dtype=torch.long, device=input_ids.device),
        word_positions=torch.tensor(word_positions, dtype=torch.long, device=input_ids.device),
    )


class ContextContrastiveModel(nn.Module):
    """Warm-started v7 with a zero-residual isolated-word fusion branch."""

    def __init__(self, base: CharDiacritizer, gate_hidden_dim: int = 256) -> None:
        super().__init__()
        if base.config.resolved_head_mode != "crf" or base.crf is None:
            raise ValueError("ContextContrastive-v15 requires standard v7 CRF")
        if base.label_head is None:
            raise ValueError("ContextContrastive-v15 requires direct label emissions")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        self.base = base
        d_model = base.config.d_model
        self.residual_projection = nn.Linear(d_model, d_model)
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 3, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    @property
    def config(self) -> Any:
        return self.base.config

    def isolated_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packing = pack_isolated_words(
            input_ids,
            attention_mask,
            space_id=self.config.space_id,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
            pad_id=self.config.pad_id,
        )
        output = input_ids.new_zeros(
            (*input_ids.shape, self.config.d_model), dtype=self.base.token_embedding.weight.dtype
        )
        content_mask = attention_mask & input_ids.ne(self.config.space_id)
        content_mask &= input_ids.ne(self.config.bos_id) & input_ids.ne(self.config.eos_id)
        if packing.input_ids.size(0) == 0:
            return output, content_mask
        isolated, _ = self.base.encode(packing.input_ids, packing.attention_mask)
        gathered = isolated[packing.word_rows, packing.word_positions]
        flat_indices = packing.sentence_indices * input_ids.size(1) + packing.sentence_positions
        output = output.reshape(-1, self.config.d_model).index_copy(0, flat_indices, gathered)
        return output.reshape(*input_ids.shape, self.config.d_model), content_mask

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        context, fusion_gate = self.base.encode(input_ids, attention_mask)
        isolated, content_mask = self.isolated_states(input_ids, attention_mask)
        difference = isolated - context
        gate_features = torch.cat((context, isolated, difference.abs()), dim=-1)
        gate = torch.sigmoid(self.gate_network(gate_features)).squeeze(-1)
        gate = gate.masked_fill(~content_mask, 0.0)
        fused = context + gate.unsqueeze(-1) * self.residual_projection(difference)
        fused = fused.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        outputs: Dict[str, torch.Tensor] = {
            "logits": self.base.label_head(fused),
            "crf_mask": content_mask,
            "context_gate": gate,
            "context_dependency_probability": 1.0 - gate,
        }
        if fusion_gate is not None:
            outputs["fusion_gate"] = fusion_gate
        return outputs

    def compute_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
        ambiguity_targets: torch.Tensor,
        auxiliary_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if auxiliary_coefficient < 0.0:
            raise ValueError("auxiliary_coefficient cannot be negative")
        nll = self.base.compute_loss(outputs, targets, 1.0)
        valid = ambiguity_targets.ne(IGNORE_INDEX) & outputs["crf_mask"].bool()
        if not bool(valid.any()):
            auxiliary = nll.new_zeros(())
        else:
            auxiliary = F.binary_cross_entropy(
                outputs["context_dependency_probability"][valid],
                ambiguity_targets[valid].to(nll.dtype),
            )
        return nll + auxiliary_coefficient * auxiliary, nll, auxiliary

    def decode_outputs(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.base.decode_outputs(outputs)

    def log_probabilities(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.base.log_probabilities(outputs)

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.decode_outputs(self.forward(input_ids, attention_mask))


class AmbiguityIndex:
    """Training-only word-position ambiguity targets."""

    def __init__(self, variants: Mapping[Tuple[str, int], Sequence[int]]) -> None:
        self.variants = {
            (str(word), int(position)): frozenset(int(label) for label in labels)
            for (word, position), labels in variants.items()
        }

    @classmethod
    def fit(cls, records: Sequence[SentenceRecord]) -> "AmbiguityIndex":
        observed: Dict[Tuple[str, int], set] = {}
        for record in records:
            if record.labels is None:
                raise ValueError("ambiguity fitting requires labels")
            for start, end, word in iter_words(record.chars):
                for offset, label in enumerate(record.labels[start:end]):
                    observed.setdefault((word, offset), set()).add(int(label))
        return cls(observed)

    def targets(
        self,
        records: Sequence[SentenceRecord],
        padded_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        targets = torch.full(
            (len(records), padded_length), IGNORE_INDEX, dtype=torch.long, device=device
        )
        for row, record in enumerate(records):
            for start, end, word in iter_words(record.chars):
                for offset in range(end - start):
                    labels = self.variants.get((word, offset))
                    if labels:
                        targets[row, start + offset + 1] = int(len(labels) > 1)
        return targets


def load_context_contrastive_checkpoint(
    path: Path, device: torch.device
) -> Tuple[ContextContrastiveModel, Dict[str, int], Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("system_type") != "context_contrastive_v15":
        raise ValueError("invalid ContextContrastive-v15 checkpoint")
    base_checkpoint = {
        "schema_version": 1,
        "model_config": checkpoint["model_config"],
        "model_state_dict": checkpoint["base_model_state_dict"],
        "vocab": checkpoint["vocab"],
    }
    base, vocab = build_model_from_checkpoint(base_checkpoint, device)
    model = ContextContrastiveModel(base, int(checkpoint["gate_hidden_dim"])).to(device)
    model.residual_projection.load_state_dict(checkpoint["residual_projection_state_dict"])
    model.gate_network.load_state_dict(checkpoint["gate_network_state_dict"])
    model.eval()
    return model, vocab, checkpoint


def save_context_contrastive_checkpoint(
    path: Path,
    model: ContextContrastiveModel,
    vocab: Mapping[str, int],
    gate_hidden_dim: int,
    epoch: int,
    auxiliary_coefficient: float,
    metrics: Mapping[str, Any],
) -> None:
    if epoch <= 0:
        raise ValueError("checkpoint epoch must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "system_type": "context_contrastive_v15",
        "model_config": asdict(model.base.config),
        "vocab": {str(key): int(value) for key, value in vocab.items()},
        "base_model_state_dict": model.base.state_dict(),
        "residual_projection_state_dict": model.residual_projection.state_dict(),
        "gate_network_state_dict": model.gate_network.state_dict(),
        "gate_hidden_dim": int(gate_hidden_dim),
        "epoch": int(epoch),
        "auxiliary_coefficient": float(auxiliary_coefficient),
        "dev_metrics": dict(metrics),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def initialize_from_v7(
    checkpoint_path: Path, device: torch.device, gate_hidden_dim: int = 256
) -> Tuple[ContextContrastiveModel, Dict[str, int], Mapping[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    base, vocab = build_model_from_checkpoint(checkpoint, device)
    return ContextContrastiveModel(base, gate_hidden_dim).to(device), vocab, checkpoint
