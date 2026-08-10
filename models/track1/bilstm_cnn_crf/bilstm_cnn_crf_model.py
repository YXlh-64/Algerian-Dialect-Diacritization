"""Track 1 P2 character-level BiLSTM-CNN-CRF architecture.

Extracted from the validated Kaggle experiment.  Runtime hyperparameters are
passed in by the training entry point so this module has no notebook globals.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def class_balanced_focal_loss(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    letter_mask: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    selected_logits = emissions[letter_mask]
    selected_labels = labels[letter_mask]
    log_probabilities = F.log_softmax(selected_logits, dim=-1, dtype=torch.float32)
    cross_entropy = F.nll_loss(
        log_probabilities,
        selected_labels,
        weight=class_weights,
        reduction="none",
    )
    log_true_probability = log_probabilities.gather(
        1, selected_labels.unsqueeze(1)
    ).squeeze(1)
    focal_factor = (1.0 - log_true_probability.exp()).pow(gamma)
    return (focal_factor * cross_entropy).mean()


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self._score_sentence(emissions, tags, mask) - self._log_partition(
            emissions, mask
        )

    def _score_sentence(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = emissions.shape
        batch_index = torch.arange(batch_size, device=emissions.device)
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[batch_index, 0, tags[:, 0]]
        for timestep in range(1, sequence_length):
            active = mask[:, timestep]
            transition_score = self.transitions[
                tags[:, timestep - 1], tags[:, timestep]
            ]
            emission_score = emissions[batch_index, timestep, tags[:, timestep]]
            score = score + (transition_score + emission_score) * active
        last_positions = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _log_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.start_transitions + emissions[:, 0]
        for timestep in range(1, emissions.size(1)):
            scores = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, timestep].unsqueeze(1)
            )
            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, timestep].unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        score = self.start_transitions + emissions[:, 0]
        history = []
        for timestep in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_previous_tag = next_score.max(dim=1)
            best_score = best_score + emissions[:, timestep]
            score = torch.where(mask[:, timestep].unsqueeze(1), best_score, score)
            history.append(best_previous_tag)
        best_last_tags = (score + self.end_transitions).argmax(dim=1)
        sequence_lengths = mask.long().sum(dim=1)
        paths = []
        for row in range(emissions.size(0)):
            length = int(sequence_lengths[row])
            tag = int(best_last_tags[row])
            path = [tag]
            for backpointer in reversed(history[: length - 1]):
                tag = int(backpointer[row, tag])
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths


class BiLSTMDiacritizer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        num_labels: int,
        use_cnn: bool,
        use_crf: bool,
        config: Any,
        pad_id: int,
    ):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_crf = use_crf
        self.config = config
        self.char_embedding = nn.Embedding(
            vocabulary_size, self.config.embedding_dim, padding_idx=pad_id
        )
        self.boundary_embedding = nn.Embedding(5, self.config.boundary_dim)
        base_dim = self.config.embedding_dim + self.config.boundary_dim
        if use_cnn:
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        base_dim,
                        self.config.cnn_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    )
                    for kernel in self.config.cnn_kernels
                ]
            )
            input_dim = base_dim + self.config.cnn_channels * len(
                self.config.cnn_kernels
            )
        else:
            self.convolutions = nn.ModuleList()
            input_dim = base_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.config.model_dim),
            nn.LayerNorm(self.config.model_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=self.config.model_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.lstm_layers,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(2 * self.config.hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(2 * self.config.hidden_dim, self.config.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.mlp_dim, num_labels),
        )
        self.crf = LinearChainCRF(num_labels) if use_crf else None

    def emissions(self, batch: dict[str, Any]) -> torch.Tensor:
        char_features = self.char_embedding(batch["tokens"])
        boundary_features_ = self.boundary_embedding(batch["boundaries"])
        features = torch.cat([char_features, boundary_features_], dim=-1)
        if self.use_cnn:
            channels_first = features.transpose(1, 2)
            convolution_features = [
                F.gelu(convolution(channels_first)).transpose(1, 2)
                for convolution in self.convolutions
            ]
            features = torch.cat([features, *convolution_features], dim=-1)
        features = self.input_projection(features)
        packed = pack_padded_sequence(
            features,
            batch["lengths"].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.bilstm(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=batch["tokens"].size(1)
        )
        emissions = self.output_head(encoded)
        invalid_on_spaces = batch["spaces"].unsqueeze(-1).expand_as(emissions).clone()
        invalid_on_spaces[..., 0] = False
        return emissions.masked_fill(invalid_on_spaces, -1e4)

    def loss(
        self, batch: dict[str, Any], class_weights: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        emissions = self.emissions(batch)
        letter_mask = batch["mask"] & ~batch["spaces"]
        focal = class_balanced_focal_loss(
            emissions,
            batch["labels"],
            letter_mask,
            class_weights,
            self.config.focal_gamma,
        )
        if self.crf is None:
            return focal, {"focal": float(focal.detach())}
        crf_loss = -self.crf.log_likelihood(
            emissions, batch["labels"], batch["mask"]
        ).mean()
        return crf_loss + self.config.crf_aux_weight * focal, {
            "crf": float(crf_loss.detach()),
            "focal": float(focal.detach()),
        }

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        if self.crf is not None:
            return self.crf.decode(emissions, mask)
        lengths = mask.long().sum(dim=1).tolist()
        predictions = emissions.argmax(dim=-1)
        return [
            predictions[row, : int(length)].tolist()
            for row, length in enumerate(lengths)
        ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
