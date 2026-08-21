"""Deterministic filtered word lattices for Track 4 CRF models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from utils.track4.Lyes.data import SentenceRecord
from utils.track4.Lyes.labels import NUM_LABELS
from utils.track4.Lyes.lexical_fusion import iter_words
from models.track4.Lyes.dual_stream_crf_head import LinearChainCRF, TransformerBlock


@dataclass(frozen=True)
class WordCandidate:
    labels: Tuple[int, ...]
    base_score: float
    candidate_hash: str


@dataclass(frozen=True)
class WordLattice:
    sent_id: str
    char_count: int
    spans: Tuple[Tuple[int, int], ...]
    candidates: Tuple[Tuple[WordCandidate, ...], ...]
    baseline_labels: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.char_count <= 0:
            raise ValueError("word lattice char_count must be positive")
        if len(self.baseline_labels) != self.char_count:
            raise ValueError("baseline labels must match char_count")
        if len(self.spans) != len(self.candidates) or not self.spans:
            raise ValueError("word spans and candidate groups must align")
        previous_end = 0
        for (start, end), group in zip(self.spans, self.candidates):
            if start < previous_end or end <= start or end > self.char_count:
                raise ValueError("word spans must be ordered and nonempty")
            if not group:
                raise ValueError("every word must have at least one candidate")
            length = end - start
            if any(len(candidate.labels) != length for candidate in group):
                raise ValueError("candidate labels must match their word span")
            if group[0].labels != self.baseline_labels[start:end]:
                raise ValueError("the first candidate must be the baseline")
            if len({candidate.labels for candidate in group}) != len(group):
                raise ValueError("word candidates must be unique")
            previous_end = end


def candidate_hash(labels: Sequence[int]) -> str:
    if not labels or any(label < 0 or label >= NUM_LABELS for label in labels):
        raise ValueError("candidate labels must be nonempty and in range")
    payload = ",".join(str(int(label)) for label in labels).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def internal_crf_score(
    labels: Sequence[int],
    emissions: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    if emissions.ndim != 2 or emissions.size(1) != NUM_LABELS:
        raise ValueError("emissions must be [word_length, 16]")
    if transitions.shape != (NUM_LABELS, NUM_LABELS):
        raise ValueError("transitions must be [16, 16]")
    if len(labels) != emissions.size(0) or not labels:
        raise ValueError("labels must match the nonempty word emissions")
    indices = torch.tensor(labels, dtype=torch.long, device=emissions.device)
    positions = torch.arange(emissions.size(0), device=emissions.device)
    score = emissions[positions, indices].sum()
    if indices.numel() > 1:
        score = score + transitions[indices[:-1], indices[1:]].sum()
    return score


@torch.no_grad()
def build_word_lattice(
    record: SentenceRecord,
    emissions: torch.Tensor,
    baseline_labels: Sequence[int],
    crf: LinearChainCRF,
    k: int,
) -> WordLattice:
    if k not in (4, 8):
        raise ValueError("filtered word lattice k must be 4 or 8")
    if emissions.shape != (len(record.chars), NUM_LABELS):
        raise ValueError("record emissions must be [characters, 16]")
    if len(baseline_labels) != len(record.chars):
        raise ValueError("baseline labels must match record characters")
    if any(
        char == " " and int(label) != 0
        for char, label in zip(record.chars, baseline_labels)
    ):
        raise ValueError("space labels must be zero")

    spans: List[Tuple[int, int]] = []
    candidate_groups: List[Tuple[WordCandidate, ...]] = []
    for start, end, _ in iter_words(record.chars):
        spans.append((start, end))
        word_emissions = emissions[start:end]
        paths, scores = crf.k_best_segments(word_emissions, k)
        baseline = tuple(int(label) for label in baseline_labels[start:end])
        baseline_score = float(
            internal_crf_score(
                baseline, word_emissions, crf.transitions
            ).item()
        )
        group: List[WordCandidate] = [
            WordCandidate(
                labels=baseline,
                base_score=baseline_score,
                candidate_hash=candidate_hash(baseline),
            )
        ]
        seen = {baseline}
        for path, score in zip(paths.tolist(), scores.tolist()):
            labels = tuple(int(label) for label in path)
            if labels in seen:
                continue
            group.append(
                WordCandidate(
                    labels=labels,
                    base_score=float(score),
                    candidate_hash=candidate_hash(labels),
                )
            )
            seen.add(labels)
            if len(group) == k:
                break
        candidate_groups.append(tuple(group))

    if not spans:
        raise ValueError("a word lattice requires at least one word")
    return WordLattice(
        sent_id=record.sent_id,
        char_count=len(record.chars),
        spans=tuple(spans),
        candidates=tuple(candidate_groups),
        baseline_labels=tuple(int(label) for label in baseline_labels),
    )


def _validate_lattice_scores(
    lattice: WordLattice,
    word_scores: Sequence[torch.Tensor],
    transitions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> None:
    if len(word_scores) != len(lattice.candidates):
        raise ValueError("word score groups must match the lattice")
    if transitions.shape != (NUM_LABELS, NUM_LABELS):
        raise ValueError("transitions must be [16, 16]")
    if start_transitions.shape != (NUM_LABELS,):
        raise ValueError("start transitions must be [16]")
    if end_transitions.shape != (NUM_LABELS,):
        raise ValueError("end transitions must be [16]")
    for scores, candidates in zip(word_scores, lattice.candidates):
        if scores.ndim != 1 or scores.numel() != len(candidates):
            raise ValueError("each score vector must match its candidates")
        if not torch.isfinite(scores).all():
            raise ValueError("lattice scores must be finite")


def base_word_scores(
    lattice: WordLattice,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    return [
        torch.tensor(
            [candidate.base_score for candidate in group],
            device=device,
            dtype=dtype,
        )
        for group in lattice.candidates
    ]


def lattice_viterbi(
    lattice: WordLattice,
    word_scores: Sequence[torch.Tensor],
    transitions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> List[int]:
    _validate_lattice_scores(
        lattice,
        word_scores,
        transitions,
        start_transitions,
        end_transitions,
    )
    first_labels = torch.tensor(
        [candidate.labels[0] for candidate in lattice.candidates[0]],
        dtype=torch.long,
        device=word_scores[0].device,
    )
    scores = word_scores[0] + start_transitions[first_labels]
    backpointers: List[torch.Tensor] = []
    for word_index in range(1, len(lattice.candidates)):
        previous_last = torch.tensor(
            [
                candidate.labels[-1]
                for candidate in lattice.candidates[word_index - 1]
            ],
            dtype=torch.long,
            device=scores.device,
        )
        current_first = torch.tensor(
            [
                candidate.labels[0]
                for candidate in lattice.candidates[word_index]
            ],
            dtype=torch.long,
            device=scores.device,
        )
        boundary = transitions[
            previous_last.unsqueeze(1), current_first.unsqueeze(0)
        ]
        candidates = scores.unsqueeze(1) + boundary
        best, previous = candidates.max(dim=0)
        scores = best + word_scores[word_index]
        backpointers.append(previous)

    final_last = torch.tensor(
        [candidate.labels[-1] for candidate in lattice.candidates[-1]],
        dtype=torch.long,
        device=scores.device,
    )
    current = int((scores + end_transitions[final_last]).argmax().item())
    selected = [current]
    for previous in reversed(backpointers):
        current = int(previous[current].item())
        selected.append(current)
    selected.reverse()

    predictions = [0] * lattice.char_count
    for (start, end), group, candidate_index in zip(
        lattice.spans, lattice.candidates, selected
    ):
        predictions[start:end] = group[candidate_index].labels
    return predictions


def lattice_marginals(
    lattice: WordLattice,
    word_scores: Sequence[torch.Tensor],
    transitions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> torch.Tensor:
    _validate_lattice_scores(
        lattice,
        word_scores,
        transitions,
        start_transitions,
        end_transitions,
    )
    device = word_scores[0].device
    dtype = word_scores[0].dtype
    alphas: List[torch.Tensor] = []
    first_labels = torch.tensor(
        [candidate.labels[0] for candidate in lattice.candidates[0]],
        dtype=torch.long,
        device=device,
    )
    alpha = word_scores[0] + start_transitions[first_labels]
    alphas.append(alpha)
    for word_index in range(1, len(lattice.candidates)):
        previous_last = torch.tensor(
            [
                candidate.labels[-1]
                for candidate in lattice.candidates[word_index - 1]
            ],
            dtype=torch.long,
            device=device,
        )
        current_first = torch.tensor(
            [
                candidate.labels[0]
                for candidate in lattice.candidates[word_index]
            ],
            dtype=torch.long,
            device=device,
        )
        boundary = transitions[
            previous_last.unsqueeze(1), current_first.unsqueeze(0)
        ]
        alpha = torch.logsumexp(alpha.unsqueeze(1) + boundary, dim=0)
        alpha = alpha + word_scores[word_index]
        alphas.append(alpha)

    last_labels = torch.tensor(
        [candidate.labels[-1] for candidate in lattice.candidates[-1]],
        dtype=torch.long,
        device=device,
    )
    log_partition = torch.logsumexp(
        alphas[-1] + end_transitions[last_labels], dim=0
    )

    betas: List[torch.Tensor] = [
        torch.empty(0, device=device, dtype=dtype)
        for _ in lattice.candidates
    ]
    beta = end_transitions[last_labels]
    betas[-1] = beta
    for word_index in range(len(lattice.candidates) - 2, -1, -1):
        previous_last = torch.tensor(
            [
                candidate.labels[-1]
                for candidate in lattice.candidates[word_index]
            ],
            dtype=torch.long,
            device=device,
        )
        next_first = torch.tensor(
            [
                candidate.labels[0]
                for candidate in lattice.candidates[word_index + 1]
            ],
            dtype=torch.long,
            device=device,
        )
        boundary = transitions[
            previous_last.unsqueeze(1), next_first.unsqueeze(0)
        ]
        beta = torch.logsumexp(
            boundary
            + word_scores[word_index + 1].unsqueeze(0)
            + beta.unsqueeze(0),
            dim=1,
        )
        betas[word_index] = beta

    marginals = torch.zeros(
        lattice.char_count, NUM_LABELS, device=device, dtype=dtype
    )
    covered = torch.zeros(
        lattice.char_count, dtype=torch.bool, device=device
    )
    for word_index, ((start, end), group) in enumerate(
        zip(lattice.spans, lattice.candidates)
    ):
        candidate_log_probs = (
            alphas[word_index] + betas[word_index] - log_partition
        )
        candidate_probs = candidate_log_probs.exp()
        for candidate_index, candidate in enumerate(group):
            probability = candidate_probs[candidate_index]
            for offset, label in enumerate(candidate.labels):
                marginals[start + offset, label] += probability
        covered[start:end] = True
    marginals[~covered, 0] = 1.0
    normalization = marginals.sum(dim=-1, keepdim=True)
    if normalization.le(0.0).any():
        raise RuntimeError("lattice produced an empty character marginal")
    return marginals / normalization


def oracle_predictions(
    lattice: WordLattice,
    gold_labels: Sequence[int],
) -> Tuple[List[int], int, int]:
    if len(gold_labels) != lattice.char_count:
        raise ValueError("gold labels must match lattice char_count")
    predictions = list(lattice.baseline_labels)
    covered_words = 0
    recovered_wrong_words = 0
    for (start, end), group in zip(lattice.spans, lattice.candidates):
        gold = tuple(int(label) for label in gold_labels[start:end])
        baseline = tuple(lattice.baseline_labels[start:end])
        if any(candidate.labels == gold for candidate in group):
            predictions[start:end] = gold
            covered_words += 1
            if baseline != gold:
                recovered_wrong_words += 1
    return predictions, covered_words, recovered_wrong_words


class WordCandidateTransformer(nn.Module):
    """One-block Transformer that predicts an additive candidate residual."""

    def __init__(
        self,
        context_dim: int = 256,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("candidate d_model must divide num_heads")
        self.context_projection = nn.Linear(context_dim, d_model)
        self.label_embedding = nn.Embedding(NUM_LABELS, d_model)
        self.block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_window=None,
            shifted=False,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(
        self,
        context: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("candidate context must be [batch, length, dim]")
        if labels.shape != context.shape[:2] or mask.shape != labels.shape:
            raise ValueError("candidate labels and mask must match context")
        if mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise ValueError("every candidate must contain a valid letter")
        if labels.lt(0).any() or labels.ge(NUM_LABELS).any():
            raise ValueError("candidate labels are outside label range")
        hidden = self.context_projection(context) + self.label_embedding(labels)
        hidden = self.block(hidden, mask)
        hidden = self.final_norm(hidden)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
        return self.output(pooled).squeeze(-1)
