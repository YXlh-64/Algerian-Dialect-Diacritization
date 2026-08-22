"""Differentiable R-Drop consistency losses for CRF sequence taggers."""

from typing import Mapping

import torch

from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer


def symmetric_log_probability_kl(
    log_probabilities_a: torch.Tensor,
    log_probabilities_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return sentence-normalized symmetric KL over active positions.

    The per-label divergence is summed over labels and scored letters, then
    averaged over sentences. This matches the sequence-level reduction used
    by the CRF negative log-likelihood.
    """

    if log_probabilities_a.shape != log_probabilities_b.shape:
        raise ValueError("R-Drop log-probability shapes must match")
    if log_probabilities_a.ndim != 3:
        raise ValueError(
            "R-Drop log probabilities must be [batch, length, labels]"
        )
    if mask.shape != log_probabilities_a.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("R-Drop mask must be boolean [batch, length]")
    if not mask.any(dim=1).all():
        raise ValueError("every R-Drop sequence must contain a scored letter")
    if not torch.isfinite(log_probabilities_a).all() or not torch.isfinite(
        log_probabilities_b
    ).all():
        raise ValueError("R-Drop log probabilities must be finite")

    probabilities_a = log_probabilities_a.exp()
    probabilities_b = log_probabilities_b.exp()
    divergence = 0.5 * (
        probabilities_a * (log_probabilities_a - log_probabilities_b)
        + probabilities_b * (log_probabilities_b - log_probabilities_a)
    )
    per_position = divergence.sum(dim=-1).masked_fill(~mask, 0.0)
    return per_position.sum(dim=-1).mean()


def symmetric_crf_marginal_kl(
    model: CharDiacritizer,
    outputs_a: Mapping[str, torch.Tensor],
    outputs_b: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Compute symmetric KL between two differentiable CRF marginals."""

    if model.config.resolved_head_mode != "crf":
        raise ValueError("CRF-marginal R-Drop requires head_mode=crf")
    mask_a = outputs_a.get("crf_mask")
    mask_b = outputs_b.get("crf_mask")
    if mask_a is None or mask_b is None:
        raise ValueError("R-Drop CRF outputs must include crf_mask")
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    if not torch.equal(mask_a, mask_b):
        raise ValueError("R-Drop forward passes produced different CRF masks")
    return symmetric_log_probability_kl(
        model.log_probabilities(outputs_a),
        model.log_probabilities(outputs_b),
        mask_a,
    )


def symmetric_emission_kl(
    outputs_a: Mapping[str, torch.Tensor],
    outputs_b: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Compute standard R-Drop KL over normalized CRF emissions."""

    mask_a = outputs_a.get("crf_mask")
    mask_b = outputs_b.get("crf_mask")
    if mask_a is None or mask_b is None:
        raise ValueError("emission R-Drop outputs must include crf_mask")
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    if not torch.equal(mask_a, mask_b):
        raise ValueError("R-Drop forward passes produced different CRF masks")
    return symmetric_log_probability_kl(
        torch.log_softmax(outputs_a["logits"], dim=-1),
        torch.log_softmax(outputs_b["logits"], dim=-1),
        mask_a,
    )
