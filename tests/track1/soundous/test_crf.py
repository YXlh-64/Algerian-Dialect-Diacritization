"""Sanity checks for the from-scratch linear-chain CRF: decode returns valid-length, in-range
label sequences, and the loss is a finite, non-negative scalar.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from models.track1.soundous.layers import CRF


def test_decode_returns_correct_lengths_and_valid_labels():
    torch.manual_seed(0)
    num_tags = 5
    crf = CRF(num_tags)
    B, T = 3, 7
    emissions = torch.randn(B, T, num_tags)
    lengths = [7, 4, 1]
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True

    paths = crf.decode(emissions, mask)
    assert len(paths) == B
    for path, L in zip(paths, lengths):
        assert len(path) == L
        assert all(0 <= t < num_tags for t in path)


def test_nll_is_finite_and_nonnegative():
    torch.manual_seed(0)
    num_tags = 4
    crf = CRF(num_tags)
    B, T = 2, 5
    emissions = torch.randn(B, T, num_tags)
    tags = torch.randint(0, num_tags, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)

    nll = crf.neg_log_likelihood(emissions, tags, mask)
    assert torch.isfinite(nll)
    assert nll.item() >= -1e-4  # NLL of the true path relative to the partition function; ~0 at worst
