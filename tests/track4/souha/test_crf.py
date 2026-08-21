"""Brute-force validation of the hand-written CRF (notebook §6).

The forward algorithm and the Viterbi decode are checked against explicit
enumeration of all K**T label sequences. K and T are kept tiny so the
enumeration is exhaustive rather than sampled.
"""

import itertools

import torch

from models.track4.souha.crf import CRF, is_intra_mask


def _brute_force(crf, em, is_intra, b, K, T):
    "Enumerate every label sequence; return (logZ, best_sequence)."
    scores, best = [], (-1e30, None)
    for seq in itertools.product(range(K), repeat=T):
        s = crf.start[seq[0]] + em[b, 0, seq[0]]
        for t in range(1, T):
            tr = crf.intra if is_intra[b, t] else crf.inter
            s = s + tr[seq[t - 1], seq[t]] + em[b, t, seq[t]]
        s = s + crf.end[seq[-1]]
        scores.append(s)
        if s > best[0]:
            best = (float(s), list(seq))
    return float(torch.logsumexp(torch.stack(scores), 0)), best[1]


def test_crf_split_transitions():
    "Notebook §6, verbatim: split intra-/inter-word transitions, no padding."
    torch.manual_seed(0)
    K, T, B = 3, 4, 2
    crf = CRF(K, split=True)
    em = torch.randn(B, T, K)
    mask = torch.ones(B, T, dtype=torch.bool)
    wid = torch.tensor([[0, 0, -1, 1], [0, 1, 1, 1]])
    ii = is_intra_mask(wid)

    decoded = crf.decode(em, mask, ii)
    for b in range(B):
        lz_brute, seq_brute = _brute_force(crf, em, ii, b, K, T)
        lz_crf = float(crf._logZ(em, mask, ii)[b])
        assert abs(lz_brute - lz_crf) < 1e-4, f"logZ b{b}: {lz_brute} vs {lz_crf}"
        assert decoded[b].tolist() == seq_brute, f"viterbi b{b}"


def test_crf_shared_transitions():
    "split=False: one transition matrix everywhere (the `split_crf` ablation)."
    torch.manual_seed(1)
    K, T, B = 3, 4, 2
    crf = CRF(K, split=False)
    crf.inter = crf.intra          # _brute_force reads both; shared here
    em = torch.randn(B, T, K)
    mask = torch.ones(B, T, dtype=torch.bool)
    wid = torch.tensor([[0, 0, -1, 1], [0, 1, 1, 1]])
    ii = is_intra_mask(wid)

    decoded = crf.decode(em, mask, ii)
    for b in range(B):
        lz_brute, seq_brute = _brute_force(crf, em, ii, b, K, T)
        lz_crf = float(crf._logZ(em, mask, ii)[b])
        assert abs(lz_brute - lz_crf) < 1e-4, f"logZ b{b}: {lz_brute} vs {lz_crf}"
        assert decoded[b].tolist() == seq_brute, f"viterbi b{b}"


def test_crf_ignores_padding():
    """A padded batch must score its real prefix exactly like an unpadded one.

    Every training batch is padded, so this path runs constantly even though
    the notebook's §6 check only covered full masks.
    """
    torch.manual_seed(2)
    K, T, L = 3, 5, 3            # row 0 is full length T, row 1 is length L
    crf = CRF(K, split=True)
    em = torch.randn(2, T, K)
    mask = torch.ones(2, T, dtype=torch.bool)
    mask[1, L:] = False
    wid = torch.tensor([[0, 0, -1, 1, 1], [0, 0, 1, -2, -2]])
    ii = is_intra_mask(wid)

    # the same short sequence, on its own, with no padding at all
    em_s = em[1:2, :L].clone()
    mask_s = torch.ones(1, L, dtype=torch.bool)
    ii_s = is_intra_mask(wid[1:2, :L])

    lz_pad = float(crf._logZ(em, mask, ii)[1])
    lz_solo = float(crf._logZ(em_s, mask_s, ii_s)[0])
    assert abs(lz_pad - lz_solo) < 1e-4, f"logZ with padding {lz_pad} vs alone {lz_solo}"

    dec_pad = crf.decode(em, mask, ii)[1, :L].tolist()
    dec_solo = crf.decode(em_s, mask_s, ii_s)[0].tolist()
    assert dec_pad == dec_solo, f"viterbi with padding {dec_pad} vs alone {dec_solo}"


if __name__ == "__main__":
    test_crf_split_transitions(); print("  [PASS] split transitions vs brute force")
    test_crf_shared_transitions(); print("  [PASS] shared transitions vs brute force")
    test_crf_ignores_padding(); print("  [PASS] padding ignored")
