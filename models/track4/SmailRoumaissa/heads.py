import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.track4.SmailRoumaissa.constants import NUM_CLASSES
from models.track4.SmailRoumaissa.crf import ChainCRF, _word_spans


class DecomposedHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.base_head = nn.Linear(dim, 8)
        self.shadda_head = nn.Linear(dim, 2)

    def raw_head_logits(self, h: torch.Tensor):
        """Un-combined logits for each sub-head, before the joint softmax.
        Needed separately so calibration can fit an independent temperature
        per head instead of one shared temperature over all 16 classes."""
        return self.base_head(h), self.shadda_head(h)

    def raw_logits(self, h: torch.Tensor) -> torch.Tensor:
        base, shadda = self.raw_head_logits(h)
        return (base.unsqueeze(-1) + shadda.unsqueeze(-2)).reshape(*base.shape[:-1], NUM_CLASSES)

    def forward(self, h: torch.Tensor, base_temperature: float = 1.0,
                shadda_temperature: float = 1.0) -> torch.Tensor:
        """Calibrated log-probabilities, with an INDEPENDENT temperature for
        the base head and the shadda head. They combine additively before the
        joint 16-way softmax, so scaling each sub-head's logits separately
        lets each head's confidence be calibrated on its own -- the entropy gate
        depends on this being accurate, and the two heads likely have different 
        confidence profiles. Leave both at 1.0 during training / uncalibrated use."""
        base, shadda = self.raw_head_logits(h)
        base = base / base_temperature
        shadda = shadda / shadda_temperature
        combined = (base.unsqueeze(-1) + shadda.unsqueeze(-2)).reshape(*base.shape[:-1], NUM_CLASSES)
        return F.log_softmax(combined, dim=-1)


class PerWordCRFHead(nn.Module):
    """Runs an independent CRF chain over each word's characters (spaces /
    BOS / EOS are boundaries). Diacritic dependencies -- especially
    word-final case endings -- are overwhelmingly intra-word, so scoping the
    CRF this way is both cheaper and a better structural fit than one chain
    spanning the whole sentence and having to "jump over" unlabeled spaces.
    """
    def __init__(self, dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.emit = DecomposedHead(dim)
        self.crf = ChainCRF(num_classes)

    def _gather(self, hidden: torch.Tensor, is_letter: torch.Tensor, tags: torch.Tensor = None):
        B, T, C = hidden.shape
        words_h, words_tags, meta = [], [], []
        for b in range(B):
            for (s, e) in _word_spans(is_letter[b]):
                words_h.append(hidden[b, s:e])
                if tags is not None:
                    words_tags.append(tags[b, s:e])
                meta.append((b, s, e))
        if not words_h:
            return None, None, None, meta
        maxlen = max(w.size(0) for w in words_h)
        Hpad = hidden.new_zeros(len(words_h), maxlen, C)
        mask = torch.zeros(len(words_h), maxlen, dtype=torch.bool, device=hidden.device)
        Tpad = torch.zeros(len(words_h), maxlen, dtype=torch.long, device=hidden.device) if tags is not None else None
        for i, w in enumerate(words_h):
            L = w.size(0)
            Hpad[i, :L] = w
            mask[i, :L] = True
            if tags is not None:
                Tpad[i, :L] = words_tags[i]
        return Hpad, mask, Tpad, meta

    def loss(self, hidden: torch.Tensor, is_letter: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        Hpad, mask, Tpad, _ = self._gather(hidden, is_letter, tags)
        if Hpad is None:
            return hidden.sum() * 0.0
        emissions = self.emit.raw_logits(Hpad)
        return self.crf.neg_log_likelihood(emissions, Tpad, mask)

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor, is_letter: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden.shape
        Hpad, mask, _, meta = self._gather(hidden, is_letter)
        preds = torch.full((B, T), -1, dtype=torch.long, device=hidden.device)
        if Hpad is None:
            return preds
        emissions = self.emit.raw_logits(Hpad)
        paths = self.crf.decode(emissions, mask)
        for i, (b, s, e) in enumerate(meta):
            preds[b, s:e] = paths[i, :e - s]
        return preds

    def marginal_log_probs(self, hidden: torch.Tensor, base_temperature: float = 1.0,
                            shadda_temperature: float = 1.0) -> torch.Tensor:
        """Per-character log-probs ignoring transition structure -- used only
        as the confidence signal for entropy-gated lexical fusion, applied
        to the emissions before the Viterbi pass."""
        return self.emit(hidden, base_temperature, shadda_temperature)

    def _gather_generic(self, tensor: torch.Tensor, is_letter: torch.Tensor):
        B, T, C = tensor.shape
        words, meta = [], []
        for b in range(B):
            for (s, e) in _word_spans(is_letter[b]):
                words.append(tensor[b, s:e])
                meta.append((b, s, e))
        if not words:
            return None, None, meta
        maxlen = max(w.size(0) for w in words)
        Wpad = tensor.new_zeros(len(words), maxlen, C)
        mask = torch.zeros(len(words), maxlen, dtype=torch.bool, device=tensor.device)
        for i, w in enumerate(words):
            L = w.size(0)
            Wpad[i, :L] = w
            mask[i, :L] = True
        return Wpad, mask, meta

    @torch.no_grad()
    def decode_from_emissions(self, emissions: torch.Tensor, is_letter: torch.Tensor) -> torch.Tensor:
        """Decode using externally-provided per-character scores (e.g. after
        lexical-prior fusion) instead of recomputing emissions from hidden
        states. `emissions` just needs to be a monotonic per-class score,
        e.g. calibrated log-probs already fused with the lexical prior."""
        B, T, _ = emissions.shape
        Wpad, mask, meta = self._gather_generic(emissions, is_letter)
        preds = torch.full((B, T), -1, dtype=torch.long, device=emissions.device)
        if Wpad is None:
            return preds
        paths = self.crf.decode(Wpad, mask)
        for i, (b, s, e) in enumerate(meta):
            preds[b, s:e] = paths[i, :e - s]
        return preds
