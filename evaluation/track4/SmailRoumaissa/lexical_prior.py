import json
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import torch

from utils.track4.SmailRoumaissa.constants import SPACE


def _words_with_labels(chars: List[str], labels: List[int]) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Split a sentence into (word, [(position_in_word, label), ...])."""
    words, cur_chars, cur_labels = [], [], []
    for c, l in zip(chars, labels):
        if c == SPACE:
            if cur_chars:
                words.append(("".join(cur_chars), list(enumerate(cur_labels))))
            cur_chars, cur_labels = [], []
        else:
            cur_chars.append(c)
            cur_labels.append(l)
    if cur_chars:
        words.append(("".join(cur_chars), list(enumerate(cur_labels))))
    return words


class LexicalPrior:
    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing
        self.counts: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))

    def fit(self, train_jsonl_path: str) -> "LexicalPrior":
        with open(train_jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for word, positions in _words_with_labels(rec["chars"], rec["labels"]):
                    for pos, label in positions:
                        self.counts[word][pos][label] += 1
        return self

    def known(self, word: str) -> bool:
        return word in self.counts

    def distribution(self, word: str, pos: int, num_classes: int = 16) -> Optional[torch.Tensor]:
        counter = self.counts.get(word, {}).get(pos)
        if not counter:
            return None
        total = sum(counter.values()) + self.smoothing * num_classes
        probs = torch.full((num_classes,), self.smoothing / total)
        for label, c in counter.items():
            probs[label] = (c + self.smoothing) / total
        return probs


def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def fuse_sentence(
    neural_log_probs: torch.Tensor,
    chars: List[str],
    lexical: LexicalPrior,
    entropy_threshold: float = 1.0,
    gate_temperature: float = 0.3,
    max_strength: float = 2.0,
    num_classes: int = 16,
) -> torch.Tensor:
    """Confidence-gated fusion:
        weight(t) = max_strength * sigmoid((entropy(t) - entropy_threshold) / gate_temperature)
    weight -> 0 when the Transformer is confident (low entropy);
    weight -> max_strength only once entropy climbs past the threshold.
    `entropy_threshold` and `gate_temperature` should be swept on dev.
    """
    out = neural_log_probs.clone()
    ent = entropy(neural_log_probs)

    word_buf, word_start = [], 0
    for i, c in enumerate(chars + [SPACE]):
        if c == SPACE:
            word = "".join(word_buf)
            if word and lexical.known(word):
                for j in range(len(word_buf)):
                    abs_i = word_start + j
                    prior = lexical.distribution(word, j, num_classes)
                    if prior is None:
                        continue
                    prior = prior.to(neural_log_probs.device)
                    w = max_strength * torch.sigmoid((ent[abs_i] - entropy_threshold) / gate_temperature)
                    lexical_logp = torch.log(prior.clamp_min(1e-9))
                    combined = neural_log_probs[abs_i] + w * lexical_logp
                    out[abs_i] = combined - torch.logsumexp(combined, dim=-1)
            word_buf = []
            word_start = i + 1
        else:
            word_buf.append(c)
    return out
