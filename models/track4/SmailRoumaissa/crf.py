import torch
import torch.nn as nn

from utils.track4.SmailRoumaissa.constants import NUM_CLASSES


class ChainCRF(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.transitions = nn.Parameter(torch.zeros(num_classes, num_classes))
        self.start = nn.Parameter(torch.zeros(num_classes))
        self.end = nn.Parameter(torch.zeros(num_classes))

    def _log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T = emissions.size(1)
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T = emissions.size(1)
        score = self.start.gather(0, tags[:, 0]) + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, T):
            emit = emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + torch.where(mask[:, t], emit + trans, torch.zeros_like(score))
        last_idx = (mask.sum(1) - 1).clamp_min(0)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        return score + self.end.gather(0, last_tags)

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (self._log_partition(emissions, mask) - self._gold_score(emissions, tags, mask)).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = emissions.shape
        backpointers = []
        score = self.start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_idx = broadcast.max(dim=1)
            best_score = best_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            backpointers.append(best_idx)
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(dim=1)
        seq_len = mask.sum(1)
        paths = torch.zeros(B, T, dtype=torch.long, device=emissions.device)
        for b in range(B):
            L = int(seq_len[b].item())
            if L == 0:
                continue
            tag = int(best_last[b].item())
            path = [tag]
            for t in range(L - 1, 0, -1):
                tag = int(backpointers[t - 1][b, tag].item())
                path.append(tag)
            path.reverse()
            paths[b, :L] = torch.tensor(path, device=emissions.device)
        return paths


def _word_spans(is_letter_row: torch.Tensor):
    """Contiguous runs of True in a 1D bool tensor -> list of (start,end)."""
    spans, start = [], None
    for t, v in enumerate(is_letter_row.tolist()):
        if v and start is None:
            start = t
        if not v and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, len(is_letter_row)))
    return spans
