"""Building blocks shared by every Track 1 architecture.

chars --> [CharEmbedding] --> [CharCNNHighway]? --> [BiLSTMEncoder] --> Linear --> [CRF]? --> labels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        return self.dropout(self.embedding(char_ids))


class HighwayLayer(nn.Module):
    """y = g*H(x) + (1-g)*x -- lets the network skip the CNN transform per position/dim when the
    raw character embedding is already what's needed (Srivastava et al. 2015)."""

    def __init__(self, size, num_layers=1, activation=F.relu):
        super().__init__()
        self.transform = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.activation = activation
        for g in self.gate:
            nn.init.constant_(g.bias, -1.0)  # bias toward "carry" at init -> stabler early training

    def forward(self, x):
        for t_layer, g_layer in zip(self.transform, self.gate):
            t = self.activation(t_layer(x))
            g = torch.sigmoid(g_layer(x))
            x = g * t + (1 - g) * x
        return x


class CharCNNHighway(nn.Module):
    """Multi-kernel 1D conv over characters (local n-gram patterns: root triples, article
    assimilation) + Highway gate. Same idea as Kim et al. 2016's character-aware LM, applied one
    level down (raw characters, not characters-within-a-word)."""

    def __init__(self, emb_dim, out_dim, kernel_sizes=(2, 3, 4, 5), num_filters=64,
                 highway_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        concat_dim = num_filters * len(kernel_sizes)
        self.highway = HighwayLayer(concat_dim, num_layers=highway_layers)
        self.proj = nn.Linear(concat_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_t = x.transpose(1, 2)
        feats = [F.relu(conv(x_t))[:, :, :x_t.size(2)] for conv in self.convs]
        cat = torch.cat(feats, dim=1).transpose(1, 2)
        cat = cat * mask.unsqueeze(-1)
        h = self.highway(cat)
        return self.dropout(self.proj(h))


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        return self.dropout(self.layer_norm(out))


class CRF(nn.Module):
    """Linear-chain CRF: forward algorithm (training NLL) + batched Viterbi decoding (inference)."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.01)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)

    def _score_sentence(self, emissions, tags, mask):
        B, T, C = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(B), 0, tags[:, 0]]
        for t in range(1, T):
            emit = emissions[torch.arange(B), t, tags[:, t]]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit + trans) * mask[:, t]
        last_idx = mask.sum(1).long() - 1
        last_tags = tags[torch.arange(B), last_idx]
        return score + self.end_transitions[last_tags]

    def _forward_alg(self, emissions, mask):
        B, T, C = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = new_alpha * m + alpha * (1 - m)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        mask = mask.float()
        gold = self._score_sentence(emissions, tags, mask)
        log_z = self._forward_alg(emissions, mask)
        return (log_z - gold).mean()

    def decode(self, emissions, mask):
        B, T, C = emissions.shape
        mask_bool = mask.bool()
        backpointers = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            broadcast_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_prev = broadcast_score.max(dim=1)
            new_score = best_score + emissions[:, t]
            m = mask_bool[:, t].unsqueeze(1)
            score = torch.where(m, new_score, score)
            backpointers.append(best_prev)
        score = score + self.end_transitions.unsqueeze(0)
        _, best_last_tag = score.max(dim=1)
        lengths = mask_bool.sum(1)
        all_paths = []
        for b in range(B):
            L = lengths[b].item()
            tag = best_last_tag[b].item()
            path = [tag]
            for t in range(L - 2, -1, -1):
                tag = backpointers[t][b, tag].item()
                path.append(tag)
            path.reverse()
            all_paths.append(path)
        return all_paths
