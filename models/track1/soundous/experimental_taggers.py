import torch
import torch.nn as nn
import torch.nn.functional as F

from models.track1.soundous.tagger import DiacritizationTagger


class MultiTaskDiacritizationTagger(DiacritizationTagger):
    """Adds an auxiliary binary "has-diacritic" head trained jointly with the main 16-class head.
    A coarser, denser auxiliary signal is a standard MLT regularizer (Caruana 1997); useful here
    given the corpus is comparatively small (~6k sentences)."""

    def __init__(self, *args, aux_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        lstm_out_dim = self.classifier.in_features
        self.aux_classifier = nn.Linear(lstm_out_dim, 2)
        self.aux_weight = aux_weight

    def forward_multitask(self, char_ids, mask, lengths, labels, no_diac_idx=0, label_smoothing=0.05):
        emb = self.embedding(char_ids)
        feats = torch.cat([emb, self.cnn(emb, mask)], dim=-1) if self.use_cnn else emb
        hidden = self.bilstm(feats, lengths)
        emissions = self.classifier(hidden)
        aux_logits = self.aux_classifier(hidden)

        main_loss = self.compute_loss(emissions, labels, mask, label_smoothing=label_smoothing)
        aux_targets = (labels != no_diac_idx).long()
        aux_loss = F.cross_entropy(aux_logits.reshape(-1, 2), aux_targets.reshape(-1), reduction="none")
        mask_flat = mask.reshape(-1).float()
        aux_loss = (aux_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

        total_loss = main_loss + self.aux_weight * aux_loss
        return total_loss, emissions, aux_loss.item()


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        key_padding_mask = ~mask
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + self.dropout(attn_out))


class AttnDiacritizationTagger(DiacritizationTagger):
    """One multi-head self-attention block between the BiLSTM and the classifier/CRF, so every
    position can attend directly to every other position in one hop -- recovers long-range
    dependencies the recurrence alone under-weights on long sentences."""

    def __init__(self, *args, num_heads=4, **kwargs):
        super().__init__(*args, **kwargs)
        lstm_out_dim = self.classifier.in_features
        self.attn = SelfAttentionBlock(lstm_out_dim, num_heads=num_heads, dropout=0.2)

    def _encode(self, char_ids, mask, lengths):
        emb = self.embedding(char_ids)
        feats = torch.cat([emb, self.cnn(emb, mask)], dim=-1) if self.use_cnn else emb
        hidden = self.bilstm(feats, lengths)
        hidden = self.attn(hidden, mask)
        return self.classifier(hidden)
