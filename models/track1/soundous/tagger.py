"""The base tagger. use_cnn/use_crf switches implement all 3 Track 1 architectures from one class:

  use_cnn=False, use_crf=False -> BiLSTM only         
  use_cnn=True,  use_crf=False -> BiLSTM-CNN
  use_cnn=False, use_crf=True  -> BiLSTM-CRF
  use_cnn=True,  use_crf=True  -> BiLSTM-CNN-CRF       
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.track1.soundous.layers import CharEmbedding, CharCNNHighway, BiLSTMEncoder, CRF

ARCHITECTURES = {
    "bilstm_cnn": dict(use_cnn=True, use_crf=False),
    "bilstm_crf": dict(use_cnn=False, use_crf=True),
    "bilstm_cnn_crf": dict(use_cnn=True, use_crf=True),
}


class DiacritizationTagger(nn.Module):
    def __init__(self, vocab_size, num_classes, pad_idx,
                 emb_dim=128, cnn_out_dim=128, lstm_hidden=256, lstm_layers=2,
                 use_cnn=True, use_crf=True, dropout=0.3, cnn_kernel_sizes=(2, 3, 4, 5),
                 cnn_num_filters=64, highway_layers=2):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_crf = use_crf

        self.embedding = CharEmbedding(vocab_size, emb_dim, pad_idx, dropout=dropout * 0.5)

        if use_cnn:
            self.cnn = CharCNNHighway(emb_dim, cnn_out_dim, kernel_sizes=cnn_kernel_sizes,
                                       num_filters=cnn_num_filters, highway_layers=highway_layers,
                                       dropout=dropout)
            lstm_input_dim = emb_dim + cnn_out_dim
        else:
            self.cnn = None
            lstm_input_dim = emb_dim

        self.bilstm = BiLSTMEncoder(lstm_input_dim, lstm_hidden, num_layers=lstm_layers, dropout=dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
        self.crf = CRF(num_classes) if use_crf else None

    def _encode(self, char_ids, mask, lengths):
        emb = self.embedding(char_ids)
        if self.use_cnn:
            feats = torch.cat([emb, self.cnn(emb, mask)], dim=-1)
        else:
            feats = emb
        hidden = self.bilstm(feats, lengths)
        return self.classifier(hidden)

    def forward(self, char_ids, mask, lengths, labels=None):
        emissions = self._encode(char_ids, mask, lengths)
        if labels is not None:
            return self.compute_loss(emissions, labels, mask), emissions
        return emissions

    def compute_loss(self, emissions, labels, mask, label_smoothing=0.0):
        if self.use_crf:
            return self.crf.neg_log_likelihood(emissions, labels, mask)
        B, T, C = emissions.shape
        loss = F.cross_entropy(emissions.reshape(-1, C), labels.reshape(-1),
                                reduction="none", label_smoothing=label_smoothing)
        mask_flat = mask.reshape(-1).float()
        return (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

    def decode(self, char_ids, mask, lengths):
        emissions = self._encode(char_ids, mask, lengths)
        if self.use_crf:
            return self.crf.decode(emissions, mask)
        preds = emissions.argmax(-1)
        L = lengths.tolist()
        return [preds[i, :L[i]].tolist() for i in range(preds.size(0))]


def build_model(arch_name, vocab_size, num_classes, pad_idx, device, **overrides):
    cfg = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx)
    cfg.update(ARCHITECTURES[arch_name])
    cfg.update(overrides)
    model = DiacritizationTagger(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{arch_name}] {n_params:,} trainable params")
    return model
