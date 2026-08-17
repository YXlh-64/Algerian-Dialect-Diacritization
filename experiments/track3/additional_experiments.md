# Additional Experiments

Model tested in `arabic-new-notebook-last_best.ipynb`
(`track3 / bilstm_crf_factorized_head / strategy_a`).

## Architecture

Pretrained Arabic transformer backbone (`CAMeL-Lab/bert-base-arabic-camelbert-mix`)
-> softmax-weighted mix of its last 4 hidden layers (`LayerPool`, ELMo-style
layer mixing) -> each subword's pooled representation is broadcast to every
character it covers (`char2tok` gather, not just the first sub-token) ->
concatenated with a learned per-character embedding and a projected
`is_word_final` feature -> fused through a LayerNorm + Linear + GELU
projection -> 2-layer bidirectional LSTM -> linear classifier over 17
classes (16 diacritics + a space pseudo-class) -> decoded with a
linear-chain CRF (Viterbi).

Three loss terms are summed during training:
1. **CRF sequence loss** (main loss) over the 17-class labels.
2. **Auxiliary weighted cross-entropy** over the same per-position labels,
   with inverse-frequency class weights so rare diacritic classes aren't
   drowned out (`aux_loss_weight`).
3. **Factorized vowel/shadda auxiliary loss** (new relative to a plain
   BiLSTM-CRF head): two small extra heads, `vowel_head` and `shadda_head`,
   split each diacritic class into "which of the 8 vowel marks" x "Shadda
   present or not," each with its own cross-entropy term
   (`vowel_shadda_loss_weight`).

Space characters and word-final positions are modeled explicitly rather
than left implicit: spaces get their own internal class (`SPACE_LABEL`),
and every character carries a binary `is_word_final` feature projected and
concatenated in before the BiLSTM.

## Config

```yaml
backbone: CAMeL-Lab/bert-base-arabic-camelbert-mix
num_diacritic_classes: 16
space_label: 16
num_labels: 17
char_emb_dim: 64
n_pool_layers: 4
hidden_dim: 384
num_lstm_layers: 2
dropout: 0.3
use_crf: true
aux_loss_weight: 0.3
vowel_shadda_loss_weight: 0.2
freeze_embeddings: true
freeze_n_layers: 0
batch_size: 64
eval_batch_size: 32
epochs: 12
patience: 3
backbone_lr: 2.0e-05
head_lr: 0.001
weight_decay: 0.01
warmup_ratio: 0.06
grad_clip: 1.0
fp16: true
seed: 42
```

## Results

| Model | Public | Private |
|---|---|---|
| camelbert_mix_09421 | 0.94646 | 0.93977 |

Dev-set metrics (best checkpoint, epoch 7/12):

| Metric | Value |
|---|---|
| Micro-F1 / Accuracy | 0.9421 |
| DER | 0.0579 |
| DER* | 0.0698 |
| CER | 0.0313 |
| WER | 0.1869 |
| WER* | 0.1697 |