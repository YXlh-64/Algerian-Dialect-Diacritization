# Paper Metric Definitions

These definitions are frozen for every model and system. The machine-readable
copy is embedded in every `*_metrics.json` artifact.

## Evaluation unit

- Split: the released 607-sentence dev split.
- Scored letters: 15,897 non-space Arabic letters.
- Spaces remain in the reconstructed text but are excluded from label metrics.
- The fixed label inventory contains all 16 competition classes.

## Metrics

### Accuracy

The number of letters whose predicted 16-class label exactly matches the
reference, divided by all scored letters. This is also official Micro-F1.

### Macro and per-class F1

For every class:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

An undefined value is reported as zero. `macro_f1` is the unweighted mean over
all fixed 16 classes, including classes absent from dev. The artifacts also
provide `macro_f1_present_classes`, which averages only classes with positive
reference support.

### WER

Each sentence is reconstructed as fully vocalized Unicode text and split on
whitespace. Corpus word-level Levenshtein edits are pooled and divided by the
total number of reference words:

```text
WER = (substitutions + deletions + insertions) / reference words
```

### CER

Corpus Levenshtein edits are computed over NFC Unicode codepoints of the fully
vocalized sentence, including Arabic letters, combining diacritics, and
spaces. The pooled edit count is divided by all reference codepoints.

### Word accuracy

A whitespace-delimited word is correct only when every aligned character label
in the word is correct. Word accuracy is exact correct words divided by all
words.

### Sentence accuracy

A sentence is correct only when every non-space character label is correct.

### Shadda accuracy

The 16 labels are collapsed into binary Shadda presence:

```text
absent: labels 0-7
present: labels 8-15
```

Accuracy, precision, recall, F1, and the complete binary counts are reported.

### Tanween accuracy

The 16 labels are collapsed into binary Tanween presence. The positive labels
are Fathatan, Dammatan, and Kasratan, with or without Shadda:

```text
positive IDs: 2, 4, 6, 10, 12, 14
```

Accuracy, precision, recall, F1, and the complete binary counts are reported.

### Confusion matrix

A 16×16 integer matrix is reported with reference labels as rows and predicted
labels as columns.

### Skeleton mismatch count

Competition diacritics are stripped from each predicted sentence. The count is
the number of sentences whose whitespace-normalized predicted skeleton differs
from the released undiacritized input. Models generated through the aligned
label pipeline must report zero.

### char-BLEU

Corpus BLEU-4 is computed over NFC Unicode codepoints after removing
whitespace. It uses modified n-gram precision, corpus brevity penalty, no
unigram smoothing, add-one smoothing for orders 2–4, and effective order.

## Implementation

Authoritative code:

```text
track4/paper_metrics.py
```

Generic prediction-file evaluation:

```bash
python -m evaluation.track4.Lyes.paper_metrics \
  --predictions path/to/dev_predictions.jsonl \
  --output path/to/paper_metrics.json
```

Evaluate all registered models:

```bash
python -m evaluation.track4.Lyes.paper_model_report \
  --device mps \
  --num-workers 0
```
