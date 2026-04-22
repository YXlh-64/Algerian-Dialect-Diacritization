# evaluation/

## Purpose
This folder will include evaluation logic and metric computation for model outputs.

## Expected Contents
Metrics to be implemented and reported:
- CER: Character Error Rate.
- WER: Word Error Rate.
- DER: Diacritic Error Rate over all diacritizable positions.
- DER*: Diacritic Error Rate excluding word-final diacritics.
- WER*: Word Error Rate excluding word-final diacritic influence.
- Accuracy: Exact diacritic match accuracy at character level.

## Naming Convention
Use metric- and scope-driven names for future files, such as:
- metrics_*
- evaluate_*
- report_*

## Important Notes
- Keep metric definitions consistent across experiments to preserve comparability.
- Document any normalization or tokenization rules used before scoring.
