# utils/

## Purpose
This folder will hold shared utility modules used across training, evaluation, and experiments.

## Expected Contents
Planned utility areas include:
- Arabic text normalization and helper functions.
- Diacritic-aware text preprocessing helpers.
- W&B logging and run metadata utilities.

## Naming Convention
Use utility-domain names for future modules, for example:
- text_*
- arabic_*
- wandb_*

## Important Notes
- Keep utilities small, reusable, and independent of specific model implementations.
- Avoid circular dependencies with model/training modules.
