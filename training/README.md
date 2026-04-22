# training/

## Purpose
This folder will contain training pipeline entry points and orchestration logic.

## Expected Contents
- Pretraining workflow components (where applicable).
- Fine-tuning workflow components on Algerian dialect data.
- Shared utilities for optimization, scheduling, checkpointing, and reproducibility.

## Naming Convention
When files are added, use stage- and role-based naming, for example:
- pretrain_*
- finetune_*
- train_*

## Important Notes
- No training scripts are included at initialization time.
- Training modules should support all experiment strategies (A, B, C, D) through configs.
