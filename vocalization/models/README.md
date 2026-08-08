# models/

## Purpose
This folder will host model architecture implementations and wrappers used in diacritization experiments.

## Expected Contents
Planned model families include:
- BiLSTM-CRF
- Transformer Encoder
- AraBERT
- MARBERT
- DziriBERT
- AraELECTRA

## Naming Convention
Use architecture-first, lowercase module naming when implementation starts, such as:
- bilstm_crf
- transformer_encoder
- arabert
- marbert
- dziribert
- araelectra

## Important Notes
- Do not store pretrained weights or checkpoints here.
- Keep code modular so the same training/evaluation pipeline can compare all architectures.
