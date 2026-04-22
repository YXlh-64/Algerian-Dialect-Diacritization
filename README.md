# Algerian Dialect Diacritization — AISI Research Project

![Python Version](https://img.shields.io/badge/python-3.x-blue)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracking-orange)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-informational)

## Overview
Diacritization is the task of restoring missing short vowels and other diacritic marks in Arabic text, which is critical for improving readability, pronunciation support, and downstream Arabic NLP tasks such as text-to-speech, speech recognition, and linguistic analysis. Algerian dialect Arabic (Derja) is particularly challenging because of its high orthographic variation, code-switching tendencies, limited standardized writing conventions, and scarcity of publicly available diacritized resources. This project targets two core contributions: (1) developing the first structured Algerian dialect diacritized dataset for research use, and (2) establishing transfer learning baselines across multiple pretraining regimes and model families.

## Experimental Strategies

| Strategy | Pretraining Data | Fine-tuning Data | Research Question |
|----------|------------------|------------------|-------------------|
| A | None | Algerian only | Low-resource baseline — how hard is the task? |
| B | MSA (Tashkeela ~50k) | Algerian | Does standard Arabic help? |
| C | NADI (~15k) | Algerian | Does multi-dialect Arabic help? |
| D | MSA + NADI (~65k) | Algerian | Does combined data help? |

## Models

| Model | Type | HuggingFace ID |
|-------|------|----------------|
| BiLSTM-CRF | Custom deep learning | N/A |
| Transformer Encoder | Custom deep learning | N/A |
| AraBERT | Pretrained LM | aubmindlab/bert-base-arabertv2 |
| MARBERT | Pretrained LM | UBC-NLP/MARBERTv2 |
| DziriBERT | Pretrained LM | alger-ia/dziribert |
| AraELECTRA | Pretrained LM | aubmindlab/araelectra-base-discriminator |

## Evaluation Metrics
- **CER**: Character Error Rate measuring normalized character-level edit distance.
- **WER**: Word Error Rate measuring normalized word-level edit distance.
- **DER**: Diacritic Error Rate on all diacritizable character positions.
- **DER\***: Diacritic Error Rate excluding word-final diacritic positions.
- **WER\***: Word Error Rate excluding word-final diacritic effects.
- **Accuracy**: Exact diacritic match accuracy per character position.

## Repository Structure

```text
algerian-diacritization/
├── README.md                # Main project description, research scope, and workflow
├── .gitignore               # Rules to exclude data, models, secrets, and artifacts
├── configs/                 # Planned experiment configuration files (one per strategy)
├── models/                  # Planned model implementations and architecture modules
├── training/                # Planned training entry points and training orchestration
├── evaluation/              # Planned evaluation scripts and metric computations
├── experiments/             # Strategy-specific experiment entry points and run layouts
├── utils/                   # Planned utility helpers (text processing, logging, etc.)
└── tests/                   # Planned unit/integration/regression test suites
```

## Tooling and Workflow
This project uses a four-tool workflow:
- **GitHub**: source-code-only repository management and version control.
- **Google Drive**: dataset storage, controlled sharing, and data access coordination.
- **W&B (Weights & Biases)**: experiment tracking, metric visualization, and run comparison.
- **Discord**: team communication, supervision updates, and experiment coordination.

## Important Note:
The data should be accessed externally from Google Drive API only, don't keep track of the dataset in this repository

## Installation (Placeholder)
Environment setup instructions and dependency definitions (including requirements.txt) will be added once implementation begins.

## Usage (Placeholder)
Command-line usage instructions for running each strategy and model will be added once training and evaluation scripts are implemented.

## Team and Acknowledgments
- **Institution**: ENSIA Research Team
- **Supervisors**: Dr. Mohamed Brahimi, Manel Ait Said, Dr. Mohamed Hadj Ameur, Dr. ElMoatez Billah Nagoudi
- **Contributors**: Aya Benali Khodja, Mazouz Ahmed Thabet, Guergour Youcef, Zyad Kherraf, Hadil Hattabi, Omar Ziyad Chaalel, Khadidja Bahfir, Selma Khelili, Younes Barmaki, Roumaissa Smail, Aya Benmansour, Souha Nour Abidat, Ines Bencherif, Basmala Randa Benmaiche, Lyes Hadjar, Yousra Kassous, Zahra Abdeli, Fatma Imene Djelili, Takoua Hidoussi, Soundous Chemam
