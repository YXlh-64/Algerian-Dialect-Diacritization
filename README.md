# Algerian Dialect Diacritization — AISI Research Project

![Python Version](https://img.shields.io/badge/python-3.x-blue)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracking-orange)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-informational)

## Overview
Diacritization is the task of restoring missing short vowels and other diacritic marks in Arabic text, which is critical for improving readability, pronunciation support, and downstream Arabic NLP tasks such as text-to-speech, speech recognition, and linguistic analysis. Algerian dialect Arabic (Derja) is particularly challenging because of its high orthographic variation, code-switching tendencies, limited standardized writing conventions, and scarcity of publicly available diacritized resources. This project targets two core contributions: (1) developing the first structured Algerian dialect diacritized dataset for research use, and (2) establishing transfer learning baselines across multiple pretraining regimes and model families.

## Experimental Strategies
Every run in this repo also belongs to one of four transfer-learning strategies (independent of which track/architecture it uses):

| Strategy | Pretraining Data | Fine-tuning Data | Research Question |
|----------|------------------|------------------|-------------------|
| A | None | Algerian only | Low-resource baseline — how hard is the task? |
| B | MSA (Tashkeela ~50k) | Algerian | Does standard Arabic help? |
| C | NADI (~15k) | Algerian | Does multi-dialect Arabic help? |
| D | MSA + NADI (~65k) | Algerian | Does combined data help? |

**All results currently in this repo (`experiments/leaderboard.md` and every `configs/**/strategy_a_*.yaml`) are Strategy A** — no external pretraining, char/token classifier trained (or fine-tuned) directly on the Algerian data. Strategies B/C/D are on the roadmap but not yet implemented.

## The Four Tracks
The project compares four families of architecture for the same char-level diacritic-tagging task. Each track is a top-level folder under `configs/`, `models/`, `training/`, `evaluation/`, and `experiments/` (e.g. `training/track3/...`).

| Track | Name | Architectures | Status |
|---|---|---|---|
| **Track 1** | BiLSTM taggers | BiLSTM-CNN, BiLSTM-CRF, BiLSTM-CNN-CRF (our P2 baseline architecture) — standalone recurrent taggers over character embeddings, no transformer backbone | Planned |
| **Track 2** | Character-based LLMs | ByT5, CANINE, and similar tokenizer-free / char-level transformer models | Planned |
| **Track 3** | Arabic-pretrained transformers | AraBERT, CAMeLBERT, MARBERT, MARBERTv2, DziriBERT, AlcLaM, and other Arabic BERT-style variants, fine-tuned for token/char classification | **Implemented** — see below |
| **Track 4** | Transformer-based models (from scratch) | Custom, from-scratch Transformer architectures for char-based tagging: plain char-level Transformer, and Transformer-CNN-CRF variants (i.e. the transformer encoder itself is trained from scratch, not fine-tuned from a pretrained checkpoint) | Planned |

All tracks — implemented and planned — are trained following **Strategy A** first (see table above), before the B/C/D pretraining-transfer variants are attempted.

### Track 3 in detail (the only track with code so far)
Track 3 fine-tunes an off-the-shelf Arabic-pretrained transformer encoder and adds one of two **heads** on top (the head is a separate axis from the track/strategy — see the note at the top of `run_pipeline.py`):

| Head type | What it is | Best Strategy-A result (DER) |
|---|---|---|
| `linear_head` | Transformer encoder + a plain linear classification layer per character | 0.0824 (`camelbert_mix`) |
| `bilstm_crf_head` | Transformer encoder + a BiLSTM-CRF classification head per character (**do not confuse this with the standalone "BiLSTM-CRF" tagger in Track 1** — here the BiLSTM-CRF only sits on top of a pretrained transformer's output, it is not trained from raw characters) | 0.0483 (`arabert_v02`) |

Backbones currently available (see `MODEL_REGISTRY` in each `training/track3/<head_type>/finetune_<head_type>.py`):

| `--model` key | HuggingFace checkpoint | Available for |
|---|---|---|
| `arabert_v02` | `aubmindlab/bert-base-arabertv02` | both heads |
| `camelbert_da` | `CAMeL-Lab/bert-base-arabic-camelbert-da` | both heads |
| `camelbert_mix` | `CAMeL-Lab/bert-base-arabic-camelbert-mix` | both heads |
| `marbert` | `UBC-NLP/MARBERT` | both heads |
| `dziribert` | `alger-ia/dziribert` | both heads |
| `marbertv2` | `UBC-NLP/MARBERTv2` | `linear_head` only |
| `alclam` | `rahbi/alclam-base-v2` | `linear_head` only |

Full ranked results across every model × head combination: [`experiments/leaderboard.md`](experiments/leaderboard.md). Per-model config, training curves, and metric report: `experiments/track3/<head_type>/strategy_a_overview.md`.

## Evaluation Metrics
- **CER**: Character Error Rate measuring normalized character-level edit distance.
- **WER**: Word Error Rate measuring normalized word-level edit distance.
- **DER**: Diacritic Error Rate on all diacritizable character positions.
- **DER\***: Diacritic Error Rate excluding word-final diacritic positions.
- **WER\***: Word Error Rate excluding word-final diacritic effects.
- **Accuracy**: Exact diacritic match accuracy per character position.

## Repository Structure
This is the actual current layout (not a plan) — every path shown here exists in the repo today. `track3/` is the only track subfolder that exists so far; Tracks 1, 2, and 4 will each get their own `track1/`, `track2/`, `track4/` subfolder under `configs/`, `models/`, `training/`, `evaluation/`, and `experiments/` once implemented, following the exact same pattern as `track3/`.

```text
Algerian-Dialect-Diacritization-main/
├── README.md                       # This file
├── requirements.txt                # pip dependencies (torch installed separately, see below)
├── run_pipeline.py                 # single entrypoint: installs deps, fetches data, dispatches to the right training script
├── .gitignore
│
├── data/                           # Dataset (bundled in this zip; gitignored in the actual repo — see "Data" below)
│   ├── README.md                   # Full schema, label scheme, filtering settings, split sizes
│   ├── vocab.json                  # {character -> integer index}
│   ├── class_labels.txt            # 16-class diacritic label scheme
│   ├── train_data/train_Algerian-DIAC.jsonl
│   ├── dev_data/dev_Algerian-DIAC.jsonl
│   └── test_data/                  # raw_sentences_test.txt, ids, sample_submission.csv, make_submission.py
│
├── configs/                        # Per-run YAML configs, one per (track, head_type, model) combo
│   ├── README.md
│   └── track3/{linear_head,bilstm_crf_head}/strategy_a_<model>_<score>.yaml
│
├── models/                         # Model/architecture implementations
│   ├── README.md
│   └── track3/{linear_head,bilstm_crf_head}/*_model.py
│
├── training/                       # Training entry points, auto-discovered by run_pipeline.py
│   ├── README.md
│   └── track3/{linear_head,bilstm_crf_head}/finetune_<head_type>.py
│
├── evaluation/                     # Metric computation + one Markdown report per finished run
│   ├── README.md
│   └── track3/{linear_head,bilstm_crf_head}/
│       ├── evaluate_<head_type>.py
│       └── report_strategy_a_<model>_<score>.md
│
├── experiments/                    # Strategy-level summaries and the overall leaderboard
│   ├── README.md
│   ├── leaderboard.md              # All runs, ranked
│   └── track3/{linear_head,bilstm_crf_head}/strategy_a_overview.md
│
├── utils/
│   ├── README.md
│   └── fetch_data.py               # Downloads data/ from Google Drive if it isn't already present locally
│
├── tests/README.md                 # No executable tests yet
└── working/                        # Local scratch space created at run time (checkpoints/, exports/) — gitignored
```

## Tooling and Workflow
This project uses a four-tool workflow:
- **GitHub**: source-code-only repository management and version control.
- **Google Drive**: dataset storage, controlled sharing, and data access coordination.
- **W&B (Weights & Biases)**: experiment tracking, metric visualization, and run comparison.
- **Discord**: team communication, supervision updates, and experiment coordination.

## Data
In the actual GitHub repo, `data/` is **gitignored** and must be fetched separately from Google Drive (`python utils/fetch_data.py`) — see `.gitignore` and `data/README.md`. **This zip is a local snapshot that already includes a working copy of `data/`**, so if you're running from this zip you can skip the Drive fetch entirely; `run_pipeline.py` auto-detects that `./data` already looks valid (has `train_data/`, `dev_data/`, `vocab.json`, `class_labels.txt`) and won't try to re-download it.

## Installation & Setup

You need Python 3.10+ and roughly 3 GB free disk space (mostly for torch + transformers + downloaded checkpoints). `run_pipeline.py` handles installing `torch` (CPU or CUDA build, auto-detected) and everything in `requirements.txt` for you on first run — you normally don't need to `pip install` anything by hand.

### Windows (PowerShell)
Windows has a ~260-character path limit by default, and a virtualenv nested deep inside `Downloads\...\Algerian-Dialect-Diacritization-main\` combined with `transformers`'/`torch`'s own deeply-nested package paths can exceed it. The fix is to put the venv itself at a short path (e.g. `C:\venv`), **outside** the project folder, and just `cd` into the project to run commands:

```powershell
# 1. Create a venv at a short path (do this once)
python -m venv C:\venv

# 2. Activate it (do this every time you open a new terminal)
C:\venv\Scripts\Activate.ps1

# 3. Move into your project and run as usual
cd "Folder directory Name"
python run_pipeline.py --track track3 --head-type linear_head --model marbert
```

If PowerShell blocks `Activate.ps1` with an execution-policy error, run this once (in the same window, no admin needed) and then retry step 2:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux / macOS (bash/zsh)
No short-path workaround needed — a venv inside the project folder is fine:
```bash
# 1. Create and activate a venv (do this once per machine, activate every new terminal)
python3 -m venv .venv
source .venv/bin/activate

# 2. Run
cd Algerian-Dialect-Diacritization-main
python run_pipeline.py --track track3 --head-type linear_head --model marbert
```

## Running Experiments

The single entrypoint for every track/head/model combination is `run_pipeline.py`. On every run it will, in order: (1) detect a GPU via `nvidia-smi` and install the matching CPU/CUDA `torch` build if it isn't already installed correctly, (2) `pip install -r requirements.txt` if requirements changed since the last run, (3) check that `./data` looks valid (skips the Drive fetch since data is already bundled here), then (4) dispatch to `training/<track>/<head_type>/finetune_<head_type>.py` with the model you asked for.

```bash
# Track 3, linear head, MARBERT backbone
python run_pipeline.py --track track3 --head-type linear_head --model marbert

# Track 3, BiLSTM-CRF head, AraBERT backbone (current best result, DER 0.0483)
python run_pipeline.py --track track3 --head-type bilstm_crf_head --model arabert_v02
```

`--track`, `--head-type`, and `--model` are required. Valid `--model` values depend on `--head-type` — see the backbone table above, or run with `--help` to list every discovered `(track, head_type)` combination:
```bash
python run_pipeline.py --help
```

Useful flags:
| Flag | Effect |
|---|---|
| `--dry-run` | Print the resolved command/working-dir instead of actually running it — good for sanity-checking before a long training job |
| `--data-dir PATH` | Point at a dataset folder that lives elsewhere instead of `./data` (the folder must be named `data`) |
| `--skip-install` / `--reinstall` | Skip, or force, the `pip install -r requirements.txt` step |
| `--skip-torch` / `--force-torch` / `--cpu-only` | Control the automatic torch install (use what's already installed, force a re-check, or force the CPU build even with a GPU present) |
| `--skip-data-fetch` / `--force-data-fetch` | Skip, or force, re-downloading `./data` from Google Drive |
| `--drive-folder-id ID` | Override the `DRIVE_FOLDER_ID` placeholder in `utils/fetch_data.py` without editing the file |
| `--cuda-index URL` | pip `--index-url` to use for the CUDA torch build (default: `cu128`; see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) if it 404s for your driver) |

Anything after these flags that `run_pipeline.py` doesn't recognize is forwarded straight through to the underlying `finetune_<head_type>.py` script's own `argparse`.

**Outputs**: checkpoints and exported artifacts land under `working/checkpoints/` and `working/exports/` (created on first run, gitignored). Training logs print to the console; metric reports are written by the evaluation scripts into `evaluation/<track>/<head_type>/report_*.md` and rolled up into `experiments/leaderboard.md`.

## Adding a New Track (Track 1 / 2 / 4)
> **⚠️ Temporary section — whoever implements Track 1, 2, or 4 should delete this section once their track lands and its own docs cover it.** This exists so the guidance isn't only living in a PR thread someone has to go dig up.

Follow the exact folder pattern `track3/` already uses — no changes to `run_pipeline.py` needed for this part, it auto-discovers any `training/<track>/<head_type>/finetune_<head_type>.py`:
```
configs/track<N>/<head_type>/strategy_a_<model>_<score>.yaml
models/track<N>/<head_type>/*_model.py
training/track<N>/<head_type>/finetune_<head_type>.py
evaluation/track<N>/<head_type>/{evaluate_<head_type>.py, report_strategy_a_*.md}
experiments/track<N>/<head_type>/strategy_a_overview.md
```

### Adding your own CLI flags
Adding a track-specific flag is two separate steps — don't conflate them:
1. **Define the flag** in your own `finetune_<head_type>.py`'s `argparse` — this part you always do yourself, same as any CLI script. `run_pipeline.py` has no idea what flags your track needs.
2. **Wire it into `run_pipeline.py`** — you *don't* have to do this. It's a thin dispatcher, not a shared argparse everyone extends. Any argument on the command line it doesn't recognize is forwarded as-is to your script, already parsed correctly (typed values, boolean flags, multi-value flags — not just plain strings).

So concretely: add `p.add_argument("--sequence-length", type=int, default=64)` to your own script, and it already works through the shared entrypoint with zero changes to `run_pipeline.py`:
```bash
python run_pipeline.py --track track1 --head-type bilstm_crf \
    --model <key> --sequence-length 128 --layers 64 128 256
```
Only touch `run_pipeline.py` itself for something meant to be shared across *all* tracks (e.g. a new shared data-source flag) — not something specific to your own track.

**Reserved names — avoid these for your own flags**, since `run_pipeline.py` already owns them and will silently swallow a same-named flag instead of forwarding it (no error, just wrong behavior): `--track`, `--head-type`, `--model`, `--data-dir`, `--dry-run`, `--skip-install`, `--reinstall`, `--skip-data-fetch`, `--force-data-fetch`, `--drive-folder-id`, `--skip-torch`, `--force-torch`, `--cpu-only`, `--cuda-index`.

### Existing pattern to follow, not reinvent
Track 3's scripts barely use CLI flags — they only take `--active-model`; every hyperparameter (batch size, lr, epochs, dropout, etc.) lives in a per-model config dict baked into the script, selected by that one flag. Intended shape: **`--head-type` picks the architecture variant** (folder-level), **`--model` picks a named preset from your own `MODEL_REGISTRY`** (not necessarily an HF checkpoint — for tracks with no pretrained backbone, it's just a preset name), and reach for a real CLI flag only when something needs to vary *per run*, not per preset.

### Rough guess at what each track will need
Going by the architectures named above and what's already precedented in Track 3 — not a spec, just a starting point so nobody's staring at a blank page:

- **Track 1 (BiLSTM taggers):** no pretrained backbone, so `--model` maps to a hyperparameter preset (e.g. `bilstm_crf_deep`), not a checkpoint. The three architectures are three `head_type` folders (`bilstm_cnn`, `bilstm_crf`, `bilstm_cnn_crf`), same split style as Track 3's `linear_head`/`bilstm_crf_head`. Beyond the preset: `--hidden-dim`, `--lstm-layers`, `--dropout`, and `--cnn-filters`/`--cnn-kernel-sizes` for the CNN variants only.
- **Track 2 (char-based LLMs):** ByT5 and CANINE aren't the same shape as each other. CANINE is encoder-only, so it fits Track 3's per-char classification pattern directly (`--model` = HF checkpoint key). ByT5 is encoder-decoder, usually meaning text-to-text *generation* instead of tagging — a different training loop, not just a different backbone. If so, that's naturally two `head_type`s: `char_tagging_head` (CANINE) and `seq2seq_head` (ByT5, wanting `--generation-max-length`/`--num-beams` as real per-run flags).
- **Track 4 (from-scratch Transformer):** trained from scratch like Track 1, so `--model` is an architecture-size preset, not a checkpoint. Since "from scratch" is the point, architecture hyperparameters are worth exposing as real flags: `--num-layers`, `--d-model`, `--num-heads`, `--ffn-dim`, `--dropout`, `--max-seq-len`, plus `--cnn-filters`/`--cnn-kernel-sizes` for the `transformer_cnn_crf` variant only.

## Team and Acknowledgments
- **Institution**: ENSIA Research Team
- **Supervisors**:  Dr. Mohamed Hadj Ameur, Dr. Mohamed Brahimi, Dr. ElMoatez Billah Nagoudi
- **Contributors**: Aya Benali Khodja, Selma Khelili, Lyes Hadjar, Soundous Chemam, Manel Ait Said, Mazouz Ahmed Thabet, Zyad Kherraf, Hadil Hattabi, Omar Ziyad Chaalel, Khadidja Bahfir, Younes Barmaki, Guergour Youcef, Roumaissa Smail, Aya Benmansour, Souha Nour Abidat, Ines Bencherif, Basmala Randa Benmaiche, Yousra Kassous, Zahra Abdeli, Fatma Imene Djelili, Takoua Hidoussi