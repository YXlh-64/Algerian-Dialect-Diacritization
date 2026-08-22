from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    """Optimisation settings, ported from `Cfg` (§1) and the training loop (§9).

    Schedule is linear warmup for `warmup` steps, then cosine decay to zero over
    the remaining steps. Model selection keeps the checkpoint with the best dev
    `select_metric`; training stops after `patience` epochs without improvement.
    """

    # Key of the dict returned by evaluation.track4.AbidatSouha.metrics.evaluate that
    # early stopping and checkpoint selection are driven by. Direction is looked
    # up in metrics.HIGHER_IS_BETTER, so "macro_f1"/"micro_f1" maximise while
    # "der_letters"/"wer" minimise. The notebook selected on "der_letters".
    select_metric: str = "macro_f1"

    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 40
    batch_size: int = 32
    warmup: int = 300                   # optimiser steps, not epochs
    grad_clip: float = 1.0
    patience: int = 8                   # epochs without dev improvement
    char_dropout: float = 0.08          # train only; simulates the 14.7% OOV rate
    label_smoothing: float = 0.05       # softmax head only, ignored when use_crf
    aux_weight: float = 0.3             # weight on the auxiliary diacritic head
    seed: int = 0


@dataclass
class EnsembleConfig:
    """Seed ensemble (§13).

    Independently initialised models of identical architecture; their
    per-position log-probabilities are averaged before Viterbi decoding. The
    seed controls weight init, dropout masks, batch shuffling and char dropout.
    """

    seeds: Tuple[int, ...] = (0, 1, 2)


@dataclass
class LexicalFallbackConfig:
    """V2 confidence-gated lexical fallback (§16).

    For each whitespace-delimited word seen in the training lexicon, the neural
    prediction is overwritten by the lexicon's majority vocalisation when all of:

        total occurrences in the lexicon  >= min_count
        majority share of those           >= min_majority
        weakest-letter confidence         <  max_conf

    plus a skeleton check that the majority vocalisation has the same letter
    count as the word being overwritten. `max_conf` is a ceiling, not a floor:
    the fallback fires only where the model is *unsure*.

    NOTE: the values below are grid midpoints, NOT tuned. They are placeholders
    until the dev grid search in experiments/track4/AbidatSouha/train_run.py has run;
    replace them with the frozen triple it prints, then never re-tune on dev.
    """

    max_conf: float = 0.90
    min_count: int = 2
    min_majority: float = 0.80
