"""Full training run: baseline -> T1 -> T5 -> seed ensemble -> threshold search.

Mirrors the notebook's §8, §10, §11, §13 and §16 in one script. The ablation
ladder (§12) is deliberately not here: it was never run, so there are no results
to reproduce.

The threshold grids live here rather than in configs/, matching
experiments/track4/SmailRoumaissa/train_run.py -- a search space is an
experiment setting, while the frozen winner is a config value.
"""

import itertools

from configs.track4.souha.model_config import ModelConfig, PLAIN_BASELINE
from configs.track4.souha.paths import find_data_paths
from configs.track4.souha.training_config import (
    EnsembleConfig,
    LexicalFallbackConfig,
    TrainingConfig,
)
from evaluation.track4.souha.baselines import lookup_baseline
from evaluation.track4.souha.inference import predict_with_confidence
from evaluation.track4.souha.lexical_fallback import gated_labels
from evaluation.track4.souha.metrics import fmt, letters_microf1
from training.track4.souha.trainer import train_model
from utils.track4.souha.data import DiacData
from utils.track4.souha.device import get_device

MAX_CONF_GRID     = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
MIN_COUNT_GRID    = [1, 2, 3, 5]
MIN_MAJORITY_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def search_thresholds(data, dev_enc, dev_labels, dev_conf, verbose=True):
    """Grid-search the V2 gate on dev (§16), then freeze it.

    Optimises `letters_microf1`, which is the leaderboard metric. Note this is
    tuned on the same dev split used for checkpoint selection, so the reported
    gain is optimistic.
    """
    neural_only = letters_microf1(dev_enc, dev_labels)
    best = (-1.0, None)
    for max_conf, min_count, min_majority in itertools.product(
            MAX_CONF_GRID, MIN_COUNT_GRID, MIN_MAJORITY_GRID):
        gated = [gated_labels(r["chars"], lab, conf, data,
                              max_conf, min_count, min_majority)
                 for r, lab, conf in zip(dev_enc, dev_labels, dev_conf)]
        score = letters_microf1(dev_enc, gated)
        if score > best[0]:
            best = (score, (max_conf, min_count, min_majority))

    max_conf, min_count, min_majority = best[1]
    if verbose:
        print(f"neural-only dev microF1 : {neural_only:.4f}")
        print(f"V2        dev microF1   : {best[0]:.4f}  (+{best[0]-neural_only:.4f})")
        print(f"frozen thresholds       : MAX_CONF={max_conf}  "
              f"MIN_COUNT={min_count}  MIN_MAJORITY={min_majority}")
    return LexicalFallbackConfig(max_conf=max_conf, min_count=min_count,
                                 min_majority=min_majority)


def run_training(epochs: int = 40, ensemble_cfg: EnsembleConfig = None,
                 train_plain_baseline: bool = True, device: str = None):
    device = device or get_device()
    paths = find_data_paths()
    missing = paths.missing()
    if missing:
        print("MISSING:", missing)
        return None

    ensemble_cfg = ensemble_cfg or EnsembleConfig()
    data = DiacData(paths)
    train_enc = data.encode(data.train)
    dev_enc = data.encode(data.dev)
    print(f"train {len(train_enc)} sents | dev {len(dev_enc)} sents | "
          f"vocab {len(data.vocab)} | device {device}")

    # ---- §8 memorisation floor ---------------------------------------------
    baseline_wer = lookup_baseline(data)

    # ---- §10 T1, the required plain char-level Transformer baseline --------
    m_t1 = None
    if train_plain_baseline:
        print("\n=== T1 plain char-level Transformer ===")
        _, m_t1, p_t1 = train_model(data, train_enc, dev_enc, PLAIN_BASELINE,
                                    TrainingConfig(epochs=epochs, seed=0), device)

    # ---- §11 T5, the full Transformer-CNN-CRF ------------------------------
    print("\n=== T5 Transformer-CNN-CRF ===")
    model_t5, m_t5, p_t5 = train_model(data, train_enc, dev_enc, ModelConfig(),
                                       TrainingConfig(epochs=epochs, seed=0), device)

    # ---- §13 seed ensemble --------------------------------------------------
    print("\n=== seed ensemble ===")
    final_models = []
    for s in ensemble_cfg.seeds:
        if s == 0:
            # identical config and seed as T5 above, and train_model reseeds on
            # entry, so retraining would reproduce it exactly. Reuse instead.
            final_models.append(model_t5)
            print(f"seed {s} | {fmt(m_t5)}  (reused from T5)")
            continue
        mdl, m, _ = train_model(data, train_enc, dev_enc, ModelConfig(),
                                TrainingConfig(epochs=epochs, seed=s), device,
                                verbose=False)
        final_models.append(mdl)
        print(f"seed {s} | {fmt(m)}")
    print(f"\n{p_t5:,} parameters per model")

    # ---- §16 freeze the V2 gate on dev -------------------------------------
    print("\n=== V2 confidence-gated lexical fallback ===")
    dev_labels, dev_conf = predict_with_confidence(final_models, data, dev_enc, device)
    fallback_cfg = search_thresholds(data, dev_enc, dev_labels, dev_conf)

    return dict(models=final_models, data=data, paths=paths, device=device,
                train_enc=train_enc, dev_enc=dev_enc,
                fallback_cfg=fallback_cfg, baseline_wer=baseline_wer,
                metrics_t1=m_t1, metrics_t5=m_t5, n_params=p_t5)


if __name__ == "__main__":
    run_training()
