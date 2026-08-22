from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Architecture of the from-scratch character-level Transformer-CNN-CRF tagger.

    Ported from the `Cfg` dataclass in the notebook (§1), keeping the field
    names unchanged. `Cfg` mixed architecture, optimisation, data location and
    runtime device in one object; here they are split:

        architecture      -> this file
        optimisation      -> training_config.TrainingConfig
        data location     -> paths.find_data_paths()
        runtime device    -> utils.track4.AbidatSouha.device.get_device()

    The booleans are the notebook's ablation switches. All-True is the full
    model (`T5` in §11); flipping them off reproduces the plain char-level
    Transformer (`T1` in §10).
    """

    # ---- vocabulary and label space ----------------------------------------
    vocab_size: int = 43                # len(vocab.json)
    pad_id: int = 0                     # notebook: PAD
    num_classes: int = 16               # notebook: N_CLASSES

    # ---- size (tuned for 133k labelled positions; do not scale up blindly) --
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.25

    # ---- positional encoding and attention biases --------------------------
    rel_pos: str = "t5"                 # "t5" | "sinusoidal" | "none"
    rel_buckets: int = 32
    rel_max_dist: int = 64
    same_word_bias: bool = True         # learned per-head intra-word attention bias

    # ---- morphological input streams (§2) ----------------------------------
    # pos_in_word, dist_start, dist_end, wlen, mater_lectionis, sun_letter
    use_features: bool = True
    feat_sizes: Tuple[int, ...] = (5, 5, 5, 7, 3, 3)

    # ---- CNN front-end (§4) -------------------------------------------------
    use_conv: bool = True
    conv_kernels: Tuple[int, ...] = (3, 5, 7)

    # ---- linear-chain CRF (§5) ----------------------------------------------
    use_crf: bool = True
    split_crf: bool = True              # separate intra-/inter-word transitions

    # ---- output head (§7) ---------------------------------------------------
    factorized_head: bool = True        # label = 8 * shadda + base
    interaction: bool = True            # 2x8 interaction table
    char_prior: bool = True             # per-character logit prior from train freqs
    aux_diac_head: bool = True          # auxiliary "is diacritic-bearing" head


# Config of the plain char-level Transformer baseline (`T1`, §10): the required
# track-4 baseline, kept deliberately vanilla — sinusoidal positional encoding,
# softmax head, no conv, no CRF, no morphological features.
PLAIN_BASELINE = ModelConfig(
    rel_pos="sinusoidal",
    same_word_bias=False,
    use_features=False,
    use_conv=False,
    use_crf=False,
    factorized_head=False,
    interaction=False,
    char_prior=False,
    aux_diac_head=False,
)
