import random

import numpy as np
import torch


def set_seed(s: int) -> None:
    """Seed every source of randomness in a training run (notebook §1).

    Called at the top of train_model, so a seed fixes weight init, dropout
    masks, batch shuffling and char dropout together. The seed ensemble in §13
    relies on this: same config, different seed, genuinely different model.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
