import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all random sources.

    Args:
        seed: a seed to set.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
