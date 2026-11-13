"""Some module tests."""


import os

from xlmexlab.randomization import seed_everything


def test_seed_everything():
    seed = 42
    seed_everything(seed)
    assert os.environ["PYTHONHASHSEED"] == str(seed)
