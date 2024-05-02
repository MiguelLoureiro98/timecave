"""
This module contains built-in functions to compute weights for validation methods that so require.
"""

import numpy as np


# Define 'splits' instead of splitting indices...
def constant_weights(n_splits: int, gap: int) -> np.ndarray:

    splits = n_splits - gap

    return np.ones(splits)


# Define functions for linear and exponential weights; based on recent/old data and the amount of data.
