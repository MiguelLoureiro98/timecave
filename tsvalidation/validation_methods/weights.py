"""
This module contains built-in functions to compute weights for validation methods that so require.
"""

import numpy as np


# Define 'splits' instead of splitting indices...
def constant_weights(n_splits: int, gap: int) -> np.ndarray:
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    n_splits : int
        _description_

    gap : int
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    splits = n_splits - gap

    return np.ones(splits)


# Define functions for linear and exponential weights; based on recent/old data and the amount of data.
