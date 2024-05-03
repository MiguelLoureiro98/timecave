"""
This module contains built-in functions to compute weights for validation methods that so require.
"""

import numpy as np


# Define 'splits' instead of splitting indices...
def constant_weights(
    n_splits: int, gap: int = 0, compensation: int = 0, params: dict = None
) -> np.ndarray:
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    n_splits : int
        _description_

    gap : int, optional
        _description_, by default 0

    compensation : int, optional
        _description_, by default 0

    params : dict, optional
        _description_, by default None

    Returns
    -------
    np.ndarray
        _description_
    """

    splits = n_splits - gap - compensation

    return np.ones(splits)


# Define functions for linear and exponential weights; based on recent/old data and the amount of data.
def linear_weights(
    n_splits: int, gap: int = 0, compensation: int = 0, params: dict = {"slope": 2}
) -> np.ndarray:
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    n_splits : int
        _description_

    gap : int, optional
        _description_, by default 0

    compensation : int, optional
        _description_, by default 0

    params : _type_, optional
        _description_, by default {"slope": 2}

    Returns
    -------
    np.ndarray
        _description_
    """

    splits = n_splits - gap - compensation
    weights = np.array([params["slope"] * i for i in range(1, splits + 1)])
    weights = weights / weights.sum()

    return weights


def exponential_weights(
    n_splits: int, gap: int = 0, compensation: int = 0, params: dict = {"base": 2}
) -> np.ndarray:
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    n_splits : int
        _description_

    gap : int, optional
        _description_, by default 0

    compensation : int, optional
        _description_, by default 0

    params : _type_, optional
        _description_, by default {"base": 2}

    Returns
    -------
    np.ndarray
        _description_
    """

    splits = n_splits - gap - compensation
    weights = np.array([params["base"] ** i for i in range(splits)])
    weights = weights / weights.sum()

    return weights
