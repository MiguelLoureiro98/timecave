"""
This module contains built-in functions to compute weights for validation methods that so require.

Functions
---------
constant_weights
    Generates constant weights (i.e. every fold is weighted equally).

linear_weights
    Folds are weighted linearly.

exponential_weights
    Folds are weighted exponentially.

Notes
-----
Users may write their own weighting function and pass it as an argument to a validation method class. 
However, to make sure that the user-defined function is compatible with TimeCaVe, the function signature 
must match those of the functions provided herein.
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

    _check_params(params["slope"])
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

    _check_params(params["base"])
    splits = n_splits - gap - compensation
    weights = np.array([params["base"] ** i for i in range(splits)])
    weights = weights / weights.sum()

    return weights


def _check_params(param: int) -> None:

    if (isinstance(param, int) or isinstance(param, float)) is False:

        raise TypeError("'base' and 'slope' should be integers or floats.")

    if param <= 0:

        raise ValueError("'base' and 'slope' should be positive.")
