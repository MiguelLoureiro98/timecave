#   Copyright 2024 Beatriz Lourenço, Miguel Loureiro, IS4
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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


def constant_weights(
    n_splits: int, gap: int = 0, compensation: int = 0, params: dict = None
) -> np.ndarray:
    """
    Compute constant weights.

    This function computes a constant weight vector. It is called by the [Growing Window](../prequential/grow.md), 
    [Rolling Window](../prequential/roll.md), and [Block CV](../CV/block.md) by default.

    Parameters
    ----------
    n_splits : int
        Number of splits the validation method will use.

    gap : int, default=0
        Number of folds separating the validation set from the training set. \
        Used by [prequential methods](../prequential/index.md).

    compensation : int, default=0
        A compensation factor that allows the function to generate the correct amount of weights. \
        0 for [CV methods](../CV/index.md), +1 for [prequential methods](../prequential/index.md). \
        Additionally, if a gap is specified, it must be added to this compensation factor as well.

    params : dict, default=None
        Used for compatibility. Irrelevant for this function.

    Returns
    -------
    np.ndarray
        Weights.

    Examples
    --------
    >>> from timecave.validation_methods.weights import constant_weights
    >>> constant_weights(5);
    array([1., 1., 1., 1., 1.])

    If a gap is specified, there will be fewer iterations. Therefore, fewer weights should be generated:

    >>> constant_weights(5, gap=1);
    array([1., 1., 1., 1.])

    For a given number of folds, CV methods will run for an additional iteration compared to prequential 
    methods. Therefore, a compensation factor of 1 must be specified if one intends to use weighted prequential 
    methods:

    >>> constant_weights(5, gap=1, compensation=1);
    array([1., 1., 1.])
    """

    splits = n_splits - gap - compensation

    return np.ones(splits)


def linear_weights(
    n_splits: int, gap: int = 0, compensation: int = 0, params: dict = {"slope": 2}
) -> np.ndarray:
    """
    Compute linear weights.

    This function computes a linear weight vector. It may be passed to the [Growing Window](../prequential/grow.md), 
    [Rolling Window](../prequential/roll.md), and [Block CV](../CV/block.md) classes.

    Parameters
    ----------
    n_splits : int
        Number of splits the validation method will use.

    gap : int, default=0
        Number of folds separating the validation set from the training set. \
        Used by [prequential methods](../prequential/index.md).

    compensation : int, default=0
        A compensation factor that allows the function to generate the correct amount of weights. \
        0 for [CV methods](../CV/index.md), +1 for [prequential methods](../prequential/index.md). \
        Additionally, if a gap is specified, it must be added to this compensation factor as well.

    params : dict, default=None
        Parameters from which to generate the weights. Only `slope` needs to be specified. 
        Any other parameter will be ignored.

    Returns
    -------
    np.ndarray
        Weights.

    Raises
    ------
    TypeError
        If `slope` is neither an integer nor a float.

    ValueError
        If `slope` is not positive.

    Examples
    --------
    >>> from timecave.validation_methods.weights import linear_weights
    >>> linear_weights(5);
    array([0.06666667, 0.13333333, 0.2       , 0.26666667, 0.33333333])

    If a gap is specified, there will be fewer iterations. Therefore, fewer weights should be generated:

    >>> linear_weights(5, gap=1);
    array([0.1, 0.2, 0.3, 0.4])

    For a given number of folds, CV methods will run for an additional iteration compared to prequential 
    methods. Therefore, a compensation factor of 1 must be specified if one intends to use weighted prequential 
    methods:

    >>> linear_weights(5, gap=1, compensation=1);
    array([0.16666667, 0.33333333, 0.5       ])
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
    Compute exponential weights.

    This function computes a exponential weight vector. It may be passed to the [Growing Window](../prequential/grow.md), 
    [Rolling Window](../prequential/roll.md), and [Block CV](../CV/block.md) classes.

    Parameters
    ----------
    n_splits : int
        Number of splits the validation method will use.

    gap : int, default=0
        Number of folds separating the validation set from the training set. \
        Used by [prequential methods](../prequential/index.md).

    compensation : int, default=0
        A compensation factor that allows the function to generate the correct amount of weights. \
        0 for [CV methods](../CV/index.md), +1 for [prequential methods](../prequential/index.md). \
        Additionally, if a gap is specified, it must be added to this compensation factor as well.

    params : dict, default=None
        Parameters from which to generate the weights. Only `base` needs to be specified. 
        Any other parameter will be ignored.

    Returns
    -------
    np.ndarray
        Weights.

    Raises
    ------
    TypeError
        If `base` is neither an integer nor a float.

    ValueError
        If `base` is not positive.

    Examples
    --------
    >>> from timecave.validation_methods.weights import exponential_weights
    >>> exponential_weights(5);
    array([0.03225806, 0.06451613, 0.12903226, 0.25806452, 0.51612903])

    If a gap is specified, there will be fewer iterations. Therefore, fewer weights should be generated:

    >>> exponential_weights(5, gap=1);
    array([0.06666667, 0.13333333, 0.26666667, 0.53333333])

    For a given number of folds, CV methods will run for an additional iteration compared to prequential 
    methods. Therefore, a compensation factor of 1 must be specified if one intends to use weighted prequential 
    methods:

    >>> exponential_weights(5, gap=1, compensation=1);
    array([0.14285714, 0.28571429, 0.57142857])
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

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);