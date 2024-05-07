"""
Module for utility functions used in time series data generation.

This module provides utility functions for generating time series data, including nonlinear 
transformation functions, ARMA model parameter generation, random parameter generation, and seed generation.

Functions
---------
_nonlin_func
    Apply various nonlinear transformations to the input value.
_get_arma_parameters
    Generate autoregressive (AR) parameters for an ARMA model.
_generate_random_parameters
    Generate random parameters based on specified possibilities.
_generate_seeds
    Generate a list of random seeds based on specified parameters.
"""

import numpy as np
import random


def _nonlin_func(nb: int, x: float) -> float:
    """
    This function applies various nonlinear transformations to the input value 'x'
    based on the specified function number 'nb'.
    """
    if nb not in [0, 1, 2, 3, 4]:
        raise ValueError("'nb' must be 0, 1, 2, 3 or 4.")

    if isinstance(x, float) is False:
        raise TypeError("'x' must be float")

    nonlin_x = {
        0: np.cos(x),
        1: np.sin(x),
        2: np.tanh(x),
        3: np.arctan(x),
        4: np.exp(-x / 10000),
    }[nb]
    return nonlin_x


def _get_arma_parameters(lags: int, max_root: int or float, seed: int = 1) -> np.array:
    """
    This function generates autoregressive (AR) parameters for an ARMA (AutoRegressive
    Moving Average) model with the specified number of lags (`lags`) and maximum root
    value (`max_root`).
    """
    np.random.seed(seed)
    if max_root <= 1.1:
        raise ValueError("max_root must be bigger than 1.1")

    if isinstance(max_root, (int, float)) is False:
        raise TypeError("'max_root' must be int or float")

    if isinstance(seed, int) is False:
        raise TypeError("'seed' must be int")

    if isinstance(lags, int) is False:
        raise TypeError("'lags' must be int")

    s = np.sign(np.random.uniform(-1, 1, lags))
    poly_roots = s * np.random.uniform(1.1, max_root, lags)

    coeff = np.array([1])
    for root in poly_roots:
        coeff = np.polymul(coeff, np.array([root * -1, 1]))

    n_coeff = coeff / coeff[0]
    params = -n_coeff[1:]

    return params


def _generate_random_parameters(param_possibilities: dict, seed=1) -> dict:
    """
    This function generates random parameters based on the possibilities provided
    in the `param_possibilities` dictionary.
    """
    if isinstance(param_possibilities, dict) is False:
        raise TypeError("'param_possibilities' must be a dictionary")

    if len(param_possibilities.keys()) == 0:
        raise ValueError("'param_possibilities' must be a non-empty dictionary")

    if isinstance(seed, int) is False:
        raise TypeError("'seed' must be int")

    random.seed(seed)
    params = {}
    for key, values in param_possibilities.items():
        if isinstance(values, tuple):
            value = random.choice(values)
        elif isinstance(values, list):
            value = random.uniform(values[0], values[1])
        else:
            value = values
        params[key] = value

    return params


def _generate_seeds(seed: int, num_seeds: int) -> list:
    """
    This function generates a list of random seeds based on the specified
    parameters.
    """
    if isinstance(seed, int) is False:
        raise TypeError("'seed' must be int")

    if isinstance(num_seeds, int) is False:
        raise TypeError("'num_seeds' must be int")

    random.seed(seed)

    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]

    return seeds
