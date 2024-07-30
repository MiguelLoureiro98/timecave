"""
A module for generating time series data using provided functions.

This module provides a class for generating time series data based on provided functions, with optional noise and weights.

Classes
-------
TimeSeriesGenerator
    A class for generating time series data using provided functions.
"""

import numpy as np

import matplotlib.pyplot as plt
from timecave.data_generation._utils import (
    _generate_random_parameters,
    _generate_seeds,
)

from typing import Callable, List, Dict


class TimeSeriesGenerator:
    """
    TimeSeriesGenerator(functions: List[Callable], length: int = 100, noise_level: float or int = 0.1, weights: List[float] = None, parameter_values: list[Dict] = None)
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    A class for generating time series data using provided functions.

    This class enables the generation of multiple time series by combining various functions with parameters.
    It allows customization of the time series length, noise level, weights for functions, and parameter values.

    Parameters
    ----------
    functions : List[Callable]
        A list of functions used to generate the time series.
    length : int, optional
        The length of the time series to be generated, by default 100.
    noise_level : float, optional
        The standard deviation of the Gaussian noise added to the time series, by default 0.1.
    weights : List[float], optional
        A list of weights corresponding to each function, by default None.
    parameter_values : list[Dict], optional
        A list of dictionaries containing parameter values for each function, by default None. Each dictionary contains
        parameter names as keys and either single values, tuples (for discrete choices), or lists (for continuous ranges)
        as values, representing the possible values or ranges for each parameter.

    Methods
    -------
    generate
        Generate time series data.
    plot
        Plot the generated time series.

    Raises
    ------
    ValueError
        If the lengths of 'functions', 'parameter_values', and 'weights' don't match.
    """

    def __init__(
        self,
        functions: List[Callable],
        length: int = 100,
        noise_level: float or int = 0.1,
        weights: List[float] = None,
        parameter_values: list[Dict] = None,
    ) -> None:
        self._check_functions(functions)
        self._check_length(length)
        self._check_noise_level(noise_level)
        self._check_weights(weights)
        self._check_parameter_values(parameter_values)

        if len(functions) != len(parameter_values):

            raise ValueError(
                "Lengths of 'functions', 'parameter_values', and 'weights' must match."
            )
        if weights is not None and (
            len(weights) != len(parameter_values) or len(functions) != len(weights)
        ):

            raise ValueError(
                "Lengths of 'functions', 'parameter_values', and 'weights' must match."
            )

        self._functions = functions
        self._length = length
        self._noise_level = noise_level
        self._parameter_values = parameter_values
        self._weights = weights
        self.time_series = []
        if weights is None:
            self._weights = [1.0] * len(functions)

    def _check_parameter_values(self, parameter_values: float):
        """
        Check if 'parameter_values' is a list of dictionaries.
        """
        if isinstance(parameter_values, list) is False or not all(
            isinstance(params, Dict) for params in parameter_values
        ):

            raise TypeError("'parameter_values' should be a list of parameter_values.")

    def _check_weights(self, weights: float):
        """
        Check if 'weights' is a positive float.
        """
        if isinstance(weights, list) is False and weights is not None:

            raise TypeError("'weights' should be an float.")

    def _check_functions(self, functions: List[Callable]) -> None:
        """
        Check if 'functions' is a list of functions.
        """
        if isinstance(functions, list) is False or not all(
            isinstance(func, Callable) for func in functions
        ):

            raise TypeError("'functions' should be a list of functions.")

    def _check_noise_level(self, noise_level: float or int):
        """
        Check if 'noise_level' is a non-negative float.
        """
        if isinstance(noise_level, (float, int)) is False:

            raise TypeError("'noise_level' should be an float or int.")

        if noise_level < 0:

            raise ValueError("'noise_level' must be greater or equal to zero.")

    def _check_length(self, length: int):
        """
        Check if 'length' is a positive integer.
        """
        if length <= 0:

            raise ValueError("'length' must be greater than zero.")

        if isinstance(length, int) is False:

            raise TypeError("'length' should be an int.")

    def generate(self, nb_sim: int, og_seed: int = 1):
        """
        Generate time series data.

        Parameters
        ----------
        nb_sim : int
            Number of simulations to generate.
        og_seed : int, optional
            The original seed for generating random numbers, by default 1.

        Returns
        -------
        List[np.array]
            A list of numpy arrays containing generated time series data.
        """

        seeds = _generate_seeds(og_seed, nb_sim)

        for seed in seeds:
            np.random.seed(seed)
            ts = np.zeros(self._length)
            for i in range(len(self._functions)):
                parameters = _generate_random_parameters(
                    self._parameter_values[i], seed=seed
                )
                ts += self._weights[i] * self._functions[i](self._length, **parameters)

            ts += np.random.normal(scale=self._noise_level, size=self._length)

            self.time_series.append(ts)

        return self.time_series

    def plot(self, indexes: np.array or list or range = None):
        """
        Plot the generated time series.

        Parameters
        ----------
        indexes : np.array or list or range, optional
            Indexes of time series to plot, by default None.
        """
        if indexes is None:
            indexes = range(len(self.time_series))
        elif isinstance(indexes, int):
            indexes = [indexes]

        for idx in indexes:
            plt.plot(self.time_series[idx], label=f"Time Series {idx}")

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Generated Time Series")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    from timecave.data_generation import frequency_modulation as dgu
    from timecave.data_generation import time_series_functions as tsf

    linear_parameters = {"max_interval_size": 1, "slope": 5, "intercept": [5, 30]}

    exp_parameters = {
        "max_interval_size": (1, 2),
        "decay_rate": [1, 25],
        "initial_value": [1, 25],
    }

    sin_parameters = {
        "max_interval_size": (1, 2),
        "amplitude": [1, 3],
        "frequency": (
            dgu.FrequencyModulationLinear(1, 20),
            dgu.FrequencyModulationWithStep(10, 0.8),
        ),
    }

    impulse_parameters = {"idx": (500, 600), "constant": [5, 10]}

    indicator_parameters = {"start_index": (700, 600), "end_index": (800, 900)}

    generator = TimeSeriesGenerator(
        length=1000,
        noise_level=0.2,
        functions=[
            tsf.linear_ts,
            tsf.indicator_ts,
            tsf.frequency_varying_sinusoid_ts,
            tsf.scaled_unit_impulse_function_ts,
            tsf.exponential_ts,
        ],
        parameter_values=[
            linear_parameters,
            indicator_parameters,
            sin_parameters,
            impulse_parameters,
            exp_parameters,
        ],
    )
    generator.generate(100)

    generator.plot(range(0, 10))
