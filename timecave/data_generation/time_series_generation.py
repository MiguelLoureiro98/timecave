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
A module for generating time series data using the functions provided by this package.

This module provides a class for generating time series data based on provided functions, with optional noise and weights.

Classes
-------
TimeSeriesGenerator
    A class for generating time series data using the functions provided by the [time_series_functions](../time_series_functions/index.md) module.
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
    A class for generating time series data using provided functions.

    This class enables the generation of multiple time series by combining various functions with parameters.
    It allows customization of the time series length, noise level, weights for functions, and parameter values.

    Parameters
    ----------
    functions : List[Callable]
        A list of functions used to generate the time series.

    length : int, default=100
        The length of the time series to be generated.

    noise_level : float, default=0.1
        The standard deviation of the Gaussian noise added to the time series.

    weights : List[float], optional
        A list of weights corresponding to each function.

    parameter_values : list[Dict], optional
        A list of dictionaries containing parameter values for each function. Each dictionary contains
        parameter names as keys and either single values, tuples (for discrete choices), or lists (for continuous ranges)
        as values, representing the possible values or ranges for each parameter.

    Attributes
    ----------
    time_series : list [np.ndarray]
        The generated time series.

    Methods
    -------
    generate(nb_sim: int, og_seed: int = 1)
        Generate time series data.

    plot(indexes: np.array or list or range = None)
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

        og_seed : int, default=1
            The original seed for generating random numbers.

        Returns
        -------
        List[np.array]
            A list of numpy arrays containing generated time series data.

        Warning
        -------
        The `number_samples` parameter is inferred from the `length` parameter. 
        Therefore, it should not be passed to the method using the `parameter_values` argument.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from timecave.data_generation.time_series_functions import linear_ts, indicator_ts, exponential_ts
        >>> from timecave.data_generation.time_series_generation import TimeSeriesGenerator

        Generate 3 time series using a combination of linear, indicator, and exponential functions:

        >>> gen = TimeSeriesGenerator([linear_ts, indicator_ts, exponential_ts],
        ...                            length=10,
        ...                            parameter_values=[{"max_interval_size": [10, 10], "slope": [1, 5]}, # A random slope between 1 and 5 will be generated.
        ...                                              {"start_index": 2, "end_index": 6},
        ...                                              {"max_interval_size": [10, 10], "decay_rate": [0.1, 10]}]); # 'max_interval_size' will always be 10.
        >>> ts = gen.generate(3);
        >>> ts
        [array([ 1.1550022 ,  5.08763814, 10.75491998, 15.05675456, 19.71847986,
               24.3300877 , 28.65378249, 32.42146755, 36.97335225, 41.32569794]), array([ 1.090524  ,  3.94981031,  7.7498048 , 10.77923442, 13.51272171,
               16.82720927, 19.84383981, 21.52150532, 24.56596177, 27.69603248]), array([ 1.03173452,  4.41648421,  9.72828385, 13.96528833, 18.37799188,
               22.61230447, 26.9295597 , 30.32151586, 34.57860333, 38.94221321])]
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
            Indexes of time series to plot. None by default.

        Warning
        -------
        This method should only be called once the `generate` method has been run.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from timecave.data_generation.time_series_generation import TimeSeriesGenerator
        >>> from timecave.data_generation.time_series_functions import linear_ts, scaled_right_indicator_ts, exponential_ts, sinusoid_ts

        Generate 5 time series using a combination of linear and indicator functions:

        >>> gen = TimeSeriesGenerator([linear_ts, scaled_right_indicator_ts],
        ...                            length=200,
        ...                            parameter_values=[{"max_interval_size": [10, 10], "slope": [1, 10]},
        ...                                              {"idx": 100, "constant": [5, 100]}]);
        >>> ts = gen.generate(5);
        >>> gen.plot([0, 1, 2, 3, 4]);

        ![gen_plots](../../../images/Gen_plots1.png)

        Using a combination of exponential functions and sinusoids instead:

        >>> gen2 = TimeSeriesGenerator([exponential_ts, sinusoid_ts],
        ...                             length=1000,
        ...                             parameter_values=[{"max_interval_size": [10, 10], "decay_rate": [1, 10], "initial_value": [1, 10]},
        ...                                               {"max_interval_size": [10, 10], "amplitude": [0.5, 3], "frequency": [0.1, 5]}]);
        >>> ts2 = gen2.generate(5);
        >>> gen2.plot([0, 1, 2, 3, 4]);

        ![gen_plots2](../../../images/Gen_plots2.png)
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

    #from timecave.data_generation import frequency_modulation as dgu
    #from timecave.data_generation import time_series_functions as tsf

    #linear_parameters = {"max_interval_size": 1, "slope": 5, "intercept": [5, 30]}

    #exp_parameters = {
    #    "max_interval_size": (1, 2),
    #    "decay_rate": [1, 25],
    #    "initial_value": [1, 25],
    #}

    #sin_parameters = {
    #    "max_interval_size": (1, 2),
    #    "amplitude": [1, 3],
    #    "frequency": (
    #        dgu.FrequencyModulationLinear(1, 20),
    #        dgu.FrequencyModulationWithStep(10, 0.8),
    #    ),
    #}

    #impulse_parameters = {"idx": (500, 600), "constant": [5, 10]}

    #indicator_parameters = {"start_index": (700, 600), "end_index": (800, 900)}

    #generator = TimeSeriesGenerator(
    #    length=1000,
    #    noise_level=0.2,
    #    functions=[
    #        tsf.linear_ts,
    #        tsf.indicator_ts,
    #        tsf.frequency_varying_sinusoid_ts,
    #        tsf.scaled_unit_impulse_function_ts,
    #        tsf.exponential_ts,
    #    ],
    #    parameter_values=[
    #        linear_parameters,
    #        indicator_parameters,
    #        sin_parameters,
    #        impulse_parameters,
    #        exp_parameters,
    #    ],
    #)
    #generator.generate(100)

    #generator.plot(range(0, 10))

    import doctest

    doctest.testmod(verbose=True);