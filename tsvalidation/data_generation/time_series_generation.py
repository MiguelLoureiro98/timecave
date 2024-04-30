import numpy as np
from tsvalidation.data_generation import time_series_functions as tsf
import matplotlib.pyplot as plt
from tsvalidation.data_generation._utils import (
    generate_random_parameters,
    generate_seeds,
)
from tsvalidation.data_generation import utils as dgu
from typing import Callable, List


class TimeSeriesGenerator:
    """
    Generate multiple random time series based on given functions and set/list of parameters.

    This class enables the generation of multiple time series by combining various functions with parameters.
    It allows customization of the time series length, noise level, weights for functions, and parameter values.

    Parameters
    ----------
    functions : List[Callable]
        List of functions to generate time series data. The function first parameter must be the time series length.
    length : int, optional
        Length of the generated time series data, by default 100.
    noise_level : float, optional
        Level of noise to be added to the generated data, by default 0.1.
    weights : List[float], optional
        Weights assigned to each function for generating combined time series data, by default None.
    parameter_values : list, optional
        List of parameter values corresponding to each function, by default None.

    Attributes
    ----------
    time_series : List[np.array]
        List containing all the generated time series data.

    """

    def __init__(
        self,
        functions: List[Callable],
        length: int = 100,
        noise_level: float = 0.1,
        weights: List[float] = None,
        parameter_values: list = None,
    ) -> None:
        self._check_functions(self, functions)
        self._check_length(self, length)
        self._check_noise_level(noise_level)

        assert len(functions) == len(parameter_values)
        self._functions = functions
        self._length = length
        self._noise_level = noise_level
        self._parameter_values = parameter_values
        self._weights = weights
        if weights == None:
            self._weights = [1.0] * len(functions)
        self.time_series = []

    def _check_functions(self, functions: List[Callable]) -> None:
        """
        Checks if the provided functions is a list of callable objects.
        """

        if not isinstance(functions, list) or not all(
            callable(func) for func in functions
        ):
            raise TypeError("'functions' must be a list of callable objects.")

        return

    def _check_length(self, length: int) -> None:
        """
        Checks if the provided length is a positive integer.
        """
        # REVER VALUE ERROR/TYPE ERROR
        if not isinstance(length, int) or length <= 0:
            raise ValueError("'length' must be a positive integer.")

        return

    def _check_noise_level(self, noise_level: float) -> None:
        """
        Checks if the provided 'noise_level' is a positive integer.
        """

        if not isinstance(noise_level, (int, float)) or not 0 <= noise_level <= 1:
            raise ValueError("'noise_level' must be a float between 0 and 1.")

        return

    def generate(self, nb_sim, og_seed=1):

        seeds = generate_seeds(og_seed, nb_sim)

        for seed in seeds:
            np.random.seed(seed)
            ts = np.zeros(self.length)
            for i in range(len(self.functions)):
                parameters = generate_random_parameters(
                    self.parameter_values[i], seed=seed
                )
                ts += self.weights[i] * self.functions[i](self.length, **parameters)

            ts += np.random.normal(scale=self.noise_level, size=self.length)

            self.time_series.append(ts)

        return self.time_series

    def plot(self, indexes=None):
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
