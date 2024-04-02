import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union


def sinusoid_ts(max_interval_size: float, 
                 number_samples: int, 
                 amplitude: float = 1, 
                 frequency: float = 1, 
                 phase: float = 0) -> np.ndarray:
    """
    Generate a sinusoidal wave based on given parameters.

    Parameters:
        max_interval_size (float): Length of the window of values starting at 0.
        number_samples (int): Number of samples to generate.
        amplitude (float, optional): Amplitude of the sinusoid. Default is 1.
        frequency (float, optional): Frequency of the sinusoid in Hz. Default is 1.
        phase (float, optional): Phase angle of the sinusoid in radians. Default is 0.

    Returns:
        numpy.ndarray: Array of generated sinusoidal values.
    """
    time_values = np.linspace(0, max_interval_size, number_samples)
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * time_values + phase)

    return sinusoid


def frequency_modulation_with_step(time: np.ndarray, 
                                    t_split: float, 
                                    freq_init: float) -> np.ndarray:
    """
    Generate frequency modulation with a step function.

    Parameters:
        time (numpy.ndarray): Time array.
        t_split (float): Time point at which the frequency of the sinusoid changes.
        freq_init (float): Initial frequency of the sinusoid.

    Returns:
        numpy.ndarray: Frequency-modulated sinusoid.
    """
    initial_period = (1 / freq_init)
    t_split_adjusted = (t_split // initial_period) * initial_period
    return np.where(time > t_split_adjusted, freq_init * 2, freq_init)


def frequency_modulation_linear(time: np.ndarray, 
                                 slope: float, 
                                 freq_init: float) -> np.ndarray:
    """
    Generate frequency modulation with a linear function.

    Parameters:
        time (numpy.ndarray): Time array.
        slope (float): Slope of the linear function.
        freq_init (float): Initial frequency of the sinusoid.

    Returns:
        numpy.ndarray: Frequency-modulated sinusoid.
    """
    return freq_init + slope * time


def time_varying_sinusoid_ts(max_interval_size: float, 
                             number_samples: int,
                             frequency_func: callable,  
                             frequency_args: tuple = (),  
                             amplitude: float = 1, 
                             phase_initial: float = 0) -> np.ndarray:
    """
    Generate a time series of a sinusoid with varying frequency.

    Parameters:
        max_interval_size (float): The maximum interval size for the time series.
        number_samples (int): The number of samples in the time series.
        frequency_func (callable): Function describing how frequency varies with time.
        frequency_args (tuple, optional): Additional arguments for the frequency function.
        amplitude (float, optional): Amplitude of the sinusoid. Default is 1.
        phase_initial (float, optional): Initial phase of the sinusoid in radians. Default is 0.

    Returns:
        numpy.ndarray: Time series data representing the sinusoid with varying frequency.
    """
    time = np.linspace(0, max_interval_size, number_samples)
    frequency = frequency_func(time, *frequency_args)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase_initial)
    return time_series


def indicator_ts(number_samples: int, start_index: int, end_index: int) -> np.ndarray:
    """
    Generate a time series based on an indicator function in the space of real numbers.

    Parameters:
        number_samples (int): Number of samples of the time series.
        start_index (int): Index where the indicator function starts being 1.
        end_index (int): Index where the indicator function ends being 1.

    Returns:
        numpy.ndarray: Indicator function array where elements are 1 between start_index and end_index (inclusive),
        and 0 elsewhere.
    """
    indicator = np.zeros(number_samples)
    indicator[start_index:end_index + 1] = 1
    return indicator


def scaled_right_indicator_ts(number_samples: int, idx: int, constant: float = 1) -> np.ndarray:
    """
    Generate a time series based on an indicator function define on a interval open on the right and scaled by a constant values.

    Parameters:
        number_samples (int): Number of samples of the time series.
        idx (int): Index where the indicator function starts being 1.

    Returns:
        numpy.ndarray: Indicator function array where elements are 1 between start_index until the end of the series,
        and 0 elsewhere.
    """
    return constant*indicator_ts(number_samples, idx, number_samples - 1)


        

def scaled_unit_impulse_function(number_samples: int, idx: int, constant: float = 1) -> np.ndarray:
    """
    Generate a time series based on the dirac delta function scaled by a constant values.

    Parameters:
        number_samples (int): Number of samples of the time series.
        idx (int): Index where the indicator is 1.

    Returns:
        numpy.ndarray: Indicator function array where elements are 1 at idx,
        and 0 elsewhere.
    """
    return constant*indicator_ts(number_samples, idx, idx)


def linear_ts(max_interval_size: float, 
               number_samples: int, 
               slope: float = 1, 
               intercept: float = 0) -> np.ndarray:
    """
    Generate a linear function.

    Parameters:
        max_interval_size (float): The maximum interval size for the time series.
        number_samples (int): The number of samples in the time series.
        slope (float, optional): Slope of the linear function. Default is 1.
        intercept (float, optional): Intercept of the linear function. Default is 0.

    Returns:
        numpy.ndarray: Linear function array generated with the formula: slope * time + intercept.
    """
    time = np.linspace(0, max_interval_size, number_samples)
    linear_series = slope * time + intercept
    return linear_series


def exponential_ts(max_interval_size: float, 
                    number_samples: int, 
                    decay_rate: float = 1, 
                    initial_value: float = 1) -> np.ndarray:
    """
    Generate an exponential function.

    Parameters:
        max_interval_size (float): The maximum interval size for the time series.
        number_samples (int): The number of samples in the time series.
        decay_rate (float, optional): Rate of decay of the exponential function. Default is 1.
        initial_value (float, optional): Initial value of the exponential function. Default is 1.

    Returns:
        numpy.ndarray: Exponential function array generated with the formula: initial_value * exp(-decay_rate * time).
    """
    time = np.linspace(0, max_interval_size, number_samples)
    exponential_series = initial_value * np.exp(-decay_rate * time)
    return exponential_series


def white_noise(number_samples: int, 
                 mean: float = 0, 
                 std_dev: float = 1) -> np.ndarray:
    """
    Generate a white noise time series.

    Parameters:
        number_samples (int): The number of samples in the time series.
        mean (float, optional): Mean of the white noise.
        std_dev (float, optional): Standard deviation of the white noise.

    Returns:
        numpy.ndarray: White noise time series.
    """
    return np.random.normal(loc=mean, scale=std_dev, size=number_samples)


def linear_combination(weight_vector: list, ts_matrix: list) -> np.ndarray:

    nb_ts = len(ts_matrix)
    len_ts = len(ts_matrix[0])
    
    w = np.array(weight_vector)
    T = np.array(ts_matrix).T
    
    assert nb_ts == len(weight_vector)
    assert np.shape(T) == (len_ts, nb_ts )
    
    return np.matmul(T, w)

import random

def generate_random_parameters(interval: dict, seed=1):
    """
    Generate random parameters based on provided intervals.

    Parameters:
        param_intervals (dict): A dictionary where keys represent parameter names
            and values represent either tuples or lists. If the value is a tuple,
            a random value from the tuple will be selected for each parameter. If
            the value is a list, it represents an interval from which a random
            value will be sampled.
        seed (int): Optional. Seed for random number generation to ensure reproducibility.
            Defaults to 1.

    Returns:
        dict: A dictionary containing randomly generated parameter values mapped to
            their respective parameter names.

    Raises:
        ValueError: If the interval type is neither tuple nor list.
    """
    random.seed(seed)
    if isinstance(interval, tuple):
        value = random.choice(interval)
    elif isinstance(interval, list):
        value = random.uniform(interval[0], interval[1])
    else:
        raise ValueError("Invalid interval type. Must be tuple or list.")
    return value


def generate_from_func(number_samples, func: callable, generate_func: callable, param_intervals: dict, nb_ts: int, seed = 1 ) -> np.ndarray:
    """Generate time series data by invoking a specified function with randomly generated parameters.

    Parameters:
        func (callable): The function to be invoked for generating time series data. It should accept parameters as keyword arguments.
        param_intervals (dict): A dictionary specifying the parameter intervals for random generation. 
            Keys are parameter names, and values are tuples (min, max) specifying the interval for each parameter.
        nb_ts (int): Number of time series to generate.
        seed (int, optional): Seed for random number generation. Default is 1 for reproducibility.

    Returns:
        np.ndarray: A numpy array containing the generated time series data. Each row corresponds to a time series."""

    np.random.seed(seed)
    ts_list = []
    
    for _ in range(nb_ts):
        params = {}
        for param, interval in param_intervals.items():
            params[param] = generate_func(interval, seed=1)
        ts_list.append(func(number_samples = number_samples, **params))
    
    print(params)
    return ts_list

if __name__=='__main__':
    max_interval_size = 1
    samples = 1000
    sample_interval = max_interval_size/samples

    # Time Series
    ts1 = linear_ts(max_interval_size, samples, slope =10)
    ts2 = time_varying_sinusoid_ts(max_interval_size, 
                             samples,
                             amplitude = 5, 
                             frequency_func = frequency_modulation_linear,
                             frequency_args = (10, 1),
                             phase_initial = 0,
                             )
    ts_list = [ts1, ts2]


    ts_linear = generate_from_func(samples, linear_ts, generate_random_parameters, param_intervals = {'max_interval_size': (1, 2), 'slope':[5, 10], 'intercept': [0,10] }, nb_ts = 1)
    ts_exponential = generate_from_func(samples, exponential_ts, generate_random_parameters, param_intervals = {'max_interval_size': (1, 2), 
                    'decay_rate': [1, 25], 
                    'initial_value': [1, 25]}, nb_ts = 2)

    time_series = linear_combination([0.2, 0.2, 0.3], ts_linear + ts_exponential)
    
    
    
    # Plot
    plt.scatter(np.arange(0, max_interval_size, max_interval_size/samples),time_series)
    plt.show()
    print()