"""
Module for generating various types of time series data.

This module provides functions to generate different types of time series data, including sinusoidal signals, indicator functions, linear patterns, exponential decays,
and time series based on autoregressive moving average (ARMA) models and nonlinear autoregressive (AR) models.

Functions
---------
sinusoid_ts
    Generate a time series of a sinusoidal signal.
frequency_varying_sinusoid_ts
    Generate a time series of a sinusoidal signal with varying frequency.
indicator_ts
    Generate time series array based on a binary indicator function with specified start and end indices.
scaled_right_indicator_ts
    Generate a time series array based on a indicator function that is 1 in the interval [idx, + inf[ and 0 otherwise.
scaled_unit_impulse_function_ts
    Generate time series array based on a scaled unit impulse function with specified index.
linear_ts
    Generate a linear time series array.
exponential_ts
    Generates a time series based on a exponential function.
arma_ts
    Generate a time series array based on an Autoregressive Moving Average (ARMA) model.
nonlinear_ar_ts
    Generate a time series array based on a nonlinear autoregressive (AR) model.
"""

import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from tsvalidation.data_generation.frequency_modulation import BaseFrequency
from tsvalidation.data_generation._utils import _nonlin_func, _get_arma_parameters


def sinusoid_ts(
    number_samples: int,
    max_interval_size: float,
    amplitude: float = 1,
    frequency: float or int = 1,
    phase: float = 0,
) -> np.ndarray:
    """
    Generate a time series of a sinusoidal signal.

    This function generates a time series of a sinusoidal signal with the specified parameters.

    Parameters
    ----------
    number_samples : int
        The number of samples in the generated time series.
    max_interval_size : float
        The maximum interval size (time duration) of the generated time series.
    amplitude : float, optional
        The amplitude of the sinusoidal signal, by default 1.
    frequency : float, optional
        The frequency of the sinusoidal signal in cycles per unit time, by default 1.
    phase : float, optional
        The phase offset of the sinusoidal signal in radians, by default 0.

    Returns
    -------
    np.ndarray
        The generated sinusoidal time series.
    """
    _check_number_samples(number_samples)
    _check_max_interval_size(max_interval_size)
    time_values = np.linspace(0, max_interval_size, number_samples)
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * time_values + phase)

    return sinusoid


def _check_max_interval_size(max_interval_size: float) -> None:
    """
    Checks if the 'max_interval_size' must be a non-negative float.
    """
    if isinstance(max_interval_size, int) is False:

        raise TypeError("'max_interval_size' should be a float.")

    if max_interval_size <= 0:

        raise ValueError("'max_interval_size' must be greater than zero.")
    return


def _check_number_samples(number_samples: int, min_samples: int = 0) -> None:
    """
    Checks if the 'number_samples' must be a non-negative int.
    """
    if isinstance(number_samples, int) is False:

        raise TypeError("'number_samples' should be a int.")

    if number_samples <= min_samples:

        raise ValueError(f"'number_samples' must be greater than {min_samples}.")

    return


def frequency_varying_sinusoid_ts(
    number_samples: int,
    max_interval_size: float,
    frequency: BaseFrequency,
    amplitude: float = 1,
    phase_initial: float = 0,
) -> np.ndarray:
    """
    Generate a time series of a sinusoidal signal with varying frequency.

    This function generates a time series of a sinusoidal signal where the frequency varies over time.

    Parameters
    ----------
    number_samples : int
        The number of samples in the time series.
    max_interval_size : float
        The maximum time interval size for the time series.
    frequency : BaseFrequency
        An object representing the base frequency of the sinusoid, which may vary over time.
    amplitude : float, optional
        The amplitude of the sinusoidal signal, by default 1.
    phase_initial : float, optional
        The initial phase of the sinusoidal signal in radians, by default 0.

    Returns
    -------
    np.ndarray
        An array representing the generated time series of the sinusoidal signal with varying frequency.
    """
    _check_number_samples(number_samples)
    _check_max_interval_size(max_interval_size)

    time = np.linspace(0, max_interval_size, number_samples)
    frequency = frequency.modulate(time=time)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase_initial)
    return time_series


def indicator_ts(number_samples: int, start_index: int, end_index: int) -> np.ndarray:
    """
    Generate time series array based on a binary indicator function with specified start and end indices.

    This function creates a time series array based on a binary indicator function of given length. The specified segment is marked as 1 and the rest as 0.

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.
    start_index : int
        The start index of the segment to be marked as 1.
    end_index : int
        The end index of the segment to be marked as 1.

    Returns
    -------
    np.ndarray
        A time series array where the specified segment is marked as 1 and the rest as 0.
    """
    _check_number_samples(number_samples)
    _check_index(start_index)
    _check_index(end_index)
    indicator = np.zeros(number_samples)
    indicator[start_index : end_index + 1] = 1
    return indicator


def _check_index(index: int) -> None:
    """
    Checks if the 'index' must be a non-negative int.
    """
    if isinstance(index, int) is False:

        raise TypeError("'index' should be a int.")

    if index <= 0:

        raise ValueError("'index' must be greater than zero.")

    return


def scaled_right_indicator_ts(
    number_samples: int, idx: int, constant: float = 1
) -> np.ndarray:
    """
    Generate a time series array based on a indicator function that is 1 in the interval [idx, + inf[ and 0 otherwise.

    This function creates a time series array of given length where the segment starting from the specified index to the end is marked as 1 and the rest as 0.
    The binary array is then scaled by a constant factor.

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.
    idx : int
        The index from which the segment starts to be marked as 1.
    constant : float, optional
        A scaling constant to multiply the array by, by default 1.

    Returns
    -------
    np.ndarray
        A scaled time series array where the segment starting from the specified index to the end is marked as 1 and the rest as 0.
    """
    _check_number_samples(number_samples)
    return constant * indicator_ts(number_samples, idx, number_samples - 1)


def scaled_unit_impulse_function_ts(
    number_samples: int, idx: int, constant: float = 1
) -> np.ndarray:
    """
    Generate time series array based on a scaled unit impulse function with specified index.

    This function creates a binary indicator time series array of given length where
    only the sample at the specified index is marked as 1 and the rest as 0.
    The binary array is then scaled by a constant factor.

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.
    idx : int
        The index at which the impulse occurs, marked as 1.
    constant : float, optional
        A scaling constant to multiply the array by, by default 1.

    Returns
    -------
    np.ndarray
        A scaled time series array where only the sample at the specified index is marked as 1 and the rest as 0.
    """
    _check_number_samples(number_samples)
    return constant * indicator_ts(number_samples, idx, idx)


def linear_ts(
    number_samples: int,
    max_interval_size: float,
    slope: float = 1,
    intercept: float = 0,
) -> np.ndarray:
    """
    Generate a linear time series array.

    This function creates a time series array of given length where the values
    follow a linear pattern determined by the slope and intercept parameters.

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.
    max_interval_size : float
        The maximum interval size for generating the time series array.
    slope : float, optional
        The slope of the linear pattern, by default 1.
    intercept : float, optional
        The intercept of the linear pattern, by default 0.

    Returns
    -------
    np.ndarray
        A time series array following a linear pattern determined by the slope and intercept parameters.
    """
    _check_number_samples(number_samples)
    time = np.linspace(0, max_interval_size, number_samples)
    linear_series = slope * time + intercept
    return linear_series


def exponential_ts(
    number_samples: int,
    max_interval_size: float,
    decay_rate: float = 1,
    initial_value: float = 1,
) -> np.ndarray:
    """
    Generates a time series based on a exponential function.

    This function creates a time series array of given length where the values
    decay exponentially over time based on the specified decay rate and initial value.

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.
    max_interval_size : float
        The maximum interval size for generating the time series array.
    decay_rate : float, optional
        The rate at which the values decay over time, by default 1.
    initial_value : float, optional
        The initial value of the time series array, by default 1.

    Returns
    -------
    np.ndarray
        An exponential decay time series array where values decay exponentially over time based on the specified decay rate and initial value.
    """
    _check_number_samples(number_samples)
    time = np.linspace(0, max_interval_size, number_samples)
    exponential_series = initial_value * np.exp(-decay_rate * time)
    return exponential_series


def arma_ts(number_samples, lags, max_root, ar=True, ma=True, seed=1, **kwargs):
    """
    Generate a time series array based on an Autoregressive Moving Average (ARMA) model.

    This function creates a time series array of given length based on an ARMA model
    with specified parameters such as number of lags and maximum root. It generates
    samples using an ARMA process.

    #TODO Reference Bergmeir

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.
    lags : int
        The number of lags to consider in the ARMA model.
    max_root : float
        The maximum root for the ARMA model. This value has to be larger than 1.1.
    ar : bool, optional
        Whether to include autoregressive (AR) component in the ARMA model, by default True.
    ma : bool, optional
        Whether to include moving average (MA) component in the ARMA model, by default True.
    seed : int, optional
        Random seed for reproducibility, by default 1.
    **kwargs : dict
        Additional keyword arguments to pass to the ARMA process generator.

    Returns
    -------
    np.ndarray
        A time series array generated based on the specified ARMA model parameters.

    Raises
    -------
    ValueError
        If the maximum root is not larger than 1.1.
    """
    params_ar = _get_arma_parameters(lags, max_root, seed=seed)
    params_ma = _get_arma_parameters(lags, max_root, seed=seed)
    ar_coeff = np.r_[1, -params_ar]
    ma_coeff = np.r_[1, params_ma]

    if ar and not ma:
        ts = ArmaProcess(ma=ar_coeff).generate_sample(nsample=number_samples, **kwargs)
    elif not ar and ma:
        ts = ArmaProcess(ar=[1], ma=ma_coeff).generate_sample(
            nsample=number_samples, **kwargs
        )
    elif ar and ma:
        ts = ArmaProcess(ar=ar_coeff, ma=ma_coeff).generate_sample(
            nsample=number_samples, **kwargs
        )
    return ts


def nonlinear_ar_ts(number_samples, init_array, params, func_idxs):
    """
    Generate a time series array based on a nonlinear autoregressive (AR) model.

    This function creates a time series array of given length based on a nonlinear AR model
    with specified initial array, parameters, and function indices.

    #TODO Reference Bergmeir

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.
    init_array : np.ndarray
        The initial array for generating the time series.
    params : list
        The parameters for the nonlinear AR model.
    func_idxs : list
        The indices of the nonlinear functions used in the model.

    Returns
    -------
    np.ndarray
        A time series array generated based on the specified nonlinear AR model parameters.
    """
    init_len = len(init_array)

    x = np.empty(number_samples + init_len)
    x[0:init_len] = init_array

    for t in range(init_len, number_samples + init_len):
        x[t] = np.random.normal(scale=0.5)

        for j in range(1, init_len + 1):
            x[t] += params[j - 1] * _nonlin_func(func_idxs[j - 1], x[t - j])

    ts = x[init_len : (number_samples + init_len)]

    return ts
