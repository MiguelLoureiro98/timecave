import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from tsvalidation.data_generation.frequency_modulation import BaseFrequency
from tsvalidation.data_generation._utils import nonlin_func, get_arma_parameters


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
    _check_amplitude(amplitude)
    _check_frequency(frequency)
    _check_phase(phase)

    time_values = np.linspace(0, max_interval_size, number_samples)
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * time_values + phase)

    return sinusoid


def _check_phase(phase: float) -> None:
    """
    Checks if the 'phase' must be a float.
    """
    if isinstance(phase, int) is False:

        raise TypeError("'phase' should be a float.")
    return


def _check_frequency(frequency: int or float) -> None:
    """
    Checks if the 'frequency' must be a non-negative float or int.
    """
    if isinstance(frequency, (float, int)) is False:

        raise TypeError("'frequency' should be a float or int.")

    if frequency > 0:

        raise ValueError("'frequency' must be greater than zero.")
    return


def _check_amplitude(amplitude: float) -> None:
    """
    Checks if the 'amplitude' must be a float.
    """
    if isinstance(amplitude, float) is False:

        raise TypeError("'amplitude' should be a float.")
    return


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
    _check_phase(phase_initial)
    _check_amplitude(amplitude)

    time = np.linspace(0, max_interval_size, number_samples)
    frequency = frequency.modulate(time=time)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase_initial)
    return time_series


def indicator_ts(number_samples: int, start_index: int, end_index: int) -> np.ndarray:
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
    return constant * indicator_ts(number_samples, idx, number_samples - 1)


def scaled_unit_impulse_function_ts(
    number_samples: int, idx: int, constant: float = 1
) -> np.ndarray:
    return constant * indicator_ts(number_samples, idx, idx)


def linear_ts(
    number_samples: int,
    max_interval_size: float,
    slope: float = 1,
    intercept: float = 0,
) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    linear_series = slope * time + intercept
    return linear_series


def exponential_ts(
    number_samples: int,
    max_interval_size: float,
    decay_rate: float = 1,
    initial_value: float = 1,
) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    exponential_series = initial_value * np.exp(-decay_rate * time)
    return exponential_series


def white_noise_ts(
    number_samples: int, mean: float = 0, std_dev: float = 1
) -> np.ndarray:
    return np.random.normal(loc=mean, scale=std_dev, size=number_samples)


def arma_ts(number_samples, lags, max_root, ar=True, ma=True, seed=1, **kwargs):
    params_ar = get_arma_parameters(lags, max_root, seed=seed)
    params_ma = get_arma_parameters(lags, max_root, seed=seed)
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
    init_len = len(init_array)

    x = np.empty(number_samples + init_len)
    x[0:init_len] = init_array

    for t in range(init_len, number_samples + init_len):
        x[t] = np.random.normal(scale=0.5)

        for j in range(1, init_len + 1):
            x[t] += params[j - 1] * nonlin_func(func_idxs[j - 1], x[t - j])

    ts = x[init_len : (number_samples + init_len)]

    return ts
