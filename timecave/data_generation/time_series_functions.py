"""
This module contains functions to generate various types of time series data.

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
from timecave.data_generation.frequency_modulation import BaseFrequency
from timecave.data_generation._utils import _nonlin_func, _get_arma_parameters


def sinusoid_ts(
    number_samples: int,
    max_interval_size: float or int,
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

    amplitude : float, default=1
        The amplitude of the sinusoidal signal.

    frequency : float, default=1
        The frequency of the sinusoidal signal in cycles per unit time.

    phase : float, default=0
        The phase offset of the sinusoidal signal in radians.

    Returns
    -------
    np.ndarray
        The generated sinusoidal time series.

    See also
    --------
    [frequency_varying_sinusoid_ts](time_varying_sin.md): Generate a time-varying sinusoid.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import sinusoid_ts
    >>> ts = sinusoid_ts(1000, 10, amplitude=3, frequency=0.5);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![sinusoid](../../../images/Sinusoid.png)

    Higher frequency sinusoids can be generated as well:

    >>> ts2 = sinusoid_ts(1000, 10, amplitude=3, frequency=5);
    >>> _ = plt.plot(np.arange(0, ts2.shape[0]), ts2);
    >>> plt.show();

    ![sinusoid_freq](../../../images/Sinusoid_high_freq.png)

    Phase shifts can be added too:

    >>> ts3 = sinusoid_ts(1000, 10, amplitude=3, frequency=0.5, phase=0.3);
    >>> _ = plt.plot(np.arange(0, ts2.shape[0]), ts3);
    >>> plt.show();

    ![sinusoid_shift](../../../images/Shifted_sinusoid.png)
    """
    _check_number_samples(number_samples)
    _check_max_interval_size(max_interval_size)
    time_values = np.linspace(0, max_interval_size, number_samples)
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * time_values + phase)

    return sinusoid


def _check_max_interval_size(max_interval_size: float) -> None:
    """
    Checks if the 'max_interval_size' must be a non-negative float or int.
    """
    if isinstance(max_interval_size, (float, int)) is False:

        raise TypeError("'max_interval_size' should be a float or int.")

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
    phase: float = 0,
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

    amplitude : float, default=1
        The amplitude of the sinusoidal signal.

    phase : float, default=1
        The initial phase of the sinusoidal signal in radians.

    Returns
    -------
    np.ndarray
        An array representing the generated time series of the sinusoidal signal with varying frequency.

    See also
    --------
    [sinusoid_ts](sinusoid.md): Generate a simple sinusoid.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.frequency_modulation import FrequencyModulationWithStep, FrequencyModulationLinear
    >>> from timecave.data_generation.time_series_functions import frequency_varying_sinusoid_ts

    Generate a sinusoid whose frequency varies abruptly:

    >>> mod = FrequencyModulationWithStep(20, 5);
    >>> ts = frequency_varying_sinusoid_ts(100, 10, frequency=mod);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![step_freq](../../../images/Step_freq.png)

    Time series with linearly varying frequencies can be generated as well:

    >>> mod_lin = FrequencyModulationLinear(1, 0.2);
    >>> ts2 = frequency_varying_sinusoid_ts(1000, 10, frequency=mod_lin, amplitude=5);
    >>> _ = plt.plot(np.arange(0, ts2.shape[0]), ts2);
    >>> plt.show();

    ![lin_freq](../../../images/Lin_freq.png)
    """
    _check_number_samples(number_samples)
    _check_max_interval_size(max_interval_size)

    time = np.linspace(0, max_interval_size, number_samples)
    frequency = frequency.modulate(time=time)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase)
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

    See also
    --------
    [scaled_right_indicator_ts](scaled_indicator.md): Scaled right indicator time series. Needs only a start index.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import indicator_ts
    >>> ts = indicator_ts(100, 20, 60);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![indicator](../../../images/Indicator.png)
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

    See also
    --------
    [indicator_ts](indicator.md): Indicator time series.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import scaled_right_indicator_ts
    >>> ts = scaled_right_indicator_ts(100, 30, 5);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![indicator](../../../images/Scaled_indicator.png)
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

    constant : float, default=1
        A scaling constant to multiply the array by.

    Returns
    -------
    np.ndarray
        A scaled time series array where only the sample at the specified index is marked as 1 and the rest as 0.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import scaled_unit_impulse_function_ts
    >>> ts = scaled_unit_impulse_function_ts(1000, 250, 20);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![impulse](../../../images/Impulse.png)
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

    slope : float, default=1
        The slope of the linear pattern.

    intercept : float, default=0
        The intercept of the linear pattern.

    Returns
    -------
    np.ndarray
        A time series array following a linear pattern determined by the slope and intercept parameters.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import linear_ts
    >>> ts = linear_ts(1000, 10);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![linear](../../../images/Linear.png)
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

    decay_rate : float, default=1
        The rate at which the values decay over time.

    initial_value : float, default=1
        The initial value of the time series array.

    Returns
    -------
    np.ndarray
        An exponential decay time series array where values decay exponentially over time based on the specified decay rate and initial value.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import exponential_ts
    >>> ts = exponential_ts(1000, 10, initial_value=10);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![exponential](../../../images/Exponential.png)
    """
    _check_number_samples(number_samples)
    time = np.linspace(0, max_interval_size, number_samples)
    exponential_series = initial_value * np.exp(-decay_rate * time)
    return exponential_series


def arma_ts(
    number_samples: int,
    lags: int,
    max_root: float,
    ar: bool = True,
    ma: bool = True,
    **kwargs,
):
    """
    Generate a time series array based on an Autoregressive Moving Average (ARMA) model.

    This function creates a time series array of given length based on an ARMA model
    with specified parameters such as number of lags and maximum root. It generates
    samples using an ARMA process.

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.

    lags : int
        The number of lags to consider in the ARMA model.

    max_root : float
        The maximum root for the ARMA model. This value has to be larger than 1.1.

    ar : bool, default=True
        Whether to include autoregressive (AR) component in the ARMA model.

    ma : bool, default=True
        Whether to include moving average (MA) component in the ARMA model.

    **kwargs : dict
        Additional keyword arguments to pass to the ARMA process generator. 
        See [ARMAProcess](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.html) for more details.

    Returns
    -------
    np.ndarray
        A time series array generated based on the specified ARMA model parameters.

    Raises
    -------
    ValueError
        If the maximum root is not larger than 1.1.

    See also
    --------
    [nonlinear_ar_ts](nonlinear_ar.md): Generate data from a nonlinear autoregressive process.

    Notes
    -----
    This method of generating synthetic time series data was first proposed by Bergmeir and Benitez (2012). 
    Please refer to [[1]](#1) for more details on this method.

    References
    ----------
    ##1
    Christoph Bergmeir and José M Benítez. On the use of cross-validation for
    time series predictor evaluation. Information Sciences, 191:192–213, 2012.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import arma_ts
    >>> ts = arma_ts(1000, 5, 2);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![arma](../../../images/ARMA.png)

    Pure autoregressive processes can also be generated:

    >>> ts2 = arma_ts(1000, 5, 2, ma=False);
    >>> _ = plt.plot(np.arange(0, ts2.shape[0]), ts2);
    >>> plt.show();

    ![ar](../../../images/AR.png)

    And so can pure moving average processes:

    >>> ts3 = arma_ts(1000, 5, 2, ar=False);
    >>> _ = plt.plot(np.arange(0, ts3.shape[0]), ts3);
    >>> plt.show();

    ![ma](../../../images/MA.png)
    """
    _check_number_samples(number_samples)

    if ar == False and ma == False:
        raise ValueError("At least one of 'ar' or 'ma' must be set to True.")

    params_ar = _get_arma_parameters(lags, max_root)
    params_ma = _get_arma_parameters(lags, max_root)
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


def nonlinear_ar_ts(
    number_samples: int, init_array: np.array, params: list, func_idxs: list
):
    """
    Generate a time series array based on a nonlinear autoregressive (AR) model.

    This function creates a time series array of a given length based on a nonlinear AR model
    with specified initial array, parameters, and function indices.

    Parameters
    ----------
    number_samples : int
        The total number of samples in the time series array.

    init_array : np.ndarray
        The initial array for generating the time series. The lengths corresponds to the number of lags.

    params : list
        The parameters for the nonlinear AR model. The index representing the specific nonlinear transformation to apply:
            0: Cosine function.
            1: Sine function.
            2: Hyperbolic tangent function.
            3: Arctangent function.
            4: Exponential decay function.
            
    func_idxs : list
        The indices of the nonlinear functions used in the model.

    Returns
    -------
    np.ndarray
        A time series array generated based on the specified nonlinear AR model parameters.

    Warnings
    --------
    The lengths of `init_array`, `params` and `func_idxs` must match.

    Notes
    -----
    This method of generating synthetic time series data was first proposed by Bergmeir et al. (2018). 
    Please refer to [[1]](#1) for more details on this method.

    References
    ----------
    ##1
    Christoph Bergmeir, Rob J Hyndman, and Bonsoo Koo. A note on the validity 
    of cross-validation for evaluating autoregressive time series prediction.
    Computational Statistics & Data Analysis, 120:70–83, 2018.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from timecave.data_generation.time_series_functions import nonlinear_ar_ts
    >>> ts = nonlinear_ar_ts(1000, init_array=np.zeros(2), params=[0.5, -0.3], func_idxs=[0, 1]);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts);
    >>> plt.show();

    ![ARsin](../../../images/Nonlinear_AR_sin.png)

    Functions other than sinusoids can be used as well:

    >>> ts2 = nonlinear_ar_ts(1000, init_array=np.zeros(4), params=[0.2, 0.6, -0.1, -0.4], func_idxs=[2, 3, 4, 3]);
    >>> _ = plt.plot(np.arange(0, ts.shape[0]), ts2);
    >>> plt.show();

    ![ARother](../../../images/Nonlinear_AR_others.png)
    """
    _check_number_samples(number_samples)
    if len(params) != len(func_idxs):
        raise ValueError("'params' and 'func_idxs' must have the same length")

    init_len = len(init_array)

    x = np.empty(number_samples + init_len)
    x[0:init_len] = init_array

    for t in range(init_len, number_samples + init_len):
        x[t] = np.random.normal(scale=0.5)

        for j in range(1, init_len + 1):
            x[t] += params[j - 1] * _nonlin_func(func_idxs[j - 1], x[t - j])

    ts = x[init_len : (number_samples + init_len)]

    return ts

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);