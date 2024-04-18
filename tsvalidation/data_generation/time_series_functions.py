import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from tsvalidation.data_generation.utils import FrequencyModulation, nonlin_func, get_arma_parameters

def sinusoid_ts(number_samples: int, 
                max_interval_size: float, 
                 amplitude: float = 1, 
                 frequency: float = 1, 
                 phase: float = 0) -> np.ndarray:
    time_values = np.linspace(0, max_interval_size, number_samples)
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * time_values + phase)

    return sinusoid


def frequency_varying_sinusoid_ts(number_samples: int,
                                  max_interval_size: float, 
                                    frequency: FrequencyModulation,  
                                    amplitude: float = 1, 
                                    phase_initial: float = 0) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    frequency = frequency.modulate(time = time)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase_initial)
    return time_series


def indicator_ts(number_samples: int, start_index: int, end_index: int) -> np.ndarray:
    indicator = np.zeros(number_samples)
    indicator[start_index:end_index + 1] = 1
    return indicator


def scaled_right_indicator_ts(number_samples: int, idx: int, constant: float = 1) -> np.ndarray:
    return constant*indicator_ts(number_samples, idx, number_samples - 1)
       

def scaled_unit_impulse_function_ts(number_samples: int, idx: int, constant: float = 1) -> np.ndarray:
    return constant*indicator_ts(number_samples, idx, idx)


def linear_ts(number_samples: int,
              max_interval_size: float, 
               slope: float = 1, 
               intercept: float = 0) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    linear_series = slope * time + intercept
    return linear_series


def exponential_ts(number_samples: int,
                   max_interval_size: float, 
                    decay_rate: float = 1, 
                    initial_value: float = 1) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    exponential_series = initial_value * np.exp(-decay_rate * time)
    return exponential_series


def white_noise_ts(number_samples: int, 
                 mean: float = 0, 
                 std_dev: float = 1) -> np.ndarray:
    return np.random.normal(loc=mean, scale=std_dev, size=number_samples)



def arma_ts(number_samples, lags, max_root, ar = True, ma = True, seed = 1, **kwargs):
    params_ar = get_arma_parameters(lags, max_root, seed = seed)
    params_ma = get_arma_parameters(lags, max_root, seed = seed)
    ar_coeff = np.r_[1, -params_ar]
    ma_coeff = np.r_[1, params_ma]

    if ar and not ma:
        model = "AR"
        ts = ArmaProcess(ma=ar_coeff).generate_sample(nsample=number_samples, **kwargs)
    elif not ar and ma:
        model = "MA"
        ts = ArmaProcess(ar=[1], ma=ma_coeff).generate_sample(nsample=number_samples, **kwargs)
    elif ar and ma:
        model = "ARMA"
        ts = ArmaProcess(ar=ar_coeff, ma=ma_coeff).generate_sample(nsample=number_samples, **kwargs)
    return ts


def nonlinear_ar_ts(number_samples, init_array, params, func_idxs):
    init_len = len(init_array)

    x = np.empty(number_samples + init_len)
    x[0:init_len] = init_array

    for t in range(init_len, number_samples + init_len):
        x[t] = np.random.normal(scale=0.5)

        for j in range(1, init_len + 1):
            x[t] += params[j-1] * nonlin_func(func_idxs[j-1], x[t-j])

    model = "nonlinear_ar"
    ts = x[init_len:(number_samples + init_len)]

    return ts





