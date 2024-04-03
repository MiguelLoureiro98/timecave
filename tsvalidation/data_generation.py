import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX



def sinusoid_ts(max_interval_size: float, 
                 number_samples: int, 
                 amplitude: float = 1, 
                 frequency: float = 1, 
                 phase: float = 0) -> np.ndarray:
    time_values = np.linspace(0, max_interval_size, number_samples)
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * time_values + phase)

    return sinusoid


def frequency_modulation_with_step(time: np.ndarray, 
                                    t_split: float, 
                                    freq_init: float) -> np.ndarray:
    initial_period = (1 / freq_init)
    t_split_adjusted = (t_split // initial_period) * initial_period
    return np.where(time > t_split_adjusted, freq_init * 2, freq_init)


def frequency_modulation_linear(time: np.ndarray, 
                                 slope: float, 
                                 freq_init: float) -> np.ndarray:
    return freq_init + slope * time


def frequency_varying_sinusoid_ts(max_interval_size: float, 
                             number_samples: int,
                             frequency_func: callable,  
                             frequency_args: tuple = (),  
                             amplitude: float = 1, 
                             phase_initial: float = 0) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    frequency = frequency_func(time, *frequency_args)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase_initial)
    return time_series


def indicator_ts(number_samples: int, start_index: int, end_index: int) -> np.ndarray:
    indicator = np.zeros(number_samples)
    indicator[start_index:end_index + 1] = 1
    return indicator


def scaled_right_indicator_ts(number_samples: int, idx: int, constant: float = 1) -> np.ndarray:
    return constant*indicator_ts(number_samples, idx, number_samples - 1)


        

def scaled_unit_impulse_function(number_samples: int, idx: int, constant: float = 1) -> np.ndarray:
    return constant*indicator_ts(number_samples, idx, idx)


def linear_ts(max_interval_size: float, 
               number_samples: int, 
               slope: float = 1, 
               intercept: float = 0) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    linear_series = slope * time + intercept
    return linear_series


def exponential_ts(max_interval_size: float, 
                    number_samples: int, 
                    decay_rate: float = 1, 
                    initial_value: float = 1) -> np.ndarray:
    time = np.linspace(0, max_interval_size, number_samples)
    exponential_series = initial_value * np.exp(-decay_rate * time)
    return exponential_series


def white_noise(number_samples: int, 
                 mean: float = 0, 
                 std_dev: float = 1) -> np.ndarray:
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

def generate_random_parameters(possible_values: list or tuple, seed=1):
    random.seed(seed)
    if isinstance(possible_values, tuple):
        value = random.choice(possible_values)
    elif isinstance(possible_values, list):
        value = random.uniform(possible_values[0], possible_values[1])
    else:
        raise ValueError("Invalid interval type. Must be tuple or list.")
    return value

def generate_random_arma_parameters(lags, max_root, seed =1):
    np.random.seed(seed)
    if max_root <= 1.1:
        raise ValueError("max_root has to be bigger than 1.1")

    s = np.sign(np.random.uniform(-1, 1, lags)) # random signs
    poly_roots = s * np.random.uniform(1.1, max_root, lags) # random roots

    # Calculate coefficients
    coeff = np.array([1])
    for root in poly_roots:
        coeff = np.polymul(coeff, np.array([root*-1, 1])) # get the polynomial coefficients


    n_coeff = coeff / coeff[0]
    params = -n_coeff[1:] #remove the bias

    return params


def generate_from_func(number_samples, func: callable, generate_func: callable, param_possibilities: dict, nb_ts: int, seed = 1 ) -> np.ndarray:

    np.random.seed(seed)
    ts_list = []
    
    for _ in range(nb_ts):
        params = {}
        for param_name, values in param_possibilities.items():
            params[param_name] = generate_func(values, seed=1)
        ts_list.append(func(number_samples = number_samples, **params))
    
    print(params)
    return ts_list

def arma_ts(number_samples, params_ar = None, params_ma = None, **kwargs):
    ar = params_ar is not None
    ma = params_ma is not None
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


    return {"params_ar": params_ar, "params_ma": params_ma, "model": model, "ts": ts}


def nonlin_func(nb, x):
        nonlin_x = {
            1: np.cos(x),
            2: np.sin(x),
            3: np.tanh(x),
            4: np.arctan(x),
            5: np.exp(-x/10000)
        }[nb]
        return nonlin_x

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

    return {'model': model, 'ts': ts}

def sar_sim(number_samples, n_simulations, ts, seed):
    #Model
    sar = SARIMAX(ts, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    sar_fit = sar.fit()

    #Simulation
    return sar_fit.simulate(number_samples, n_simulations, random_state=seed)



if __name__=='__main__':
    max_interval_size = 1
    samples = 1000
    sample_interval = max_interval_size/samples

    # Time Series
    ts1 = linear_ts(max_interval_size, samples, slope =10)
    ts2 = frequency_varying_sinusoid_ts(max_interval_size, 
                             samples,
                             amplitude = 5, 
                             frequency_func = frequency_modulation_linear,
                             frequency_args = (10, 1),
                             phase_initial = 0,
                             )
    ts_list = [ts1, ts2]
    #data = pd.read_csv('accidental-deaths-in-usa-monthly.csv', usecols=[1], names = ['accidental_deaths'], skiprows=1)
    #ts_sar=sar_sim(samples, 1, data.accidental_deaths.to_numpy(), seed=1)

    ts_arma = arma_ts(samples, params_ar = generate_random_arma_parameters(4,5), params_ma = generate_random_arma_parameters(4,5))
    ts_nonlinear_ar = nonlinear_ar_ts(samples, 
                                  params = generate_random_arma_parameters(4,5), 
                                  init_array = np.random.normal(scale=0.5, size=4), 
                                  func_idxs = np.ceil(np.random.uniform(low=0, high=5, size=4)))


    ts_linear = generate_from_func(samples, linear_ts, generate_random_parameters, param_intervals = {'max_interval_size': (1, 2), 'slope':[5, 10], 'intercept': [0,10] }, nb_ts = 1)
    ts_exponential = generate_from_func(samples, exponential_ts, 
                                        generate_random_parameters, 
                                        param_intervals = {'max_interval_size': (1, 2), 
                                            'decay_rate': [1, 25], 
                                            'initial_value': [1, 25]}, nb_ts = 2)

    time_series = linear_combination([0.2, 0.2, 0.3], ts_linear + ts_exponential)
    
    # Plot
    plt.scatter(np.arange(0, max_interval_size, max_interval_size/samples),time_series)
    plt.show()
    print()