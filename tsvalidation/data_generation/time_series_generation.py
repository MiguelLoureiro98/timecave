import numpy as np
import random
from tsvalidation.data_generation import time_series_functions as tsf
import inspect

def linear_combination(weight_vector: list, ts_matrix: list) -> np.ndarray:

    nb_ts = len(ts_matrix)
    len_ts = len(ts_matrix[0])
    
    w = np.array(weight_vector)
    T = np.array(ts_matrix).T
    
    assert nb_ts == len(weight_vector)
    assert np.shape(T) == (len_ts, nb_ts )
    
    return np.matmul(T, w)

def get_parameter_types(func):
    signature = inspect.signature(func)
    parameter_types = [param.annotation for param in signature.parameters.values()]
    return parameter_types

def generate_random_parameters(param_possibilities: list, seed=1):
    random.seed(seed)
    params = []
    for values in param_possibilities:
        if isinstance(values, tuple):
            value = random.choice(values)
        elif isinstance(values, list):
            value = random.uniform(values[0], values[1])
        else:
            value = values
        params.append(value)
    
    return params


def generate_from_func(number_samples, 
                       ts_func: callable, 
                       param_possibilities: dict, 
                       nb_ts: int, 
                       generate_param: callable = generate_random_parameters, 
                       seed = 1 ) -> np.ndarray:
    ts_list = []
    
    for _ in range(nb_ts):
        params = {}
        params = generate_param(param_possibilities, seed = seed)
        ts_list.append(ts_func(number_samples = number_samples, **params))
    
    return ts_list

import numpy as np


class TimeSeriesGenerator:
    def __init__(self, functions, length=100, noise_level=0.1, parameter_values=None):
        self.functions = functions
        self.length = length
        self.noise_level = noise_level
        self.parameter_values = parameter_values

    def generate(self, seed=1):
        time_series = np.zeros(self.length)
        
        for func in self.functions:
            # Generate random parameters within specified values
            parameters = generate_random_parameters(self.parameter_values[func.__name__])
            
            # Evaluate the function with the random parameters
            time_series += func(self.length, *parameters)
        
        # Add noise
        noise = np.random.normal(scale=self.noise_level, size=self.length)
        time_series += noise

        return time_series

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # Initialize TimeSeriesGenerator with the list of functions and parameter values
    generator = TimeSeriesGenerator([tsf.linear_ts], parameter_values={'linear_ts': [5, [5, 30]] })
    
    # Generate a time series
    time_series = generator.generate()

    # Plot the generated time series
    plt.plot(time_series)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Generated Time Series')
    plt.show()
