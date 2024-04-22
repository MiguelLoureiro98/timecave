import numpy as np
import random
from tsvalidation.data_generation import time_series_functions as tsf

def linear_combination(weight_vector: list, ts_matrix: list) -> np.ndarray:

    nb_ts = len(ts_matrix)
    len_ts = len(ts_matrix[0])
    
    w = np.array(weight_vector)
    T = np.array(ts_matrix).T
    
    assert nb_ts == len(weight_vector)
    assert np.shape(T) == (len_ts, nb_ts )
    
    return np.matmul(T, w)


def generate_random_parameters(param_possibilities: dict, seed=1):
    random.seed(seed)
    params = {}
    for key, values in param_possibilities.items():
        if isinstance(values, tuple):
            value = random.choice(values)
        elif isinstance(values, list):
            value = random.uniform(values[0], values[1])
        else:
            value = values
        params[key] = value
    
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
    def __init__(self, functions,  length=100, noise_level=0.1, weights = None, parameter_values = None):
        self.functions = functions
        self.length = length
        self.noise_level = noise_level
        self.parameter_values = parameter_values
        self.weights = weights
        if weights == None:
            self.weights = np.ones(len(functions))
        self.time_series = np.zeros(length)

    def generate(self, seed=1):
        time_series = np.zeros(self.length)
        
        for i in range(len(self.functions)):
            # Generate random parameters within specified values
            parameters = generate_random_parameters(self.parameter_values[i], seed = seed)
            
            # Evaluate the function with the random parameters
            time_series += self.weights[i]*self.functions[i](self.length, **parameters)
        
        # Add noise
        noise = np.random.normal(scale=self.noise_level, size=self.length)
        time_series += noise

        self.time_series = time_series

        return time_series
    
    def viz(self):
        # Plot the generated time series
        plt.plot(self.time_series)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Generated Time Series')
        plt.show()

        return


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # Initialize TimeSeriesGenerator with the list of functions and parameter values
    generator = TimeSeriesGenerator([tsf.linear_ts, tsf.linear_ts], 
                                    parameter_values= [{'max_interval_size':1, 'slope':5, 'intercept':[5, 30]}, {'max_interval_size':1, 'slope':5, 'intercept':[5, 30]}],
                                    weights = [2,1])
    
    # Generate a time series
    time_series = generator.generate()

    generator.viz()
