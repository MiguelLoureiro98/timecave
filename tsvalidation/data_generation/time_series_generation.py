import numpy as np
import random
from tsvalidation.data_generation import time_series_functions as tsf
import matplotlib.pyplot as plt

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


def generate_seeds(seed, num_seeds):
    random.seed(seed)
    
    # Generate array of seeds
    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
    
    return seeds


class TimeSeriesGenerator:
    def __init__(self, functions,  length=100, noise_level=0.1, weights = None, parameter_values = None):
        assert len(functions) == len(parameter_values)
        self.functions = functions
        self.length = length
        self.noise_level = noise_level
        self.parameter_values = parameter_values
        self.weights = weights
        if weights == None:
            self.weights = np.ones(len(functions))
        self.time_series = []

    def generate(self, nb_sim, og_seed=1):
        
        seeds = generate_seeds(og_seed, nb_sim)

        for seed in seeds:
            np.random.seed(seed)
            ts = np.zeros(self.length) 
            for i in range(len(self.functions)):
                parameters = generate_random_parameters(self.parameter_values[i], seed = seed)
                ts += self.weights[i]*self.functions[i](self.length, **parameters)

            ts += np.random.normal(scale=self.noise_level, size=self.length)

            self.time_series.append(ts)

        return self.time_series
    
    def plot(self, indexes=None):
            if indexes is None:
                indexes = range(len(self.time_series))
            elif isinstance(indexes, int):
                indexes = [indexes]
            
            for idx in indexes:
                plt.plot(self.time_series[idx], label=f'Time Series {idx}')
            
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Generated Time Series')
            plt.legend()
            plt.show()


# Example usage
if __name__ == "__main__":
    from tsvalidation.data_generation import utils as dgu

    linear_parameters = {
        'max_interval_size':1, 
        'slope':5, 
        'intercept':[5, 30] 
        }

    exp_parameters = {
        'max_interval_size': (1, 2),
        'decay_rate': [1, 25],
        'initial_value': [1, 25]
        }

    sin_parameters = {
        'max_interval_size': (1, 2), 
        'amplitude':[1,3],
        'frequency':(dgu.FrequencyModulationLinear(1,20), dgu.FrequencyModulationWithStep(10,0.8))
        }

    impulse_parameters = {
        'idx': (500, 600),
        'constant': [5, 10]
        }

    indicator_parameters = {
        'start_index': (700, 600), 
        'end_index': (800, 900)
        }
    generator = TimeSeriesGenerator(
        length = 1000,
        noise_level=0.2,
        functions = [tsf.linear_ts, tsf.indicator_ts ,tsf.frequency_varying_sinusoid_ts, tsf.scaled_unit_impulse_function_ts, tsf.exponential_ts ], 
        parameter_values = [linear_parameters, indicator_parameters, sin_parameters, impulse_parameters, exp_parameters]
        )
    generator.generate(10)
    generator.plot()
