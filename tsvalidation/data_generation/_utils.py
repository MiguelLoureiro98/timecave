import numpy as np
import random

def nonlin_func(nb, x):
        nonlin_x = {
            1: np.cos(x),
            2: np.sin(x),
            3: np.tanh(x),
            4: np.arctan(x),
            5: np.exp(-x/10000)
        }[nb]
        return nonlin_x


def get_arma_parameters(lags, max_root, seed =1):
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



def generate_seeds(seed, num_seeds):
    random.seed(seed)
    
    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
    
    return seeds

