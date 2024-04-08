import numpy as np

class FrequencyModulation:
    def __init__(self):
        pass

    def modulate(self, time):
        raise NotImplementedError("Subclasses must implement modulate method")


class FrequencyModulationWithStep(FrequencyModulation):
    def __init__(self, freq_init):
        super().__init__()
        self.freq_init = freq_init

    def modulate(self, time, t_split):
        initial_period = 1 / self.freq_init
        t_split_adjusted = (t_split // initial_period) * initial_period
        return np.where(time > t_split_adjusted, self.freq_init * 2, self.freq_init)


class FrequencyModulationLinear(FrequencyModulation):
    def __init__(self, freq_init, slope):
        super().__init__(freq_init)
        self.slope = slope
        self.freq_init = freq_init

    def modulate(self, time):
        return self.freq_init + self.slope * time


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