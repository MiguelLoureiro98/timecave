import numpy as np

class FrequencyModulation:
    def __init__(self):
        pass

    def modulate(self, time):
        raise NotImplementedError("Subclasses must implement modulate method")


class FrequencyModulationWithStep(FrequencyModulation):
    def __init__(self, freq_init, t_split):
        super().__init__()
        self.freq_init = freq_init
        self.t_split = t_split

    def modulate(self, time):
        initial_period = 1 / self.freq_init
        t_split_adjusted = (self.t_split // initial_period) * initial_period
        return np.where(time > t_split_adjusted, self.freq_init * 2, self.freq_init)


class FrequencyModulationLinear(FrequencyModulation):
    def __init__(self, freq_init, slope):
        super().__init__()
        self.slope = slope
        self.freq_init = freq_init

    def modulate(self, time):
        return self.freq_init + self.slope * time

