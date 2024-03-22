import numpy as np
import matplotlib.pyplot as plt

def sinusoid_ts(length:int, frequency:float = 1, amplitude:float = 1, phase:float = 0) -> np.array:
    """
    Generate a sinusoidal time series.

    Parameters:
    - length (int): Length of the time series.
    - frequency (float, optional): Frequency of the sinusoidal wave in cycles per unit time. Default is 1.
    - amplitude (float, optional): Amplitude of the sinusoidal wave. Default is 1.
    - phase (float, optional): Phase shift of the sinusoidal wave in radians. Default is 0.

    Returns:
    - time_series (ndarray): Sinusoidal time series.
    """
    time = np.arange(length)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    time_series = sine_wave
    return time_series


def time_varying_sinusoid_ts(length: int, frequency: float = 1, amplitude: float = 1, phase: float = 0, frequency_slope: float = 0) -> np.array:
    """
    Generate a time-varying sinusoidal time series.

    Parameters:
    - length (int): Length of the time series.
    - frequency (float, optional): Frequency of the sinusoidal wave in cycles per unit time. Default is 1.
    - amplitude (float, optional): Amplitude of the sinusoidal wave. Default is 1.
    - phase (float, optional): Phase shift of the sinusoidal wave in radians. Default is 0.
    - phase_variation (float, optional): Variation of phase over time in radians. Default is 0.

    Returns:
    - time_series (ndarray): Sinusoidal time series.
    """
    time = np.linspace(0, 10, length)
    frequency = frequency + frequency_slope * time
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return time_series

def time_varying_sinusoid_ts(length: int, frequency_initial: float = 1, amplitude: float = 1, phase_initial: float = 0, t_split: float = 0) -> np.array:
    """
    Generate a time-varying sinusoidal time series.

    Parameters:
    - length (int): Length of the time series.
    - frequency (float, optional): Frequency of the sinusoidal wave in cycles per unit time. Default is 1.
    - amplitude (float, optional): Amplitude of the sinusoidal wave. Default is 1.
    - phase (float, optional): Phase shift of the sinusoidal wave in radians. Default is 0.
    - phase_variation (float, optional): Variation of phase over time in radians. Default is 0.

    Returns:
    - time_series (ndarray): Sinusoidal time series.
    """
    time = np.linspace(0, length, length)
    #fase = time % (2 * np.pi)
    period_phase = t_split/(1/frequency_initial) % 1
    frequency = np.where(time > t_split, frequency_initial*2, frequency_initial)
    phase = np.where(time > t_split, phase_initial + period_phase, phase_initial)
    
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return time_series


def indicator_ts(length: int, start_index:int, end_index:int):
    """
    Generate a time series based on an indicator function, also known as a characteristic function.

    Parameters:
    - length (int): Length of the indicator function.
    - start_index (int): Index where the indicator function starts being 1.
    - end_index (int): Index where the indicator function ends being 1.

    Returns:
    - indicator (ndarray): Indicator function array where elements are 1 between start_index and end_index (inclusive),
      and 0 elsewhere.
    """
    indicator = np.zeros(length)
    indicator[start_index:end_index + 1] = 1
    return indicator

def particular_indicator_ts(length, idx):
    """
    Generate a time series based on an indicator function, also known as a characteristic function.

    Parameters:
    - length (int): Length of the indicator function.
    - idx (int): Index where the indicator function starts being 1.

    Returns:
    - indicator (ndarray): Indicator function array where elements are 1 between start_index until the end of the series,
      and 0 elsewhere.
    """
    return indicator_ts(length, idx, length-1)

def unit_impulse_function(length, idx):
    """
    Generate a time series based on the dirac delta function, also known as the unit impulse.

    Parameters:
    - length (int): Length of the indicator function.
    - idx (int): Index where the indicator is 1.

    Returns:
    - indicator (ndarray): Indicator function array where elements are 1 at idx,
      and 0 elsewhere.
    """
    return indicator_ts(length, idx, idx)

def linear_ts(length, slope=1, intercept=0):
    """
    Generate a linear function.

    Parameters:
    - length (int): Length of the linear function.
    - slope (float, optional): Slope of the linear function. Default is 1.
    - intercept (float, optional): Intercept of the linear function. Default is 0.

    Returns:
    - linear_series (ndarray): Linear function array generated with the formula: slope * time + intercept.
    """
    time = np.arange(length)
    linear_series = slope * time + intercept
    return linear_series

def exponential_function(length, decay_rate=1, initial_value=1):
    """
    Generate an exponential function.

    Parameters:
    - length (int): Length of the exponential function.
    - decay_rate (float, optional): Rate of decay of the exponential function. Default is 1.
    - initial_value (float, optional): Initial value of the exponential function. Default is 1.

    Returns:
    - exponential_series (ndarray): Exponential function array generated with the formula: initial_value * exp(-decay_rate * time).
    """
    time = np.arange(length)
    exponential_series = initial_value * np.exp(-decay_rate * time)
    return exponential_series


if __name__=='__main__':
    #plt.plot(sinusoid_ts(100, frequency=0.05, amplitude=1, phase=np.pi/2), )
    time_series = time_varying_sinusoid_ts(100, frequency_initial=5, t_split=50, amplitude=1, phase_initial=0)
    plt.plot(time_series)
    #plt.plot(indicator_ts(100, 10, 20))
    plt.show()
    print()