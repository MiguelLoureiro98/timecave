import numpy as np
import matplotlib.pyplot as plt


def sinusoid_ts(max_interval_size, number_samples, amplitude=1, frequency=1, phase=0):
    """
    Generate a sinusoidal wave based on given parameters.

    Parameters:
        length (float): Length of the window of values starting at 0.
        number_samples (int): Number of samples to generate.
        amplitude (float): Amplitude of the sinusoid. Default is 1.
        frequency (float): Frequency of the sinusoid in Hz. Default is 1.
        phase (float): Phase angle of the sinusoid in radians. Default is 0.

    Returns:
        numpy.ndarray: Array of generated sinusoidal values.
    """
    # Calculate the time values
    time_values = np.linspace(0, max_interval_size, number_samples)

    # Generate the sinusoid
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * time_values + phase)

    return sinusoid

def frequency_modulation_with_step(time, t_split, freq_init):
    initial_period = (1/freq_init)
    t_split_adjusted = (t_split // initial_period)* initial_period
    return np.where(time > t_split_adjusted, freq_init*2, freq_init)

def time_varying_sinusoid_ts(max_interval_size: float, 
                             number_samples: int, 
                             frequency_initial: float = 1, 
                             amplitude: float = 1, 
                             phase_initial: float = 0, 
                             t_split: float = 0) -> np.array:
    """
    Generate a time series of a sinusoid with varying frequency.

    Parameters:
    - max_interval_size (float): The maximum interval size for the time series.
    - number_samples (int): The number of samples in the time series.
    - frequency_initial (float, optional): Initial frequency of the sinusoid. Default is 1.
    - amplitude (float, optional): Amplitude of the sinusoid. Default is 1.
    - phase_initial (float, optional): Initial phase of the sinusoid in radians. Default is 0.
    - t_split (float, optional): Time point at which the frequency of the sinusoid changes. Default is 0.

    Returns:
    - np.array: Time series data representing the sinusoid with varying frequency.
    """

    time = np.linspace(0, max_interval_size, number_samples)
    initial_period = (1/frequency_initial)
    t_split_adjusted = (t_split // initial_period)* initial_period
    frequency = np.where(time > t_split_adjusted, frequency_initial*2, frequency_initial)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase_initial)
    return time_series


def time_varying_sinusoid_ts(max_interval_size: float, 
                             number_samples: int,
                             frequency_func: callable,  
                             frequency_args: tuple = (),  
                             amplitude: float = 1, 
                             phase_initial: float = 0) -> np.array:
    """
    Generate a time series of a sinusoid with varying frequency.

    Parameters:
    - max_interval_size (float): The maximum interval size for the time series.
    - number_samples (int): The number of samples in the time series.
    - frequency_initial (float, optional): Initial frequency of the sinusoid. Default is 1.
    - amplitude (float, optional): Amplitude of the sinusoid. Default is 1.
    - phase_initial (float, optional): Initial phase of the sinusoid in radians. Default is 0.
    - t_split (float, optional): Time point at which the frequency of the sinusoid changes. Default is 0.

    Returns:
    - np.array: Time series data representing the sinusoid with varying frequency.
    """

    time = np.linspace(0, max_interval_size, number_samples)
    frequency = frequency_func(time,*frequency_args)
    time_series = amplitude * np.sin(2 * np.pi * frequency * time + phase_initial)
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
    max_interval_size = 1
    samples = 1000
    sample_interval = max_interval_size/samples
    #time_series = time_varying_sinusoid_ts(max_interval_size, samples, frequency_initial=5, t_split=0.5, amplitude=1, phase_initial=0.5)
    #time_series = sinusoid_ts(max_interval_size, samples, frequency=5, amplitude=1, phase=0)
    time_series = time_varying_sinusoid_ts(max_interval_size, 
                             samples,
                             amplitude = 5, 
                             frequency_func = frequency_modulation_with_step,
                             frequency_args = (0.5, 5),
                             phase_initial = 0,
                             )
    plt.scatter(np.arange(0,max_interval_size,max_interval_size/samples),time_series)
    #plt.plot(indicator_ts(100, 10, 20))
    plt.show()
    print()