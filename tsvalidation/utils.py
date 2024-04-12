"""
This module contains utility functions to help the user.

Functions
---------
Nyquist_min_samples
heuristic_min_samples
true_test_indices
"""

import numpy as np

def Nyquist_min_samples(fs: float | int, freq_limit: float | int) -> int:
    
    """
    Compute the minimum number of samples each partition should have
    for the Nyquist theorem to be satisfied.

    This function computes the minimum length for each training and 
    validation partition, assuming the time series was sampled at
    a frequency of 'fs' Hertz and the largest frequency of interest
    for modelling purposes is 'freq_limit' Hertz. Additionally,
    the function computes the largest frequency that can be captured
    using 'fs' as the sampling frequency, as well as the smallest sampling 
    frequency that would be required to capture 'freq_limit'. Both of these 
    results are directly derived from the Nyquist sampling theorem.

    Parameters
    ----------
    fs : float | int
        The time series' sampling frequency (Hz).

    freq_limit : float | int
        Largest frequency of interest (Hz).

    Returns
    -------
    int
        Minimum number of samples required to capture freq_limit
        with a sampling frequency of fs, according to the Nyquist
        sampling theorem.

    Raises
    ------
    TypeError
        If either 'fs' or 'freq_limit' is neither a float nor an integer.

    ValueError
        If either 'fs' or 'freq_limit' are non-positive.

    Warning
        If the choice of 'fs' and 'freq_limit' does not satisfy the Nyquist sampling theorem.
    """

    _check_frequencies(fs, freq_limit);
    _check_Nyquist(fs, freq_limit);

    ts = 1 / fs;
    T_limit = 1 / freq_limit;
    t_final = 2 * T_limit;
    n_samples = int(np.ceil(t_final / ts));

    print("Nyquist theorem results");
    print("-----------------------");

    print(f"Maximum frequency that can be extracted using a sampling frequency of {fs} Hz : {fs/2} Hz");
    print(f"Sampling rate required to capture a frequency of {freq_limit} Hz : {2*freq_limit} Hz");
    print("------------------------------------------------------------------------------------------");
    print(f"Minimum number of samples required to capture a frequency of {freq_limit} Hz with a");
    print(f"sampling frequency of {fs} Hz: {n_samples} samples");

    return n_samples;

def heuristic_min_samples(fs: float | int, freq_limit: float | int) -> dict:
    
    """
    Compute the minimum number of samples each partition should have
    according to the 10 / 20 sampling heuristic.

    This function computes the minimum and maximum lengths for each 
    training and validation partition, assuming the time series was 
    sampled at a frequency of 'fs' Hertz and the largest frequency 
    of interest for modelling purposes is 'freq_limit' Hertz. The
    interval in which the sampling frequency should lie for 'freq_limit' 
    to be effectively captured is also derived. The 10 / 20 sampling 
    heuristic is used to derive both results.

    Parameters
    ----------
    fs : float | int
        The time series' sampling frequency (Hz).

    freq_limit : float | int
        Largest frequency of interest (Hz).

    Returns
    -------
    dict
        Minimum and maximum number of samples (Min_samples and Max_samples, respectively)
        required to capture freq_limit with a sampling frequency of fs, 
        according to the 10 / 20 heuristic rule.

    Raises
    ------
    TypeError
        If either 'fs' or 'freq_limit' is neither a float nor an integer.

    ValueError
        If either 'fs' or 'freq_limit' are non-positive.

    Warning
        If the choice of 'fs' and 'freq_limit' does not abide by the 10 / 20 heuristic.
    """

    _check_frequencies(fs, freq_limit);
    _check_heuristic(fs, freq_limit);

    ts = 1 / fs;
    T_limit = 1 / freq_limit;
    t_lower = 10 * T_limit;
    t_upper = 20 * T_limit;
    n_lower = int(np.ceil(t_lower / ts));
    n_upper = int(np.ceil(t_upper / ts));

    print("10 / 20 sampling heuristic results");
    print("----------------------------------");
    
    print(f"Minimum sampling rate required to capture a frequency of {freq_limit} Hz : {10*freq_limit} Hz");
    print(f"Maximum sampling rate required to capture a frequency of {freq_limit} Hz : {20*freq_limit} Hz");
    print("----------------------------------------------------------------------------------------------");
    print(f"Capturing a frequency of {freq_limit} Hz with a sampling frequency of {fs} Hz would require:");
    print(f"{n_lower} to {n_upper} samples");

    return {"Min_samples": n_lower, "Max_samples": n_upper};

def _check_frequencies(fs: float | int, freq_limit: float | int) -> None:

    """
    Perform type and value checks on frequencies. Raises a TypeError if the frequencies are neither floats nor integers.
    If either frequency is not positive, a ValueError is raised.
    """

    if((isinstance(fs, float) or isinstance(fs, int)) is False or (isinstance(freq_limit, float) or isinstance(freq_limit, int)) is False):

        raise TypeError("Both 'fs' and 'freq_limit' should be either integers or floats.");

    if(fs <= 0 or freq_limit <= 0):

        raise ValueError("Frequencies should be non-negative.");

    return;

def _check_Nyquist(fs: float | int, freq_limit: float | int) -> None:

    """
    Check whether 'fs' and 'freq_limit' satisfy the Nyquist theorem.
    """

    if(fs < 2 * freq_limit):

        raise Warning("According to the Nyquist theorem, the selected frequency cannot be captured using this sampling frequency.");

    return;

def _check_heuristic(fs: float | int, freq_limit: float | int) -> None:
    
    """
    Check whether 'fs' and 'freq_limit' ... .
    """

    if(fs < 10 * freq_limit or fs > 20 * freq_limit):

        raise Warning("This choice of sampling frequency and frequency of interest is not compliant with the 10 / 20 sampling heuristic.");

    return;

def true_test_indices(test_ind: np.ndarray, model_order: int) -> np.ndarray:
    
    """
    Modify an array of validation indices for modelling purposes.

    This function modifies the array of validation indices yielded by a 
    splitter so that it includes the previous time steps that should be passed 
    as inputs to the model in order for it to predict the series' next value.

    Parameters
    ----------
    test_ind : np.ndarray
        Array of test (validation, really) indices yielded by a splitter.

    model_order : int
        The number of previous time steps the model needs to take as input in order
        to predict the series' value at the next time step.

    Returns
    -------
    np.ndarray
        Array of validation indices including the time steps required by the model
        to predict the series' value at the first validation index.
    """

    new_ind = np.arange(test_ind[0] - model_order, test_ind[0]);
    full_test_ind = np.hstack((new_ind, test_ind));
    
    return full_test_ind;