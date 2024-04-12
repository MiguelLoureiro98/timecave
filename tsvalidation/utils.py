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
    """

    _check_frequencies(fs, freq_limit);

    ts = 1 / fs;
    T_limit = 1 / freq_limit;
    t_final = 2 * T_limit;
    n_samples = np.ceil(t_final / ts, dtype=np.int128);

    print("Nyquist theorem results \
           -----------------------");

    print(f"Maximum frequency that can be extracted using a sampling frequency of {fs} Hz : {fs/2} Hz \
            Sampling rate required to capture a frequency of {freq_limit} Hz : {2*freq_limit} Hz \
            ----------------------------------------------------------------------------------------- \
            Minimum number of samples required to capture a frequency of {freq_limit} Hz with a \
            sampling frequency of {fs} Hz: {n_samples} samples");

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
    """

    _check_frequencies(fs, freq_limit);

    ts = 1 / fs;
    T_limit = 1 / freq_limit;
    t_lower = 10 * T_limit;
    t_upper = 20 * T_limit;
    n_lower = np.ceil(t_lower / ts, dtype=np.int128);
    n_upper = np.ceil(t_upper / ts, dtype=np.int128);

    print("10 / 20 sampling heuristic results \
           ----------------------------------");
    
    print(f"Minimum sampling rate required to capture a frequency of {freq_limit} Hz : {10*freq_limit} Hz \
            Maximum sampling rate required to capture a frequency of {freq_limit} Hz : {20*freq_limit} Hz \
            --------------------------------------------------------------------------------------------- \
            Capturing a frequency of {freq_limit} Hz with a sampling frequency of {fs} Hz would require: \
            {n_lower} to {n_upper} samples");

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

def true_test_indices(test_ind: np.ndarray, model_order: int) -> np.ndarray:
    
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    test_ind : np.ndarray
        Array of test (validation, really) indices yielded by a splitter.

    model_order : int
        _description_

    Returns
    -------
    np.ndarray
        Array of validation indices including ... .
    """

    new_ind = np.arange(test_ind[0] - model_order, test_ind[0]);
    full_test_ind = np.hstack((new_ind, test_ind));
    
    return full_test_ind;