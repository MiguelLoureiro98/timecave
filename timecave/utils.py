"""
This module contains utility functions to help the users make the most of their data. \
More specifically, it provides routines to aid users in [on / with] their [during the] data collection process and \
their validation procedures.

Functions
---------
Nyquist_min_samples
    Computes the minimum amount of samples needed to capture a frequency of interest using the Nyquist theorem.

heuristic_min_samples
    Computes the minimum amount of samples needed to capture a frequency of interest using an heuristic algorithm.

true_test_indices
    Readies an array of validation indices for insertion into a model.
"""

import numpy as np

def Nyquist_min_samples(fs: float | int, freq_limit: float | int) -> int:
    
    """
    Compute the minimum number of samples the series should have
    for the Nyquist theorem to be satisfied.

    This function computes the minimum series length for capturing a given frequency,
    assuming the time series was sampled at
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

    See also
    --------
    [heuristic_min_samples](heuristic.md):
        Performs the same computations using an heuristic rule.

    Notes
    -----
    The Nyquist sampling theorem is a fundamental result in digital signal processing. It states that, \
    for one to be able to reconstruct a continuous-time signal from its discrete counterpart, the sampling \
    frequency should be at least twice as high as the largest frequency of interest [the signal should be sampled at a rate / frequency at least twice as high as ...].
    Mathematically speaking, the sampling frequency should be [is] given by:
    
    $$
    f_s >= 2 \cdot f
    $$
    
    where $f_s$ is the sampling frequency and $f$ is the frequency of interest. \
    The Nyquist sampling theorem is discussed in several reference books, of which [[1]](#1) is but an example.

    Since time series are essentially signals, the minimum number of samples that need to be collected if one needs [is] to capture a given frequency [if a given frequency is to be captured] can be computed \
    using the Nyquist theorem.

    References
    ----------
    ##1
    A. Oppenheim, R. Schafer, and J. Buck. Discrete-Time Signal Processing.
    Prentice Hall, 1999.

    Examples
    --------
    >>> from timecave.utils import Nyquist_min_samples
    >>> n_samples = Nyquist_min_samples(100, 20);
    Nyquist theorem results
    -----------------------
    Maximum frequency that can be extracted using a sampling frequency of 100 Hz : 50.0 Hz
    Sampling rate required to capture a frequency of 20 Hz : 40 Hz
    ------------------------------------------------------------------------------------------
    Minimum number of samples required to capture a frequency of 20 Hz with a
    sampling frequency of 100 Hz: 10 samples
    >>> n_samples
    10

    If the frequency of interest cannot be captured using the sampling frequency provided \
    by the user according to the Nyquist theorem, an exception is thrown:

    >>> samples = Nyquist_min_samples(1, 2);
    Traceback (most recent call last):
    ...
    Warning: According to the Nyquist theorem, the selected frequency cannot be captured using this sampling frequency.

    If negative frequencies are passed, or if their values are neither integers nor floats, exceptions are thrown as well:

    >>> samples = Nyquist_min_samples(-2, 1);
    Traceback (most recent call last):
    ...
    ValueError: Frequencies should be non-negative.
    >>> samples = Nyquist_min_samples(1, "a");
    Traceback (most recent call last):
    ...
    TypeError: Both 'fs' and 'freq_limit' should be either integers or floats.
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
    Compute the minimum number of samples the series should have
    according to the 10 / 20 sampling heuristic.

    This function computes the minimum and maximum lengths for capturing a given frequency, 
    assuming the time series was 
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

    See also
    --------
    [Nyquist_min_samples](nyquist.md):
        Performs the same computations using the Nyquist theorem.
    
    Notes
    -----
    Under certain circumstances, the conditions of the Nyquist theorem might not be enough to guarantee that the reconstruction of the signal is possible.
    [To address this isssue,] A heuristic has been developed in the field of control engineering, whereby [according to which] the sampling frequency should be 10 to 20 times higher than the largest \
    frequency of interest:
    
    $$
    10 \cdot f <= f_s <= 20 \cdot f
    $$

    Theoretically, the higher the sampling frequency, the better (i.e. the easier it can be to reconstruct the original signal), \
    though hardware limitations naturally come into play here.

    Examples
    --------
    >>> from timecave.utils import heuristic_min_samples
    >>> n_samples = heuristic_min_samples(150, 10);
    10 / 20 sampling heuristic results
    ----------------------------------
    Minimum sampling rate required to capture a frequency of 10 Hz : 100 Hz
    Maximum sampling rate required to capture a frequency of 10 Hz : 200 Hz
    ----------------------------------------------------------------------------------------------
    Capturing a frequency of 10 Hz with a sampling frequency of 150 Hz would require:
    150 to 300 samples
    >>> n_samples
    {'Min_samples': 150, 'Max_samples': 300}

    If the frequency of interest cannot be captured using the sampling frequency provided \
    by the user according to the heuristic, an exception is thrown:

    >>> samples = heuristic_min_samples(80, 10);
    Traceback (most recent call last):
    ...
    Warning: This choice of sampling frequency and frequency of interest is not compliant with the 10 / 20 sampling heuristic.

    If negative frequencies are passed, or if their values are neither integers nor floats, exceptions are thrown as well:

    >>> samples = heuristic_min_samples(-2, 1);
    Traceback (most recent call last):
    ...
    ValueError: Frequencies should be non-negative.
    >>> samples = heuristic_min_samples(1, "a");
    Traceback (most recent call last):
    ...
    TypeError: Both 'fs' and 'freq_limit' should be either integers or floats.
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

    Raises
    ------
    TypeError
        If 'model_order' is not an integer.

    ValueError
        If 'model_order' is not positive.

    ValueError
        If 'model_order' is larger than the amount of samples in the training set, assuming it precedes the validation set.

    Examples
    --------
    >>> from timecave.utils import true_test_indices
    >>> test_indices = np.array([8, 9, 10]);
    >>> model_order = 2;
    >>> true_test_indices(test_indices, model_order)
    array([ 6,  7,  8,  9, 10])

    The order of a model must be an integer value:

    >>> true_test_indices(test_indices, 0.5)
    Traceback (most recent call last):
    ...
    TypeError: 'model_order' should be an integer.

    The order of a model must not be a negative value:

    >>> true_test_indices(test_indices, -1)
    Traceback (most recent call last):
    ...
    ValueError: 'model_order' should be positive.

    This function assumes training and validation are done sequentially. \
    Therefore, an exception will be thrown if the amount of samples preceding the validation \
    set is smaller than the order of the model:

    >>> true_test_indices(test_indices, 10)
    Traceback (most recent call last):
    ...
    ValueError: 'model_order' should be smaller than the amount of samples in the training set.
    """

    _check_order(model_order);
    _check_ind(test_ind, model_order);

    new_ind = np.arange(test_ind[0] - model_order, test_ind[0]);
    full_test_ind = np.hstack((new_ind, test_ind));
    
    return full_test_ind;

def _check_order(model_order: int) -> None:

    """
    Perform type and value checks on the 'model_order' parameter.
    """

    if(isinstance(model_order, int) is False):

        raise TypeError("'model_order' should be an integer.");

    if(model_order <= 0):

        raise ValueError("'model_order' should be positive.");

    return;

def _check_ind(test_ind: np.ndarray, model_order: int) -> None:

    """
    Check whether the order of model is smaller than the amount of samples on the training set (assuming it precedes the validation set).
    """

    if(test_ind[0] < model_order):

        raise ValueError("'model_order' should be smaller than the amount of samples in the training set.");

    return;

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);