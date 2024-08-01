"""
This module contains several metrics to evaluate the performance of model validation methods.

Functions
---------
PAE
    Implements the Predictive Accuracy Error metric.

APAE
    Implements the Absolute Predictive Accuracy Error metric.

RPAE
    Implements the Relative Predictive Accuracy Error metric.

RAPAE
    Implements the Relative Absolute Predictive Accuracy Error metric.

sMPAE
    Implements the symmetric Mean Predictive Accuracy Error metric.

MC_metric
    Statistical summary for Monte Carlo experiments regarding validation methods.

under_over_estimation
    Separate statistical summaries for [the] underestimation and overestimation cases.

Notes
-----
- PAE and APAE are absolute metrics. Their values may range from $-L_m$ to $\infty$ and from $0$ to $\infty$, respectively. \
These should not be used to compare results obtained with different models or using different time series.
- RPAE and RAPAE are relative metrics, as they measure how large the validation error is with respect to the true (test) error, thus eliminating the latter's influence on the metric. \
Their values lie in the $[-1, \infty]$ and $[0, \infty]$ intervals, respectively. \
These can be used to compare results for different models and/or time series.
- sMPAE is a scaled, symmetric version of the PAE. It can be used to compare results for different models and/or time series.
"""

import numpy as np

def PAE(estimated_error: float | int, test_error: float | int) -> float:

    """
    Compute the Predictive Accuracy Error (PAE).

    This function computes the PAE metric. Both the estimated (i.e. validation) error
    and the test error must be passed as parameters.

    Parameters
    ----------
    estimated_error : float | int
        Validation error.

    test_error : float | int
        True (i.e. test) error.

    Returns
    -------
    float
        Predictive Accuracy Error.

    See also
    --------
    [APAE](apae.md):
        Absolute Predictive Accuracy Error.

    [RPAE](rpae.md): 
        Relative Predictive Accuracy Error.

    [RAPAE](rapae.md):
        Relative Absolute Predictive Accuracy Error.

    [sMPAE](smpae.md):
        Symmetric Mean Predictive Accuracy Error.

    Notes
    -----
    The Predictive Accuracy Error is defined as the difference between the estimate of a model's error given by a validation method\
    and the model's true error:
    
    $$
    PAE = \hat{L}_m - L_m
    $$ 
    
    One can infer from the sign [The sign is important / relevant to determine ...] whether the validation method is overestimating or underestimating the model's true error:\
    a negative value denotes an underestimation, while a positive value corresponds to an overestimation.\
    
    Note that, in all likelihood, the true error will not be known. It is usually estimated using an independent test set. For more details, please refer to [[1]](#1).

    References
    ----------
    ##1
    Cerqueira, V., Torgo, L., Mozetiˇc, I., 2020. Evaluating time series forecasting
    models: An empirical study on performance estimation methods.
    Machine Learning 109, 1997–2028.

    Examples
    --------
    >>> from timecave.validation_strategy_metrics import PAE
    >>> PAE(10, 3)
    7
    >>> PAE(1, 5)
    -4
    >>> PAE(8, 8)
    0
    """

    return estimated_error - test_error;

def APAE(estimated_error: float | int, test_error: float | int) -> float:
    
    """
    Compute the Absolute Predictive Accuracy Error (APAE).

    This function computes the APAE metric. Both the estimated (i.e. validation) error
    and the test error must be passed as parameters.

    Parameters
    ----------
    estimated_error : float | int
        Validation error.

    test_error : float | int
        True (i.e. test) error.

    Returns
    -------
    float
        Absolute Predictive Accuracy Error.

    See also
    --------
    [PAE](pae.md):
        Predictive Accuracy Error.

    [RPAE](rpae.md): 
        Relative Predictive Accuracy Error.

    [RAPAE](rapae.md):
        Relative Absolute Predictive Accuracy Error.

    [sMPAE](smpae.md):
        Symmetric Mean Predictive Accuracy Error.

    Notes
    -----
    The Absolute Predictive Accuracy Error is defined as the absolute value of the difference between the \
    estimate of a model's error given by a validation method \
    and the model's true error. In other words, it is the absolute value of the Predictive Accuracy Error:
    
    $$
    APAE = |\hat{L}_m - L_m| = |PAE|
    $$ 
    
    Since the APAE is always non-negative, this metric does not measure [cannot be used to determine] whether the validation method is overestimating or underestimating\
    the model's true error.
    
    Note that, in all likelihood, the true error will not be known. It is usually estimated using an independent test set. For more details, please refer to [[1]](#1).

    References
    ----------
    ##1
    Cerqueira, V., Torgo, L., Mozetiˇc, I., 2020. Evaluating time series forecasting
    models: An empirical study on performance estimation methods.
    Machine Learning 109, 1997–2028.

    Examples
    --------
    >>> from timecave.validation_strategy_metrics import APAE
    >>> APAE(10, 3)
    7
    >>> APAE(1, 5)
    4
    >>> APAE(8, 8)
    0
    """

    return abs(estimated_error - test_error);

def RPAE(estimated_error: float | int, test_error: float | int) -> float:
    
    """
    Compute the Relative Predictive Accuracy Error (RPAE).

    This function computes the RPAE metric. Both the estimated (i.e. validation) error
    and the test error must be passed as parameters.

    Parameters
    ----------
    estimated_error : float | int
        Validation error.

    test_error : float | int
        True (i.e. test) error.

    Returns
    -------
    float
        Relative Predictive Accuracy Error.

    Raises
    ------
    ValueError
        If 'test_error' is zero.

    See also
    --------
    [PAE](pae.md):
        Predictive Accuracy Error.

    [APAE](apae.md): 
        Absolute Predictive Accuracy Error.

    [RAPAE](rapae.md):
        Relative Absolute Predictive Accuracy Error.

    [sMPAE](smpae.md):
        Symmetric Mean Predictive Accuracy Error.

    Notes
    -----
    The Relative Predictive Accuracy Error is obtained by dividing the Predictive Accuracy Error (PAE) by the model's true error:
    
    $$
    RPAE = \\frac{\hat{L}_m - L_m}{L_m} = \\frac{PAE}{L_m}
    $$ 
    
    [By doing so, the metric is made ... .]
    This makes this metric scale-independent with respect to the model's true error, which in turn makes it useful for comparing validation methods \
    [that have been applied on different ...] across different time series and/or forecasting models. Since this is essentially a scaled version of the PAE, \
    the sign retains its significance (negative sign for underestimation, positive sign for overestimation). \
    However, it should be noted that the RPAE is asymmetric: in case of an underestimation, its values will be contained in the interval of $[-1, 0[$; if the error is \
    overestimated, however, the RPAE can take any value in the range of $]0, \infty[$. A value of zero denotes a perfect estimate.

    Note that, in all likelihood, the true error will not be known. It is usually estimated using an independent test set.

    Examples
    --------
    >>> from timecave.validation_strategy_metrics import RPAE
    >>> RPAE(15, 5)
    2.0
    >>> RPAE(1, 5)
    -0.8
    >>> RPAE(8, 8)
    0.0

    If the true error is zero, the metric is undefined:

    >>> RPAE(5, 0)
    Traceback (most recent call last):
    ...
    ValueError: The test error is zero. RPAE is undefined.
    """

    if(test_error == 0):

        raise ValueError("The test error is zero. RPAE is undefined.");

    return (estimated_error - test_error) / test_error;

def sMPAE(estimated_error: float | int, test_error: float | int) -> float:
    
    """
    Compute the symmetric Mean Predictive Accuracy Error (sMPAE).

    This function computes the sMPAE metric. Both the estimated (i.e. validation) error
    and the test error must be passed as parameters.

    Parameters
    ----------
    estimated_error : float | int
        Validation error.

    test_error : float | int
        True (i.e. test) error.

    Returns
    -------
    float
        Symmetric Mean Predictive Accuracy Error.

    Raises
    ------
    ValueError
        If 'test_error' is zero.

    See also
    --------
    [PAE](pae.md):
        Predictive Accuracy Error.

    [APAE](apae.md):
        Absolute Predictive Accuracy Error.

    [RPAE](rpae.md):
        Relative Predictive Accuracy Error.

    [RAPAE](rapae.md):
        Relative Absolute Predictive Accuracy Error.

    Notes
    -----
    The symmetric Mean Predictive Accuracy Error is obtained by dividing the Predictive Accuracy Error (PAE) \
    by half the sum of the absolute values of both the error estimate and the true error:

    $$
    sMPAE = 2 \cdot \\frac{(\hat{L}_m - L_m)}{|\hat{L}_m| + |L_m|} = 2 \cdot \\frac{PAE}{|\hat{L}_m| + |L_m|}
    $$

    Similarly to the Relative Predictive Accuracy Error (RPAE), this metric can be seen as a scaled version of \
    the PAE. Unlike the RPAE, however, the sMPAE is symmetric, as all possible values lie in the interval of $[-2, 2]$. If the \
    error estimate is equal to the true error (perfect estimation), the sMPAE is zero. \
    Since this metric is based on the PAE, the sign retains its significance (negative sign for underestimation, positive sign for overestimation).

    Note that, in all likelihood, the true error will not be known. It is usually estimated using an independent test set.

    Examples
    --------
    >>> from timecave.validation_strategy_metrics import sMPAE
    >>> sMPAE(3, 2)
    0.4
    >>> sMPAE(3, 5)
    -0.5
    >>> sMPAE(5, 5)
    0.0
    >>> sMPAE(5, 0)
    2.0
    >>> sMPAE(0, 5)
    -2.0

    If both the true error and the estimated error are zero, this metric is undefined:

    >>> sMPAE(0, 0)
    Traceback (most recent call last):
    ...
    ValueError: sMPAE is undefined.
    """

    if((abs(estimated_error) + abs(test_error)) == 0):

        raise ValueError("sMPAE is undefined.");

    return 2*(estimated_error - test_error) / (abs(estimated_error) + abs(test_error));



def RAPAE(estimated_error: float | int, test_error: float | int) -> float:
    
    """
    Compute the Relative Absolute Predictive Accuracy Error (RAPAE).

    This function computes the RAPAE metric. Both the estimated (i.e. validation) error
    and the test error must be passed as parameters.

    Parameters
    ----------
    estimated_error : float | int
        Validation error.

    test_error : float | int
        True (i.e. test) error.

    Returns
    -------
    float
        Relative Absolute Predictive Accuracy Error.

    Raises
    ------
    ValueError
        If 'test_error' is zero.

    See also
    --------
    [PAE](pae.md):
        Predictive Accuracy Error.

    [APAE](apae.md): 
        Absolute Predictive Accuracy Error.

    [RPAE](rpae.md):
        Relative Predictive Accuracy Error.

    [sMPAE](smpae.md):
        Symmetric Mean Predictive Accuracy Error.    

    Notes
    -----
    The Relative Absolute Predictive Accuracy Error is defined as the Absolute Predictive Accuracy Error (APAE) divided by the \
    model's true error. It can also be seen as the absolute value of the Relative Predictive Accuracy Error (RPAE):
    
    $$
    RAPAE = \\frac{|\hat{L}_m - L_m|}{L_m} = \\frac{|PAE|}{L_m} = \\frac{APAE}{L_m} = |RPAE|
    $$ 
    
    This metric essentially takes the absolute value of the RPAE, and can be used in a similar fashion. However, since it uses the \
    absolute value, it cannot be used to determine whether a validation method is overestimating or underestimating the model's \
    true error. Like the RPAE, it is an asymmetric measure.

    Note that, in all likelihood, the true error will not be known. It is usually estimated using an independent test set.

    Examples
    --------
    >>> from timecave.validation_strategy_metrics import RAPAE
    >>> RAPAE(15, 5)
    2.0
    >>> RAPAE(1, 5)
    0.8
    >>> RAPAE(8, 8)
    0.0

    If the true error is zero, the metric is undefined:

    >>> RAPAE(5, 0)
    Traceback (most recent call last):
    ...
    ValueError: The test error is zero. RAPAE is undefined.
    """

    if(test_error == 0):

        raise ValueError("The test error is zero. RAPAE is undefined.");

    return abs(estimated_error - test_error) / test_error;

def MC_metric(estimated_error_list: list[float | int], test_error_list: list[float | int], metric: callable) -> dict:
    
    """
    Compute validation strategy metrics for N different experiments (MC stands for Monte Carlo).

    This function processes the results of a Monte Carlo experiment and outputs [relevant statistics...] a statistical summary of the results. \
    This [it] can be useful if one needs to analyse the performance of a given validation method on several different time series or using different models.  
    Users may provide a custom metric if they so desire, but it must have the same function signature as the metrics provided by this package.

    Parameters
    ----------
    estimated_error_list : list[float  |  int]
        List of estimated (i.e. validation) errors, one for each experiment / trial.

    test_error_list : list[float  |  int]
        List of test errors, one for each experiment / trial.

    metric : callable
        Validation strategy metric.

    Returns
    -------
    dict
        A statistical summary of the results.

    Raises
    ------
    ValueError
        If the estimator error list and the test error list differ in length.

    See also
    --------
    [under_over_estimation](under_over.md):
        Computes separate statistics for overestimation and underestimation cases.

    Examples
    --------
    >>> from timecave.validation_strategy_metrics import PAE, MC_metric
    >>> true_errors = [10, 30, 10, 50];
    >>> validation_errors = [20, 20, 50, 30];
    >>> MC_metric(validation_errors, true_errors, PAE)
    {'Mean': 5.0, 'Median': 0.0, '1st_Quartile': -12.5, '3rd_Quartile': 17.5, 'Minimum': -20.0, 'Maximum': 40.0, 'Standard_deviation': 22.9128784747792}

    If the lengths of the estimated error and test error lists do not match, an exception is thrown:

    >>> MC_metric(validation_errors, [10], PAE)
    Traceback (most recent call last):
    ...
    ValueError: The estimated error and test error lists must have the same length.
    """

    if(len(estimated_error_list) != len(test_error_list)):

        raise ValueError("The estimated error and test error lists must have the same length.");

    metric_array = np.zeros(len(estimated_error_list));

    for ind, (val_error, test_error) in enumerate(zip(estimated_error_list, test_error_list)):

        metric_array[ind] = metric(val_error, test_error);
    
    mean = metric_array.mean();
    minimum = metric_array.min();
    maximum = metric_array.max();
    median = np.median(metric_array);
    Q1 = np.quantile(metric_array, 0.25);
    Q3 = np.quantile(metric_array, 0.75);
    std = metric_array.std();

    results = {"Mean": mean,
               "Median": median,
               "1st_Quartile": Q1,
               "3rd_Quartile": Q3,
               "Minimum": minimum,
               "Maximum": maximum,
               "Standard_deviation": std};

    return results;

def under_over_estimation(estimated_error_list: list[float | int], test_error_list: list[float | int], metric: callable) -> tuple[dict]:
    
    """
    Compute separate validation strategy metrics for underestimation and overestimation instances (for N different experiments).

    This function processes the results of a Monte Carlo experiment and outputs two separate
    sets of summary statistics: one for cases where the true error is underestimated, and another one for cases 
    where the validation method overestimates the error.
    This [it] can be useful if one needs to analyse the performance of a given validation method on several different time series or using different models.  
    Users may provide a custom metric if they so desire, but it must have the same function signature as the metrics provided by this package.

    Parameters
    ----------
    estimated_error_list : list[float  |  int]
        List of estimated (i.e. validation) errors, one for each experiment / trial.

    test_error_list : list[float  |  int]
        List of test errors, one for each experiment / trial.

    metric : callable
        Validation strategy metric.

    Returns
    -------
    tuple[dict]
        [Separate] Statistical summaries for the overestimation and underestimation cases. \
        The first dictionary is for the underestimation cases.

    Raises
    ------
    ValueError
        If the estimator error list and the test error list differ in length.

    See also
    --------
    [MC_metric](MC_metric.md):
        Computes relevant statistics for the whole Monte Carlo experiment (i.e. does not differentiate between overestimation and underestimation).

    Examples
    --------
    >>> from timecave.validation_strategy_metrics import under_over_estimation, PAE
    >>> true_errors = [10, 30, 10, 50];
    >>> validation_errors = [20, 20, 50, 30];
    >>> under_over_estimation(validation_errors, true_errors, PAE)
    ({'Mean': -15.0, 'Median': -15.0, '1st_Quartile': -17.5, '3rd_Quartile': -12.5, 'Minimum': -20.0, 'Maximum': -10.0, 'Standard_deviation': 5.0, 'N': 2, '%': 50.0}, {'Mean': 25.0, 'Median': 25.0, '1st_Quartile': 17.5, '3rd_Quartile': 32.5, 'Minimum': 10.0, 'Maximum': 40.0, 'Standard_deviation': 15.0, 'N': 2, '%': 50.0})
    
    If there are no overestimation or underestimation cases, the respective dictionary will be empty:

    >>> under_over_estimation([10, 20, 30], [5, 10, 15], PAE)
    No errors were underestimated. Underestimation data dictionary empty.
    ({}, {'Mean': 10.0, 'Median': 10.0, '1st_Quartile': 7.5, '3rd_Quartile': 12.5, 'Minimum': 5.0, 'Maximum': 15.0, 'Standard_deviation': 4.08248290463863, 'N': 3, '%': 100.0})

    If the lengths of the estimated error and test error lists do not match, an exception is thrown:

    >>> under_over_estimation(validation_errors, [10], PAE)
    Traceback (most recent call last):
    ...
    ValueError: The estimated error and test error lists must have the same length.
    """

    if(len(estimated_error_list) != len(test_error_list)):

        raise ValueError("The estimated error and test error lists must have the same length.");

    estimated_errors = np.array(estimated_error_list);
    test_errors = np.array(test_error_list);

    under_est = estimated_errors[estimated_errors < test_errors].tolist();
    under_test = test_errors[estimated_errors < test_errors].tolist();
    over_est = estimated_errors[estimated_errors > test_errors].tolist();
    over_test = test_errors[estimated_errors > test_errors].tolist();

    if(len(under_est) > 0):

        under_estimation_stats = MC_metric(under_est, under_test, metric);
        under_estimation_stats["N"] = len(under_est);
        under_estimation_stats["%"] = np.round(len(under_est) / len(estimated_error_list) * 100, 2);
    
    else:

        under_estimation_stats = {};
        print("No errors were underestimated. Underestimation data dictionary empty.");
    
    if(len(over_est) > 0):

        over_estimation_stats = MC_metric(over_est, over_test, metric);
        over_estimation_stats["N"] = len(over_est);
        over_estimation_stats["%"] = np.round(len(over_est) / len(estimated_error_list) * 100, 2);
    
    else:

        over_estimation_stats = {};
        print("No errors were overestimated. Overestimation data dictionary empty.");

    return (under_estimation_stats, over_estimation_stats);

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);