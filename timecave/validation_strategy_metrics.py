"""
This module contains metrics to evaluate the performance of model validation methods.
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
        The model's validation error.

    test_error : float | int
        The model's test error.

    Returns
    -------
    float
        Predictive Accuracy Error.

    See also
    --------
    [APAE](apae.md) :
        Absolute Predictive Accuracy Error.

    [RPAE](rpae.md) : 
        Relative Predictive Accuracy Error.

    [RAPAE](rapae.md) :
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
        The model's validation error.

    test_error : float | int
        The model's test error.

    Returns
    -------
    float
        Absolute Predictive Accuracy Error.

    See also
    --------
    [PAE](pae.md) :
        Predictive Accuracy Error.

    [RPAE](rpae.md) : 
        Relative Predictive Accuracy Error.

    [RAPAE](rapae.md) :
        Relative Absolute Predictive Accuracy Error.

    [sMPAE](smpae.md):
        Symmetric Mean Predictive Accuracy Error.

    Notes
    -----
    The Absolute Predictive Accuracy Error is defined as the absolute value of the difference between the\
    estimate of a model's error given by a validation method\
    and the model's true error. In other words, it is the absolute value of the Predictive Accuracy Error:
    
    $$
    APAE = |\hat{L}_m - L_m| = |PAE|
    $$ 
    
    Since the APAE is always non-negative, this metric does not measure [cannot be used to determine] whether the validation method is overestimating or underestimating\
    the model's true error.\
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
        The model's validation error.

    test_error : float | int
        The model's test error.

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
    [PAE](pae.md) :
        Predictive Accuracy Error.

    [APAE](apae.md) : 
        Absolute Predictive Accuracy Error.

    [RAPAE](rapae.md) :
        Relative Absolute Predictive Accuracy Error.

    [sMPAE](smpae.md):
        Symmetric Mean Predictive Accuracy Error.

    Notes
    -----
    The Relative Predictive Accuracy Error divides the Predictive Accuracy Error (PAE) by the model's true error:
    
    $$
    RPAE = \\frac{\hat{L}_m - L_m}{L_m} = \\frac{PAE}{L_m}
    $$ 
    
    [By doing so, the metric is made ... .]
    This makes this metric scale-independent with respect to the model's true error, which in turn makes it useful for comparing validation methods\
    [that have been applied on different ...] across different time series and/or forecasting models. Since this is essentially a scaled version of the PAE, the sign retains its significance.\
    However, it should be noted that the RPAE is asymmetric: in case of an underestimation, its values will be contained in the interval [-1, 0[; if the error is\
    overestimated, however, the RPAE can take any value in the range ]0, $\infty$[.

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
        The model's validation error.

    test_error : float | int
        The model's test error.

    Returns
    -------
    float
        Symmetric Mean Predictive Accuracy Error.

    Raises
    ------
    ValueError
        If 'test_error' is zero.
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
        The model's validation error.

    test_error : float | int
        The model's test error.

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
    [PAE](pae.md) :
        Predictive Accuracy Error.

    [APAE](apae.md) : 
        Absolute Predictive Accuracy Error.

    [RPAE](rpae.md) :
        Relative Predictive Accuracy Error.

    [sMPAE](smpae.md):
        Symmetric Mean Predictive Accuracy Error.    

    Notes
    -----
    The Relative Absolute Predictive Accuracy Error is defined as the Absolute Predictive Accuracy Error (APAE) divided by the\
    model's true error. It can also be seen as the absolute value of the Relative Predictive Accuracy Error (RPAE):
    
    $$
    RAPAE = \\frac{|\hat{L}_m - L_m|}{L_m} = \\frac{|PAE|}{L_m} = \\frac{APAE}{L_m} = |RPAE|
    $$ 
    
    This metric essentially takes the absolute value of the RPAE, and can be used in a similar fashion. However, since it uses the\
    absolute value, it cannot be used to determine whether a validation method is overestimating or underestimating the model's\
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
    Compute validation strategy metrics for N different experiments.

    This function computes ... .

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
        _description_

    Raises
    ------
    ValueError
        If the estimator error list and the test error list differ in length.

    See also
    --------
    [under_over_estimation](under_over.md):
        _description_
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

    This function can be used to compute ... .

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
        _description_

    Raises
    ------
    ValueError
        If the estimator error list and the test error list differ in length.

    See also
    --------
    [MC_metric](MC_metric.md):
        _description_
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