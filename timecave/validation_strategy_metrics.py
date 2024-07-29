"""
This module contains metrics to evaluate the performance of model validation methods.

>>> print("Hey")
Hey
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

    [RAPAE](rapae.md):
        Relative Absolute Predictive Accuracy Error.

    Notes
    -----
    The Predictive Accuracy Error is defined as the difference between the estimate of a model's error given by a validation method\
    and the model's true error:
    
    $$
    PAE = \hat{L}_m - L_m
    $$ 
    
    Note that, in all likelihood, the true error will not be known. It is usually estimated using an independent test set. For more details, please refer to [[1]](#1).

    References
    ----------
    <a id="1">[1]</a>
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
    """

    if(test_error == 0):

        raise ValueError("The test error is zero. RPAE is undefined.");

    return (estimated_error - test_error) / test_error;

def sMPAE(estimated_error: float | int, test_error: float | int) -> float:
    
    """
    Compute the Relative Predictive Accuracy Error (sMPAE).

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
        Relative Predictive Accuracy Error.

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