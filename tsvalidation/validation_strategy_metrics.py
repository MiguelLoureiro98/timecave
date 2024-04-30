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
    """

    return (estimated_error - test_error) / test_error;

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
    """

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

    under_estimation_stats = MC_metric(under_est, under_test, metric);
    over_estimation_stats = MC_metric(over_est, over_test, metric);

    under_estimation_stats["N"] = len(under_est);
    under_estimation_stats["%"] = np.round(len(under_est) / len(estimated_error_list) * 100, 2);
    over_estimation_stats["N"] = len(over_est);
    over_estimation_stats["%"] = np.round(len(over_est) / len(estimated_error_list) * 100, 2);

    return (under_estimation_stats, over_estimation_stats);