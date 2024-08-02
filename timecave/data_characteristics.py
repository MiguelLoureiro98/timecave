"""
This module contains functions to compute time series features.

The 'get_features' function extracts all features supported by this package.
Functions to extract the strength of trend, mean-crossing rate, and median-crossing
rate are also provided.
For the remaining features, the [tsfel](https://github.com/fraunhoferportugal/tsfel) package should be used.

Functions
---------
get_features
    Extract 13 different features from a time series.

strength_of_trend
    Compute the time series' strength of trend.

mean_crossing_rate
    Compute the time series' mean-crossing rate.

median_crossing_rate
    Compute the time series' median-crossing rate.
"""

import numpy as np
import pandas as pd
import tsfel

def get_features(ts: np.ndarray | pd.Series, fs: float | int) -> pd.DataFrame:
    
    """
    Compute time series features.

    This function extracts features from a time series. The tsfel package is used to extract most features,
    and should be used if only these are required. The exceptions are the 'Strength of Trend',
    'Mean-crossing rate', and 'Median-crossing rate' features, for which custom functions were
    developed (these were also made available to the user). 

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int
        Sampling frequency (Hz).

    Returns
    -------
    pd.DataFrame
        Data frame containing all time series features supported by this package.

    Raises
    ------
    TypeError
        If 'ts' is neither an Numpy array nor a Pandas series.

    TypeError
        If 'fs' is neither a float nor an integer.

    ValueError
        If 'fs' is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from timecave.data_characteristics import get_features
    >>> t = np.arange(0, 10, 0.01);
    >>> time_series = np.sin(2 * np.pi * t);
    >>> sampling_frequency = 1 / 0.01;
    >>> get_features(time_series, sampling_frequency)
               Mean        Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
    0  3.552714e-18 -3.673940e-16 -1.0  1.0       0.5            2.0    -0.000191                1.0               1.0      6.485530e-29          15.926086             0.02002              0.019019

    If the time series is neither an array nor a series, an exception is thrown:

    >>> get_features([0, 1, 2], sampling_frequency)
    Traceback (most recent call last):
    ...
    TypeError: Time series must be either a Numpy array or a Pandas series.

    The same happens if the sampling frequency is neither a float nor an integer:

    >>> get_features(time_series, "Hello")
    Traceback (most recent call last):
    ...
    TypeError: The sampling frequency should be either a float or an integer.

    A different exception is raised if the sampling frequency is negative:

    >>> get_features(time_series, -1)
    Traceback (most recent call last):
    ...
    ValueError: The sampling frequency should be larger than zero.
    """

    _check_type(ts);
    _check_sampling_rate(fs);

    #feature_list = ["0_Mean", "0_Median", "0_Min", "0_Max", "0_Variance", "0_Peak to peak distance"];

    #cfg = tsfel.get_features_by_domain("statistical");
    #stat_feat_df = tsfel.time_series_features_extractor(cfg, ts, fs);
    
    #relevant_feat_df = stat_feat_df[feature_list].copy();
    #new_names = [feat[2:] for feat in feature_list];
    #cols = {name: new_name for (name, new_name) in zip(feature_list, new_names)};
    #relevant_feat_df = relevant_feat_df.rename(columns=cols);
    #relevant_feat_df = relevant_feat_df.rename(columns={"Peak to peak distance": "P2P_amplitude"});

    mean = tsfel.calc_mean(ts);
    median = tsfel.calc_median(ts);
    minimum = tsfel.calc_min(ts);
    maximum = tsfel.calc_max(ts);
    variance = tsfel.calc_var(ts);
    p2p = tsfel.pk_pk_distance(ts);
    feature_list = [mean, median, minimum, maximum, variance, p2p];
    feature_names = ["Mean", "Median", "Min", "Max", "Variance", "P2P_amplitude"];

    relevant_feat_df = pd.DataFrame(data={name: [feat] for name, feat in zip(feature_names, feature_list)});
    relevant_feat_df["Trend_slope"] = tsfel.slope(ts);
    relevant_feat_df["Spectral_centroid"] = tsfel.spectral_centroid(ts, fs);
    relevant_feat_df["Spectral_rolloff"] = tsfel.spectral_roll_off(ts, fs);
    relevant_feat_df["Spectral_entropy"] = tsfel.spectral_entropy(ts, fs);
    relevant_feat_df["Strength_of_trend"] = strength_of_trend(ts);
    relevant_feat_df["Mean_crossing_rate"] = mean_crossing_rate(ts);
    relevant_feat_df["Median_crossing_rate"] = median_crossing_rate(ts);

    return relevant_feat_df;

def strength_of_trend(ts: np.ndarray | pd.Series) -> float:
    
    """
    Compute the strength of trend of a time series.

    This function computes the strength of trend of a given time series using the method
    employed by Cerqueira et. al (2020) (i.e. the ratio between the time series' standard deviation and
    that of the differenced time series).

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    Returns
    -------
    float
        Strength of trend.

    Raises
    ------
    TypeError
        If 'ts' is neither a Numpy array nor a Pandas series.

    Notes
    -----
    Let $\sigma$ be the standard deviation of a given time series. The strength of trend of a series is defined \
    by Cerqueira et al [[1]](#1) as:

    $$
    SOT = \\frac{\sigma_{ts}}{\sigma_{diff}}
    $$

    where $ts$ stands for the time series itself and $diff$ denotes the differenced time series.

    References
    ----------
    ##1
    Cerqueira, V., Torgo, L., Mozetiˇc, I., 2020. Evaluating time series forecasting
    models: An empirical study on performance estimation methods.
    Machine Learning 109, 1997–2028.

    Examples
    --------
    >>> import numpy as np
    >>> from timecave.data_characteristics import strength_of_trend
    >>> rng = np.random.default_rng(seed=1);
    >>> noise = rng.uniform(low=0, high=0.01, size=10);
    >>> constant_series = np.ones(10);
    >>> strength_of_trend(constant_series + noise)
    0.5717034302917938

    For a series with a strong trend, this value will be larger:

    >>> series_trend = np.arange(0, 10);
    >>> strength_of_trend(series_trend + noise)
    543.4144869043147

    For pure trends, the strength of trend is infinite:

    >>> strength_of_trend(series_trend)
    inf

    If the time series is neither an array nor a series, an exception is thrown:

    >>> strength_of_trend([0, 1, 2])
    Traceback (most recent call last):
    ...
    TypeError: Time series must be either a Numpy array or a Pandas series.
    """

    _check_type(ts);

    if(isinstance(ts, np.ndarray) is True):

        diff_ts = np.diff(ts);

    else:
        
        diff_ts = ts.diff().dropna();

    ts_std = ts.std();
    diff_std = diff_ts.std();

    if(diff_std == 0):

        SOT = np.inf;
    
    else:
        
        SOT = ts_std / diff_std;

    return SOT;

def mean_crossing_rate(ts: np.ndarray | pd.Series) -> float:
    
    """
    Compute the series' mean-crossing rate.

    This function computes the mean-crossing rate of a given time series.
    The mean-crossing rate is defined as the rate at which the values of a time
    series change from being below its mean value to above said value.
    In practice, the mean is subtracted from the time series, and the zero-crossing
    rate is then computed.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    Returns
    -------
    float
        Mean-crossing rate.

    Raises
    ------
    TypeError
        If 'ts' is neither a Numpy array nor a Pandas series.

    See also
    --------
    [median_crossing_rate](med_cr.md):
        Uses the median instead of the mean.

    Notes
    -----
    The mean-crossing rate is defined as the fraction of times a mean-crossing takes place [the mean is crossed] in the whole time series. \
    A mean-crossing occurs when two adjacent values have different signs with respect to the mean \
    (i.e. the first one is below the mean while the second one is above it, and vice-versa).

    $$
    MCR = \\frac{1}{n-1} \sum_{i=2}^{n} |sign(a_i - \mu) - sign(a_{i-1} - \mu)|  
    $$

    where $n$ is the number of samples in the time series, $a_i$ are its values, and $\mu$ represents its mean.
    For more details, please refer to [[1]](#1).

    References
    ----------
    ##1
    Bohdan Myroniv, Cheng-Wei Wu, Yi Ren, Albert Christian, Ensa Bajo,
    and Yu-chee Tseng. Analyzing user emotions via physiology signals. Data
    Science and Pattern Recognition, 2, 12 2017.

    Examples
    --------
    >>> import numpy as np
    >>> from timecave.data_characteristics import mean_crossing_rate
    >>> ts = np.array([0, 20, 0, 20, 0]);
    >>> mean_crossing_rate(ts)
    1.0
    >>> ts2 = np.ones(10);
    >>> mean_crossing_rate(ts2)
    0.0
    >>> ts3 = np.array([50, 50, 50, 0, 0]);
    >>> mean_crossing_rate(ts3)
    0.25

    If the time series is neither an array nor a series, an exception is thrown:

    >>> mean_crossing_rate([0, 1, 2])
    Traceback (most recent call last):
    ...
    TypeError: Time series must be either a Numpy array or a Pandas series.
    """

    _check_type(ts);

    new_ts = ts - ts.mean();

    if(isinstance(ts, pd.Series) is True):

        ts = ts.to_numpy();
    
    mcr = np.nonzero(np.diff(np.sign(new_ts)))[0].shape[0] / (ts.shape[0] - 1);

    return mcr;

def median_crossing_rate(ts: np.ndarray | pd.Series) -> float:
    
    """
    Compute the series' median-crossing rate.

    This function computes the median-crossing rate of a given time series.
    The median-crossing rate is defined as the rate at which the values of a time
    series change from being below its median value to above said value.
    In practice, the median is subtracted from the time series, and the zero-crossing
    rate is then computed.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    Returns
    -------
    float
        Median-crossing rate.

    Raises
    ------
    TypeError
        If 'ts' is neither a Numpy array nor a Pandas series.

    See also
    --------
    [mean_crossing_rate](mean_cr.md)
        Uses the mean instead of the median.

    Notes
    -----
    The median-crossing rate is similar to the mean-crossing rate, but it uses the median as a reference value. \
    It can be computed from the following formula:

    $$
    MedCR = \\frac{1}{n-1} \sum_{i=2}^{n} |sign(a_i - Med) - sign(a_{i-1} - Med)|  
    $$

    where $n$ is the number of samples in the time series, $a_i$ are its values, and $Med$ represents its median.
    The formula for the mean-crossing rate can be found in [[1]](#1).
    
    References
    ----------
    ##1
    Bohdan Myroniv, Cheng-Wei Wu, Yi Ren, Albert Christian, Ensa Bajo,
    and Yu-chee Tseng. Analyzing user emotions via physiology signals. Data
    Science and Pattern Recognition, 2, 12 2017.
    
    Examples
    --------
    >>> import numpy as np
    >>> from timecave.data_characteristics import median_crossing_rate
    >>> ts = np.array([0, 20, 0, 20, 0]);
    >>> median_crossing_rate(ts)
    1.0
    >>> ts2 = np.ones(10);
    >>> median_crossing_rate(ts2)
    0.0
    >>> ts3 = np.array([50, 50, 50, 0, 0]);
    >>> median_crossing_rate(ts3)
    0.25
    >>> ts4 = np.array([0, 20, 5, 5, 5]);
    >>> median_crossing_rate(ts4)
    0.5

    If the time series is neither an array nor a series, an exception is thrown:

    >>> median_crossing_rate([0, 1, 2])
    Traceback (most recent call last):
    ...
    TypeError: Time series must be either a Numpy array or a Pandas series.
    """

    _check_type(ts);

    if(isinstance(ts, pd.Series) is True):

        ts = ts.to_numpy();

    new_ts = ts - np.median(ts);
    
    mcr = np.nonzero(np.diff(np.sign(new_ts)))[0].shape[0] / (ts.shape[0] - 1);

    return mcr;

def _check_type(ts: np.ndarray | pd.Series) -> None:
    
    """
    Check the time series' type. Raises a TypeError if the series is not a 
    Numpy array nor a Pandas series.
    """

    if(isinstance(ts, np.ndarray) is False and isinstance(ts, pd.Series) is False):

        raise TypeError("Time series must be either a Numpy array or a Pandas series.");

    return;

def _check_feature_list(feature_list: list, n_features_max: int) -> None:
    
    """
    Perform checks on the feature list.
    """

    if(isinstance(feature_list, list) is False):

        raise TypeError("'feature_list' should be a list.");

    if(len(feature_list) > n_features_max):

        raise ValueError("The number of features should not surpass the maximum number of features supported by this package.");

    return;

def _check_sampling_rate(fs: float | int) -> None:
    
    """
    Perform checks on the sampling rate. Raises a TypeError if the sampling rate is neither a float
    nor an integer and a ValueError if the sampling frequency is negative (or zero).
    """

    if((isinstance(fs, float) or isinstance(fs, int)) is False):

        raise TypeError("The sampling frequency should be either a float or an integer.");

    if(fs <= 0):

        raise ValueError("The sampling frequency should be larger than zero.");

    return;

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);