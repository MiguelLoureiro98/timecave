"""
This module contains functions to compute time series features.

The 'get_features' function extracts all features supported by this package.
Functions to extract the strength of trend, mean-crossing rate, and median-crossing
rate are also provided.
For the remaining features, the tsfel package should be used.

Functions
---------

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
    """

    _check_type(ts);
    _check_sampling_rate(fs);

    feature_list = ["0_Mean", "0_Median", "0_Min", "0_Max", "0_Variance", "0_Peak to peak distance"];

    cfg = tsfel.get_features_by_domain("statistical");
    stat_feat_df = tsfel.time_series_features_extractor(cfg, ts, fs);
    
    relevant_feat_df = stat_feat_df[feature_list].copy();
    new_names = [feat[2:] for feat in feature_list];
    cols = {name: new_name for (name, new_name) in zip(feature_list, new_names)};
    relevant_feat_df = relevant_feat_df.rename(columns=cols);
    relevant_feat_df = relevant_feat_df.rename(columns={"Peak to peak distance": "P2P_amplitude"});
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
    """

    _check_type(ts);

    if(isinstance(ts, np.ndarray) is True):

        diff_ts = np.diff(ts);

    else:
        
        diff_ts = ts.diff().dropna();

    ts_std = ts.std();
    diff_std = diff_ts.std();
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

    #TODO Add reference (i.e. papers, formula ...).

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    Returns
    -------
    float
        Mean-crossing rate.
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

    #TODO Add reference (i.e. papers, formula ...)

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    Returns
    -------
    float
        Median-crossing rate.
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
    or an integer and a ValueError if the sampling frequency is negative (or zero).
    """

    if((isinstance(fs, float) or isinstance(fs, int)) is False):

        raise TypeError("The sampling frequency should be either a float or an integer.");

    if(fs <= 0):

        raise ValueError("The sampling frequency should be larger than zero.");

    return;