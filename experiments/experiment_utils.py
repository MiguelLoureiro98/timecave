import numpy as np
import pandas as pd


def get_univariate_series(dataset: pd.DataFrame) -> list[pd.Series]:
    """
    Split a multivariate time series into several univariate ones.
    """

    series_list = [dataset[column].copy() for column in dataset.columns[1:]]

    return series_list


def split_series(ts: pd.Series, test_set_proportion: float = 0.2) -> tuple[pd.Series]:
    """
    Split a univariate time series into training and test sets.
    """

    splitting_index = np.round((1 - test_set_proportion) * ts.shape[0])

    train = ts.iloc[:splitting_index]
    test = ts.iloc[splitting_index:]

    return (train, test)


def shape_series(ts: pd.Series, n_lags: int = 5) -> pd.DataFrame:
    """
    Format a time series so it can be fed to an LSTM and a Decision Tree.
    """

    lags = [ts.shift(lag) for lag in range(0, -(n_lags + 1), -1)]
    col_names = ["t {}".format(i) for i in range(-n_lags, 0)] + ["t"]

    series = pd.concat(lags, axis=1)
    # series = pd.concat([ts, lags], axis=1)
    series.columns = col_names
    # series = series.dropna(how="all")
    series = series[:-n_lags].reset_index(drop=True)

    return series


def get_X_y(ts: pd.Series, n_lags: int = 5) -> tuple[np.array]:
    """
    Reshapes data in order to be used for the lstm or the decision tree. Returns input/output numpy arrays
    """
    series = shape_series(ts, n_lags)
    return series.iloc[:, :-1].values, series.iloc[:, -1].values
