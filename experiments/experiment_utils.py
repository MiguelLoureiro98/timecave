import numpy as np
import pandas as pd
from os import getcwd
import glob
import pandas.api.types as pdt
from timecave.validation_methods._base import base_splitter


def initialize_tables():
    table_A = pd.DataFrame(
        columns=[
            "filename",
            "column_index",
            "method",
            "iteration",
            "model",
            "mse",
            "mae",
            "rmse",
        ]
    )
    colname_stats = [
        "filename",
        "column_index",
        "Frequency",
        "Mean",
        "Median",
        "Min",
        "Max",
        "Variance",
        "P2P_amplitude",
        "Trend_slope",
        "Spectral_centroid",
        "Spectral_rolloff",
        "Spectral_entropy",
        "Strength_of_trend",
        "Mean_crossing_rate",
        "Median_crossing_rate",
    ]
    table_B = pd.DataFrame(columns=colname_stats)
    stats_total = pd.DataFrame(columns=colname_stats)
    stats_train = pd.DataFrame(columns=colname_stats)
    stats_val = pd.DataFrame(columns=colname_stats)
    return table_A, table_B, stats_total, stats_train, stats_val


def update_stats_tables(
    stats_total: pd.DataFrame,
    stats_train: pd.DataFrame,
    stats_test: pd.DataFrame,
    method: base_splitter,
    filename: str,
    col_idx: int,
    freq,
):
    """
    Updates the statistics dataframes.
    """
    df1, df2, df3 = method.statistics()
    df1["filename"], df2["filename"], df3["filename"] = (
        filename,
        filename,
        filename,
    )

    df1["column_index"], df2["column_index"], df3["column_index"] = (
        col_idx,
        col_idx,
        col_idx,
    )

    df1["Frequency"], df2["Frequency"], df3["Frequency"] = (
        freq,
        freq,
        freq,
    )

    assert set(df1.columns) == set(stats_total.columns), "Columns do not match"
    assert set(df2.columns) == set(stats_train.columns), "Columns do not match"
    assert set(df3.columns) == set(stats_test.columns), "Columns do not match"

    stats_total = pd.concat([stats_total, df1])
    stats_train = pd.concat([stats_train, df2])
    stats_test = pd.concat([stats_test, df3])

    return stats_total, stats_train, stats_test


def get_freq(df: pd.DataFrame, date_column: str):
    """
    Gets frequency given the date column of a pandas dataframe.
    """
    if not pdt.is_datetime64_any_dtype(df[date_column]):
        return 1
    df["diff"] = df[date_column].diff()
    return 1 / df["diff"].iloc[1].total_seconds()


def get_csv_filenames(folder: str):
    folder = getcwd() + "\\" + folder + "\\*.csv"
    return glob.glob(folder)


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

    splitting_index = int(np.round((1 - test_set_proportion) * ts.shape[0]))

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
