import numpy as np
import pandas as pd
from os import getcwd
import glob
import pandas.api.types as pdt
from timecave.validation_methods._base import base_splitter
from timecave.validation_methods.OOS import (
    Holdout,
    Repeated_Holdout,
    Rolling_Origin_Update,
    Rolling_Origin_Recalibration,
    Fixed_Size_Rolling_Window,
)
from timecave.validation_methods.markov import MarkovCV
from timecave.validation_methods.prequential import Growing_Window, Rolling_Window
from timecave.validation_methods.CV import Block_CV, hv_Block_CV
from timecave.validation_methods.weights import (
    constant_weights,
    linear_weights,
    exponential_weights,
)
from datetime import datetime
import re, os
import statsmodels.api as sm


def save_tables(
    table_A: pd.DataFrame,
    table_B: pd.DataFrame,
    stats_total: pd.DataFrame,
    stats_train: pd.DataFrame,
    stats_val: pd.DataFrame,
    dir: str,
    add_name: str = "",
    add_timestamp: bool = True,
):
    """
    Saves all the tables resulting from experiments.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d__%H_%M_%S") if add_timestamp else ""
    append_to_name = f"{add_name}_{timestamp}.csv"
    table_A.to_csv(dir + "/table_A_" + append_to_name, index=False)
    table_B.to_csv(dir + "/table_B_" + append_to_name, index=False)
    stats_total.to_csv(dir + "/stats_total_" + append_to_name, index=False)
    stats_train.to_csv(dir + "/stats_train_" + append_to_name, index=False)
    stats_val.to_csv(dir + "/stats_val_" + append_to_name, index=False)


def get_lastest_csv_dirs(folder_dir: str):
    """
    Gets the latest .csv names within a folder based on the timestamps in their name.
    """
    pattern = re.compile(
        rf"table_A_(\d{{4}}_\d{{2}}_\d{{2}}__\d{{2}}_\d{{2}}_\d{{2}}).csv"
    )
    files = os.listdir(folder_dir)

    timestamps = []
    for file in files:
        match = pattern.match(file)
        if match:
            timestamps.append((file, match.group(1)))

    if not timestamps:
        return None, None, None, None, None

    timestamps.sort(
        key=lambda x: datetime.strptime(x[1], "%Y_%m_%d__%H_%M_%S"), reverse=True
    )
    latest_timestamp = timestamps[0][1]

    table_A_file = f"table_A__{latest_timestamp}.csv"
    table_B_file = f"table_B__{latest_timestamp}.csv"
    stats_total_file = f"stats_total__{latest_timestamp}.csv"
    stats_train_file = f"stats_train__{latest_timestamp}.csv"
    stats_val_file = f"stats_val__{latest_timestamp}.csv"
    return (
        table_A_file,
        table_B_file,
        stats_total_file,
        stats_train_file,
        stats_val_file,
    )


def read_tables(
    table_A_file: str,
    table_B_file: str,
    stats_total_file: str,
    stats_train_file: str,
    stats_val_file: str,
    dir: str,
):
    table_A = pd.read_csv(dir + table_A_file)
    table_B = pd.read_csv(dir + table_B_file)
    stats_total = pd.read_csv(dir + stats_total_file)
    stats_train = pd.read_csv(dir + stats_train_file)
    stats_val = pd.read_csv(dir + stats_val_file)

    return table_A, table_B, stats_total, stats_train, stats_val


def get_last_iteration(df: pd.DataFrame):
    last_row = df.iloc[-1].to_dict()


def get_autocorrelation_order(ts, nlags=5):
    acf_values, confint = sm.tsa.acf(ts, nlags=nlags, alpha=0.05)

    # Determine the order by finding the first lag where the acf value is not significant
    # Confint provides the confidence interval for each lag; if the acf value is within this interval, it is not significant.
    lower_bound = confint[:, 0]
    upper_bound = confint[:, 1]
    autocorrelation_order = np.where(
        (acf_values < lower_bound) | (acf_values > upper_bound),
    )[0]
    if len(autocorrelation_order) == 0:
        # If no significant lag is found, return the maximum lag
        return nlags
    else:
        # Return the first significant lag
        return autocorrelation_order[0]


def get_methods_list(ts, freq):
    holdout = Holdout(ts, freq, validation_size=0.7)
    rep_hold = Repeated_Holdout(
        ts, freq, iterations=5, splitting_interval=[0.7, 0.8], seed=0
    )
    rol_origin_update = Rolling_Origin_Update(ts, freq, origin=0.7)
    rol_origin_cal = Rolling_Origin_Recalibration(ts, freq, origin=0.7)
    fix_size_roll_wind = Fixed_Size_Rolling_Window(ts, freq, origin=0.7)
    grow_window = Growing_Window(5, ts, freq, gap=0)
    gap_grow_window = Growing_Window(5, ts, freq, gap=1)
    # weighted_grow_window = Growing_Window(5,ts,freq,gap=3, weight_function=exponential_weights,params={"base": 2})
    roll_window = Rolling_Window(5, ts, freq, gap=0)
    gap_roll_window = Rolling_Window(5, ts, freq, gap=1)
    # weighted_roll_window = Rolling_Window(5,ts,freq,gap=3,weight_function=exponential_weights,params={"base": 2},)
    block_cv = Block_CV(5, ts, freq)
    # weight_block_cv = Block_CV( 5, ts, freq, weight_function=exponential_weights, params={"base": 2})
    hv_block = hv_Block_CV(ts, freq, h=5, v=5)
    p = get_autocorrelation_order(ts)
    markov = MarkovCV(ts, p, seed=1)

    return [
        holdout,
        rep_hold,
        rol_origin_update,
        rol_origin_cal,
        fix_size_roll_wind,
        grow_window,
        gap_grow_window,
        roll_window,
        gap_roll_window,
        block_cv,
        hv_block,
        markov,
    ]


def initialize_tables():
    colname_A = [
        "filename",
        "column_index",
        "method",
        "iteration",
        "model",
        "mse",
        "mae",
        "rmse",
    ]
    table_A = pd.DataFrame(columns=colname_A)
    colname_B = [
        "filename",
        "column_index",
        "model",
        "mse",
        "mae",
        "rmse",
    ]
    table_B = pd.DataFrame(columns=colname_B)
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


if __name__ == "__main__":
    ts = np.arange(100)
    assert get_autocorrelation_order(ts, 5) == 5

    ts = np.ones(100)
    assert get_autocorrelation_order(ts, 5) == 0
    print()
