import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt


def plot_time_series(df, legend=True, title="Time Series Data"):
    df = df.set_index(df.columns[0])

    df.plot(legend=legend)

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title)

    plt.show()


def check_column_order(df, time_col_name, fix=False):
    cols = list(df.columns)
    if cols[0] != time_col_name:
        print("1. FAILED: First column is not time column!")

        if fix:
            cols.remove(time_col_name)
            new_cols = [time_col_name].extend(cols)
            return df[new_cols]
        return

    else:
        print("1. Column order is correct!")
        return


def check_col_types(df):
    wrong_types = 0

    if not pd.api.types.is_integer_dtype(df.index):
        print("Index type is not int")
        wrong_types += 1

    first_column_type = df.iloc[:, 0].dtype
    if not pd.api.types.is_datetime64_any_dtype(first_column_type):
        print("First column type is not datetime-like")
        wrong_types += 1

    for col in df.columns[1:]:
        col_type = df[col].dtype
        if not pd.api.types.is_numeric_dtype(col_type):
            print(f"Column '{col}' type is not float or int: {col_type}")
            wrong_types += 1

    if wrong_types == 0:
        print("2. All column types are correct!")
    else:
        print(f"2. Total {wrong_types} columns have wrong types.")


def check_time_column(
    df: pd.DataFrame,
    time_col_name="Date",
    freq="D",
    fix: bool = False,
    business_days: bool = False,
) -> pd.DataFrame:
    """
    In order to check all 'freq' options, follow the link: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    If fix is set to True:
    - If there are duplicated timestamps, only the first occurrence is kept.
    - If there are missing timestamps, they are added, and NA values are filled in for the remaining columns.
    """
    if business_days == False:
        date_range = pd.date_range(
            start=df[time_col_name].min(), end=df[time_col_name].max(), freq=freq
        )
    else:
        date_range = pd.bdate_range(
            start=df[time_col_name].min(), end=df[time_col_name].max(), freq=freq
        )
    nb_na_timesteps = len(date_range) - df[time_col_name].nunique()
    print(f"3. Number of missing timesteps: {nb_na_timesteps}")

    nb_dup_timesteps = len(df[time_col_name][df[time_col_name].duplicated()])
    print(f"4. Number of duplicated timesteps: {nb_dup_timesteps}")

    if fix:
        if nb_dup_timesteps > 0:
            df = df.drop_duplicates(subset=[time_col_name], keep="first")

        if nb_na_timesteps > 0:
            full_date_range_df = pd.DataFrame(
                index=pd.date_range(
                    start=df[time_col_name].min(),
                    end=df[time_col_name].max(),
                    freq=freq,
                )
            )
            df = full_date_range_df.merge(
                df, left_index=True, right_on=time_col_name, how="left"
            ).reset_index(drop=True)

        return df
    else:
        return


def check_missing_values(df, alpha=1, fix=False, check_na=False):
    na_df = df.isnull()
    missing_counts = na_df.sum()
    missing_percentage = df.isnull().mean()
    nb_col_with_na = 0
    selected_columns = missing_percentage[missing_percentage <= alpha].index
    if check_na:
        print(df[na_df.any(axis=1)].date)

    for column, count in missing_counts.items():
        if count > 0 and column in selected_columns:
            nb_col_with_na += 1
            print(f"Number of missing values in column '{column}' : {count}")
    print(f"5. Number of Time Series with missing values : {nb_col_with_na}")

    if fix == True:
        df_filtered = df[selected_columns].copy()
        df_filtered.ffill(inplace=True)
        return df_filtered
    else:
        return


def data_report(
    df: pd.DataFrame, time_col_name="Date", freq="D", business_days: bool = False
):
    print("_" * 64)
    print(" " * 25, "DATA REPORT", " " * 25)
    print("_" * 64)
    check_column_order(df, time_col_name)
    check_col_types(df)
    check_time_column(df, time_col_name, freq, business_days=business_days, fix=False)
    check_missing_values(df, fix=False)
    print("_" * 64)
