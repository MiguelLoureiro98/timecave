import pandas as pd


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
    if business_days == "False":
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
            df_new = df.drop_duplicates(subset=[time_col_name], keep="first")

        if nb_na_timesteps > 0:
            full_date_range_df = pd.DataFrame(
                index=pd.date_range(
                    start=df_new[time_col_name].min(),
                    end=df_new[time_col_name].max(),
                    freq=freq,
                )
            )
            df_new = full_date_range_df.merge(
                df_new, left_index=True, right_on=time_col_name, how="left"
            ).reset_index(drop=True)

        return df_new
    else:
        return


def check_missing_values(df, fix=False):
    missing_counts = df.isnull().sum()
    nb_col_with_na = 0
    for column, count in missing_counts.items():
        if count > 0:
            nb_col_with_na += 1
            print(f"Number of missing values in column '{column}' : {count}")
    if fix == True:
        df.fillna(method="ffill", inplace=True)
    print(f"5. Number of Time Series with missing values : {nb_col_with_na}")
    return df


def check_col_types(df):
    wrong_types = 0

    # Check index type
    if not pd.api.types.is_integer_dtype(df.index):
        print("Index type is not int")
        wrong_types += 1

    # Check first column type
    first_column_type = df.iloc[:, 0].dtype
    if not pd.api.types.is_datetime64_any_dtype(first_column_type):
        print("First column type is not datetime-like")
        wrong_types += 1

    # Check remaining columns
    for col in df.columns[1:]:
        col_type = df[col].dtype
        if not pd.api.types.is_numeric_dtype(col_type):
            print(f"Column '{col}' type is not float or int: {col_type}")
            wrong_types += 1

    if wrong_types == 0:
        print("2. All column types are correct!")
    else:
        print(f"2. Total {wrong_types} columns have wrong types.")


def data_report(
    df: pd.DataFrame, time_col_name="Date", freq="D", business_days: bool = False
):
    print("_" * 64)
    print(" " * 25, "DATA REPORT", " " * 25)
    print("_" * 64)
    check_col_types(df)
    check_time_column(df, time_col_name, freq, business_days=business_days, fix=False)
    check_missing_values(df, fix=False)
    print("_" * 64)
