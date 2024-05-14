import pandas as pd


def check_time_column(
    df: pd.DataFrame, time_col_name="Date", freq="D", fix: bool = False
) -> pd.DataFrame:
    """
    In order to check all 'freq' options, follow the link: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    If fix is set to True:
    - If there are duplicated timestamps, only the first occurrence is kept.
    - If there are missing timestamps, they are added, and NA values are filled in for the remaining columns.
    """
    date_range = pd.date_range(
        start=df[time_col_name].min(), end=df[time_col_name].max(), freq=freq
    )
    nb_na_timesteps = len(date_range) - df[time_col_name].nunique()
    print(f"Number of missing timesteps:{nb_na_timesteps}")

    nb_dup_timesteps = len(df[time_col_name][df[time_col_name].duplicated()])
    print(f"Number of duplicated timesteps:{nb_dup_timesteps}")

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
