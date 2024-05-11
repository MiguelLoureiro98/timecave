import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from pandas import Timedelta

def check_time_column(df:pd.Dataframe, time_col_name='Date', freq: str or Timedelta or timedelta or DateOffset, fix:bool = False) -> pd.DataFrame:
    """
    In order to check all 'freq' options, follow the link: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    """
    date_range = pd.date_range(start=df[time_col_name].min(), end=df[time_col_name].max(), freq=freq)
    nb_na_timesteps = len(date_range) - df[time_col_name].nunique()
    print(f'Number of missing timesteps:{nb_na_timesteps}')

    nb_dup_timesteps = len(df[time_col_name][df[time_col_name].duplicated()])
    print(f'Number of duplicated timesteps:{nb_dup_timesteps}')

    if fix:
        