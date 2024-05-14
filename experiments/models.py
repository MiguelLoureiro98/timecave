import pandas as pd
from pmdarima.arima import auto_arima
import numpy as np


def predict_arima(series: pd.DataFrame or pd.Series or np.array, pred_window: int) -> np.array:
    # auto_arima chooses the best values for p,d,q alone.
    model = auto_arima(series)

    forecast = model.predict(n_periods=pred_window)

    print("Forecasted values for the next", pred_window, "timesteps:")
    print(forecast)

    return np.array(forecast)
