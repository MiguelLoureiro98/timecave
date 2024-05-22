import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from experiment_utils import split_series, shape_series, input_output

def predict_tree(ts_train: pd.Series | np.ndarray, ts_val: pd.Series | np.ndarray, model: DecisionTreeRegressor) -> dict:
    
    """
    Train and test a decision tree model.
    """

    reshaped_train = shape_series(ts_train);
    reshaped_val = shape_series(ts_val);
    X_train, y_train = input_output(reshaped_train);
    X_val, y_val = input_output(reshaped_val);

    model.fit(X_train, y_train);

    y_pred = model.predict(X_val);
    mse = mean_squared_error(y_true=y_val, y_pred=y_pred);
    rmse = np.sqrt(mse);
    mae = mean_absolute_error(y_true=y_val, y_pred=y_pred);

    return {
        "prediction": y_pred,
        "model": model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    };

def predict_ARMA(ts_train: pd.Series | np.ndarray, ts_val: pd.Series | np.ndarray, n_lags: int = 5) -> dict:

    """
    Train and test an ARMA model.
    """

    model = ARIMA(ts_train, order=(n_lags, n_lags, 0));
    res = model.fit();

    y_pred = model.forecast(ts_val);
    mse = mean_squared_error(y_true=ts_val, y_pred=y_pred);
    rmse = np.sqrt(mse);
    mae = mean_absolute_error(y_true=ts_val, y_pred=y_pred);

    return {
        "prediction": y_pred,
        "model": model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    };

if __name__ == "__main__":

    a = np.ones(100);
    b = np.ones(10);

    model = ARIMA(a, order=(2, 2, 0));
    res = model.fit();

    print(res.forecast(10));

    