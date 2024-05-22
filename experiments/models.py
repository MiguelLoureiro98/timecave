import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from experiment_utils import shape_series, get_X_y

def recursive_forecast_tree(ts_val: np.ndarray, n_lags: int, pred_window: int, model: DecisionTreeRegressor) -> np.ndarray:

    """
    Recursive forecasting for decision trees.
    """

    forecasts = np.zeros(pred_window);
    input = ts_val[:n_lags];

    for ind in range(pred_window):

        forecasts[ind] = model.predict(input);
        input = np.hstack((input[1:], forecasts[ind]));

    return forecasts;

def predict_tree(ts_train: pd.Series | np.ndarray, ts_val: pd.Series | np.ndarray, n_lags: int) -> dict:
    
    """
    Train and test a decision tree model.
    """

    reshaped_train = shape_series(ts_train);
    reshaped_val = shape_series(ts_val);
    X_train, y_train = get_X_y(reshaped_train);
    X_val, y_val = get_X_y(reshaped_val);

    model = DecisionTreeRegressor();

    model.fit(X_train, y_train);

    y_pred = recursive_forecast_tree(X_val, n_lags, y_val.shape[0], model);
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

    y_pred = res.forecast(ts_val.shape[0]);
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

    #model = ARIMA(a, order=(5, 5, 0));
    #res = model.fit();

    #print(res.forecast(10));

    #ARMA_results = predict_ARMA(a, b);

    #print(ARMA_results);

    DT_res = predict_tree(a, b, 5);

    print(DT_res);