import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from experiment_utils import get_X_y

def recursive_forecast_tree(ts_val: np.ndarray, pred_window: int, model: DecisionTreeRegressor) -> np.ndarray:

    """
    Recursive forecasting for decision trees.
    """

    forecasts = np.zeros(pred_window);
    input = ts_val[0];

    for ind in range(pred_window):

        forecasts[ind] = model.predict(input.reshape(1, -1)).item();
        input = np.hstack((input[1:], forecasts[ind]));

    return forecasts;

def predict_tree(ts_train: pd.Series, ts_val: pd.Series) -> dict:
    
    """
    Train and test a decision tree model.
    """

    #reshaped_train = shape_series(ts_train);
    #reshaped_val = shape_series(ts_val);
    X_train, y_train = get_X_y(ts_train);
    X_val, y_val = get_X_y(ts_val);

    model = DecisionTreeRegressor();

    model.fit(X_train, y_train);

    y_pred = recursive_forecast_tree(X_val, y_val.shape[0], model);
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

    a = np.append(np.arange(100), np.arange(100, 120));
    b = np.arange(100, 110, 1);

    ARMA_results = predict_ARMA(a, b);

    print(ARMA_results);

    a = pd.Series(a);
    b = pd.Series(b);

    DT_res = predict_tree(a, b);

    print(DT_res);