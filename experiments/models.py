import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from experiment_utils import get_X_y
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, GRU, SimpleRNN, LSTM
from timecave.validation_methods._base import base_splitter
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
import time


def rnn_model(lags: int):
    """
    Defines the LSTM architecture to be used.
    """
    model = Sequential()
    model.add(Input(shape=(lags, 1)))
    model.add(SimpleRNN(50))
    model.add(Dense(1))
    return model


def recursive_forecast(
    ts_val: np.ndarray,
    pred_window: int,
    model: DecisionTreeRegressor,
    args: dict = {},
) -> np.ndarray:
    """
    Recursive forecasting for decision trees.
    """

    forecasts = np.zeros(pred_window)
    input = ts_val[0]

    for ind in range(pred_window):

        forecasts[ind] = model.predict(input.reshape(1, -1), **args).item()
        input = np.hstack((input[1:], forecasts[ind]))

    return forecasts


def predict_lstm(
    train_series: pd.Series or pd.DataFrame,
    val_series: pd.Series or pd.DataFrame,
    lags: int = 5,
    epochs: int = 200,
    verbose: int = 0,
    one_step_head_eval: bool = True,
) -> np.array:
    """
    Predict future values using Long Short-Term Memory (LSTM).
    """

    X_val, y_val = get_X_y(val_series, lags)
    X_train, y_train = get_X_y(train_series, lags)

    # LSTM model
    model = rnn_model(lags)
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)

    pred_window = len(y_val)

    # Forecast
    if not one_step_head_eval:
        forecast = recursive_forecast(
            X_val, pred_window, model, args={"verbose": verbose}
        )
    else:
        forecast = model.predict(X_val, verbose=0)
    mse = mean_squared_error(y_val, forecast)
    mae = mean_absolute_error(y_val, forecast)

    return {
        "prediction": np.array(forecast),
        "model": model,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mae,
    }


def predict_tree(
    ts_train: pd.Series, ts_val: pd.Series, one_step_head_eval: bool = True
) -> dict:
    """
    Train and test a decision tree model.
    """

    # reshaped_train = shape_series(ts_train);
    # reshaped_val = shape_series(ts_val);
    X_train, y_train = get_X_y(ts_train)
    X_val, y_val = get_X_y(ts_val)

    model = DecisionTreeRegressor()

    model.fit(X_train, y_train)

    if not one_step_head_eval:
        y_pred = recursive_forecast(X_val, y_val.shape[0], model)
    else:
        y_pred = model.predict(X_val)
    mse = mean_squared_error(y_true=y_val, y_pred=y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true=y_val, y_pred=y_pred)

    return {"prediction": y_pred, "model": model, "mse": mse, "rmse": rmse, "mae": mae}


def next_pred(obs, model_params, kwargs):
    fake_model = SARIMAX(obs, **kwargs)
    res = fake_model.filter(model_params)
    return res.forecast(1).item()


def predict_ARMA_osh(
    ts_train: pd.Series | np.ndarray,
    ts_val: pd.Series | np.ndarray,
    n_lags: int = 5,
    one_step_head_eval: bool = True,
) -> dict:

    X_val, y_val = get_X_y(ts_val)

    # Define SARIMAX model parameters
    kwargs = {"order": (n_lags, 0, n_lags)}
    # kwargs = {"lags": n_lags}

    # Initialize SARIMAX model with initial data
    model = SARIMAX(ts_train, **kwargs)
    res_fit = model.fit()
    params = res_fit.params

    y_pred = np.apply_along_axis(
        lambda obs: next_pred(obs, params, kwargs), axis=1, arr=X_val
    )

    mse = mean_squared_error(y_true=y_val, y_pred=y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true=y_val, y_pred=y_pred)

    return {"prediction": y_pred, "model": model, "mse": mse, "rmse": rmse, "mae": mae}


def predict_ARMA(
    ts_train: pd.Series | np.ndarray, ts_val: pd.Series | np.ndarray, n_lags: int = 5
) -> dict:
    """
    Train and test an ARMA model.
    """

    model = ARIMA(ts_train, order=(n_lags, 0, n_lags))
    res = model.fit()

    y_pred = res.forecast(ts_val.shape[0])

    mse = mean_squared_error(y_true=ts_val, y_pred=y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true=ts_val, y_pred=y_pred)

    return {"prediction": y_pred, "model": model, "mse": mse, "rmse": rmse, "mae": mae}


def predict_models(
    train: pd.Series,
    val: pd.Series,
    filename,
    col_idx,
    table: pd.DataFrame,
    method: base_splitter = None,
    it: int = None,
    models: list[str] = ["ARMA", "LSTM", "Tree"],
):
    """
    Runs all models and saves results to the given table.
    """
    if "tree" in models:
        tree_results = predict_tree(train, val)
        row = pd.Series(
            {
                "filename": filename,
                "column_index": col_idx,
                "method": method,
                "iteration": it,
                "model": "Tree",
                "mse": tree_results["mse"],
                "mae": tree_results["mae"],
                "rmse": tree_results["rmse"],
            }
        )
        table.loc[len(table.index)] = row[table.columns]

    if "ARMA" in models:
        ARMA_results = predict_ARMA(train, val, n_lags=5)

        row = pd.Series(
            {
                "filename": filename,
                "column_index": col_idx,
                "method": method,
                "iteration": it,
                "model": "ARMA",
                "mse": ARMA_results["mse"],
                "mae": ARMA_results["mae"],
                "rmse": ARMA_results["rmse"],
            }
        )
        table.loc[len(table.index)] = row[table.columns]

    if "LSTM" in models:
        lstm_results = predict_lstm(train, val, lags=5, epochs=50, verbose=0)
        row = pd.Series(
            {
                "filename": filename,
                "column_index": col_idx,
                "method": method,
                "iteration": it,
                "model": "LSTM",
                "mse": lstm_results["mse"],
                "mae": lstm_results["mae"],
                "rmse": lstm_results["rmse"],
            }
        )
        table.loc[len(table.index)] = row[table.columns]

    return


if __name__ == "__main__":

    a = np.ones(100)
    b = np.ones(10)

    a = np.append(np.arange(100), np.arange(100, 120))
    b = np.arange(100, 110, 1)

    a = np.arange(1000)
    b = np.arange(1000, 1200)

    a = pd.Series(a)
    b = pd.Series(b)

    # --------------------- TIME -------------------- #
    start_time = time.time()
    # --------------------- TIME -------------------- #
    ARMA_results = predict_ARMA_osh(a, b)

    # --------------------- TIME -------------------- #
    end_time = time.time()
    runtime_seconds = end_time - start_time

    minutes = int(runtime_seconds // 60)
    seconds = int(runtime_seconds % 60)
    print(f"ARIMA: {minutes} minutes {seconds} seconds")
    # --------------------- TIME -------------------- #

    ARMA_results = predict_ARMA(a, b)

    print(ARMA_results)

    a = pd.Series(a)
    b = pd.Series(b)

    DT_res = predict_tree(a, b)

    print(DT_res)

    results = predict_lstm(
        pd.Series(np.arange(80)), pd.Series(np.arange(80, 100 + 1)), epochs=200
    )
    print(results)
