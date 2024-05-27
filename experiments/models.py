import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from experiment_utils import get_X_y
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from timecave.validation_methods._base import base_splitter


def lstm_model(lags: int):
    """
    Defines the LSTM architecture to be used.
    """
    model = Sequential()
    model.add(Input(shape=(lags, 1)))
    model.add(LSTM(50, activation="relu"))
    model.add(Dense(1))
    return model


def recursive_forecast(model, forecast_origin, pred_window: int, lags: int):
    """
    Performs recursive forecasting using a trained LSTM model,
    predicting 'pred_window' future timesteps based on single-step predictions.
    """
    # Make predictions
    forecast = []
    x_input = forecast_origin.reshape(
        (1, forecast_origin.shape[0], forecast_origin.shape[1])
    )
    for _ in range(pred_window):
        yhat = model.predict(x_input, verbose=0)
        pred = yhat[0][0]
        forecast.append(pred)
        x_input = np.append(x_input[:, -lags + 1 :, :], [[[pred]]], axis=1)

    return forecast


def predict_lstm(
    train_series: pd.Series or pd.DataFrame,
    test_series: pd.Series or pd.DataFrame,
    lags: int = 3,
    epochs: int = 200,
    verbose: int = 0,
) -> np.array:
    """
    Predict future values using Long Short-Term Memory (LSTM).
    """

    X_test, y_test = get_X_y(test_series, lags)
    X_train, y_train = get_X_y(train_series, lags)

    # Reshape input to be [samples, time steps, features]
    n_features = 1  # univariate time series
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    # LSTM model
    model = lstm_model(lags)
    model.compile(optimizer="adam", loss="mse")

    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)

    forecast_origin = X_test[0]
    pred_window = len(y_test)

    forecast = recursive_forecast(model, forecast_origin, pred_window, lags)
    mse = mean_squared_error(y_test, forecast)
    mae = mean_absolute_error(y_test, forecast)

    return {
        "prediction": np.array(forecast),
        "trained_model": model,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mae,
    }


def recursive_forecast_tree(
    ts_val: np.ndarray, pred_window: int, model: DecisionTreeRegressor
) -> np.ndarray:
    """
    Recursive forecasting for decision trees.
    """

    forecasts = np.zeros(pred_window)
    input = ts_val[0]

    for ind in range(pred_window):

        forecasts[ind] = model.predict(input.reshape(1, -1)).item()
        input = np.hstack((input[1:], forecasts[ind]))

    return forecasts


def predict_tree(ts_train: pd.Series, ts_val: pd.Series) -> dict:
    """
    Train and test a decision tree model.
    """

    # reshaped_train = shape_series(ts_train);
    # reshaped_val = shape_series(ts_val);
    X_train, y_train = get_X_y(ts_train)
    X_val, y_val = get_X_y(ts_val)

    model = DecisionTreeRegressor()

    model.fit(X_train, y_train)

    y_pred = recursive_forecast_tree(X_val, y_val.shape[0], model)
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
):

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

    lstm_results = predict_lstm(train, val, lags=5, epochs=5, verbose=0)
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

    return tree_results["model"], lstm_results["model"], ARMA_results["model"]


if __name__ == "__main__":

    a = np.ones(100)
    b = np.ones(10)

    a = np.append(np.arange(100), np.arange(100, 120))
    b = np.arange(100, 110, 1)

    a = np.arange(100)
    b = np.arange(100, 120)

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
