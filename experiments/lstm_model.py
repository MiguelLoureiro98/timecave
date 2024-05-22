import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from experiment_utils import get_X_y


def lstm_model(lags: int):
    """
    Defines the lstm architecture to be used.
    """
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(lags, 1)))
    model.add(Dense(1))
    return model


def recursive_forecast(model, forecast_origin, pred_window: int, lags: int):
    """
    Performs recursive forecasting using a trained LSTM model,
    predicting 'pred_window' future timesteps based on single-step predictions.
    """
    # Make predictions
    forecast = []
    x_input = forecast_origin
    for _ in range(pred_window):
        yhat = model.predict(x_input, verbose=0)
        forecast.append(yhat[0][0])
        x_input = np.append(x_input[:, -lags:, :], [[yhat[0]]], axis=1)

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

    # Define the LSTM model
    model = lstm_model(lags)
    model.compile(optimizer="adam", loss="mse")

    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)

    forecast_origin = X_test[0]
    pred_window = len(y_test)

    forecast = recursive_forecast(model, forecast_origin, pred_window, lags)
    mse = mean_squared_error(y_test, forecast)
    mae = mean_absolute_error(y_test, forecast)

    print("Forecasted values for the next", len(test_series), "timesteps:")
    print(forecast)
    return {
        "prediction": np.array(forecast),
        "trained_model": model,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mae,
    }


if __name__ == "__main__":
    results = predict_lstm(
        pd.Series(np.arange(80)), pd.Series(np.arange(80, 100 + 1)), epochs=50
    )
    print(results)
