import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def split_sequence_for_lstm(sequence, lags):
    """
    Reshapes input in order to be used for the lstm.
    """
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + lags
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def lstm_model(lags: int):
    """
    Defines the lstm architecture to be used.
    """
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(lags, 1)))
    model.add(Dense(1))
    return model


def recursive_forecast(model, series, pred_window, lags):
    """
    Performs recursive forecasting using a trained LSTM model,
    predicting 'pred_window' future timesteps based on single-step predictions.
    """
    # Make predictions
    forecast = []
    x_input = series[-lags:].reshape((1, lags, 1))
    for _ in range(pred_window):
        yhat = model.predict(x_input, verbose=0)
        forecast.append(yhat[0][0])
        x_input = np.append(x_input[:, 1:, :], [[yhat[0]]], axis=1)


def predict_lstm(
    series: np.array, pred_window: int, lags: int = 3, epochs: int = 200
) -> np.array:
    """
    Predict future values using Long Short-Term Memory (LSTM).
    """
    X, y = split_sequence_for_lstm(series, lags)

    # Reshape input to be [samples, time steps, features]
    n_features = 1  # univariate time series
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Define the LSTM model
    model = lstm_model(lags, n_features)
    model.compile(optimizer="adam", loss="mse")

    # Fit the model
    model.fit(X, y, epochs=epochs, verbose=0)

    forecast = recursive_forecast(model, series, pred_window, lags)

    print("Forecasted values for the next", pred_window, "timesteps:")
    print(forecast)
    return np.array(forecast)
