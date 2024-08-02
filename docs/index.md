# TimeCaVe documentation

Welcome to TimeCaVe's documentation! The package that helps you validate your time series models!

## What is TimeCaVe?

TimeCaVe is a Python package that provides several off-the-shelf validation methods for time series modelling tasks.
These methods were conceived with time series forecasting in mind, though, in principle, they could also be applied to other time series related tasks.

The methods' generic implementation is flexible enough for them to be used with **ANY** kind of time series model, including classical time series forecasting 
models (e.g. AR models), classical machine learning models (e.g. decision trees) and deep learning models specifically tailored to handle sequence data (e.g. LSTMs).

In addition to the validation methods themselves, TimeCaVe provides several utility functions and classes [to be used at one's discretion]. These include specific metrics to measure the performance of a validation method, data generation methods, functions to aid in the data collection process, and functions to extract important features from the data.

## Why use TimeCaVe?

Model validation is a crucial step in the machine learning modelling process, and time series forecasting is no different. However, while implementations of model validation methods for tabular data abound, there is a shortage of publicly available packages that provide easy-to-use implementations of validation methods for time series data. This is exactly what TimeCaVe does.

### Main features:
- **Versatility**: Our flexible, generic implementations are able to accomodate a wide variety of time series forecasting models.
- **Scikit-learn-like syntax**: TimeCaVe's validation methods are implemented in a similar way to those of Scikit-learn, thereby smoothing the learning curve.
- **Extra functionality**: In addition to splitting your data, TimeCaVe allows you to plot your partitioned data, compute relevant statistics for all training and validation sets, and access information regarding how much data is being used for training and validation. All this to ensure you make the best possible use of your data.
- **Data generation**: Generate synthetic time series data easily with TimeCaVe's data generation capabilities [allow you to generate synthetic time series data without breaking a sweat].
- **Validation metrics**: Several metrics are provided to help you select the most appropriate validation method, if need be.
- **Data collection**: TimeCaVe provides utility functions that can help determine the necessary amount of samples to capture a given dominant frequency.

## Documentation Guide

For a quick overview of how to install and use TimeCaVe, please check our [Getting Started](starters.md) guide.

Detailed information about each function or class is provided in TimeCaVe's [API reference](API_ref/index.md).

For more information regarding the package authors, please refer to the [About](about.md) section.