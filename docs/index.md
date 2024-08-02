# TimeCaVe documentation

Welcome to the TimeCaVe documentation! The package that helps you validate your time series models!

## What is TimeCaVe?

TimeCaVe is a Python package that provides several off-the-shelf validation methods for time series modelling tasks.
These methods were conceived with time series forecasting in mind, though, in principle, they could also be applied to other time series related tasks.

The methods' generic implementation is flexible enough for them to be used with **ANY** kind of time series model, including classical time series forecasting 
models (e.g. AR models), classical machine learning models (e.g. decision trees) and deep learning models specifically tailored to handle sequence data (e.g. LSTMs).

In addition to the validation methods themselves, TimeCaVe provides several utility functions and classes [to be used at one's discretion]. These include specific metrics to measure the performance of a validation method, data generation methods, functions to aid in the data collection process, and functions to extract important features from the data.

## Why use TimeCaVe?

Model validation is a crucial step in the machine learning modelling process, and time series forecasting is no different. However, while implementations of model validation methods for tabular data abound, there is a shortage of publicly available packages that provide easy-to-use implementations of validation methods for time series data. This is exactly what TimeCaVe does.

### Main features:
- **Flexibility**: Our implementations are generic and flexible. ... .
- **Statistics/Info/Plot**: ... .
- **Data generation**: ... .
- **Validation metrics**: ... .
- **Data collection**: ... .

## Documentation Guide

For a quick overview ... [Getting Started](starters.md) guide.

Detailed information about each function or class is provided in TimeCaVe's [API reference](API_ref/index.md).

For more information regarding the package authors, please refer to the [About](about.md) section.