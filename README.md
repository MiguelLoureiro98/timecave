# Welcome to TimeCaVe :chart_with_upwards_trend:

TimeCaVe is a Python package that provides off-the-shelf model validation methods for time series modelling tasks.
These methods were conceived with time series forecasting in mind, though, in principle, they could also be applied to other time series related tasks.

The methods' generic implementation is flexible enough for them to be used with **ANY** kind of time series model, including classical time series forecasting 
models (e.g. AR models), classical machine learning models (e.g. decision trees) and deep learning models specifically tailored to handle sequence data (e.g. LSTMs).

In addition to the validation methods themselves, TimeCaVe provides several utility functions and classes. These include specific metrics to measure the performance of a validation method, data generation methods, functions to aid in the data collection process, and functions to extract important features from the data.

# Features

- **Versatility**: Our flexible, generic implementations are able to accomodate a wide variety of time series forecasting models.
- **Scikit-learn-like syntax**: TimeCaVe's validation methods are implemented in a similar way to those of Scikit-learn, thereby smoothing the learning curve.
- **Extra functionality**: In addition to splitting your data, TimeCaVe allows you to plot your partitioned data, compute relevant statistics for all training and validation sets, and access information regarding how much data is being used for training and validation. All this to ensure you make the best possible use of your data.
- **Data generation**: Generate synthetic time series data easily with TimeCaVe's data generation capabilities.
- **Validation metrics**: Several metrics are provided to help you select the most appropriate validation method, if need be.
- **Data collection**: TimeCaVe provides utility functions that can help determine the necessary amount of samples to capture a given dominant frequency.

# Installation

## Using pip

TimeCaVe can be directly installed from PyPi using pip:

```
pip install timecave
```

To install the development version, simply type:

```
pip install "timecave[dev]"
```

This will install dependencies that have been used to develop TimeCaVe and its documentation, such as Black and MKDocs.

## Using git

TimeCaVe can also be installed using git. To do so, clone the repository:

```
git clone https://github.com/MiguelLoureiro98/timecave.git
```

Then, move into the cloned repository and install the package using pip:

```
cd timecave
pip install .
```

Again, to install development dependencies, simply type:

```
pip install ".[dev]"
```

# Basic Usage

TimeCaVe is, above all else, built to provide easy-to-use validation methods for time series forecasting models. The syntax is relatively similar to that of the methods provided by Scikit-learn (e.g. K-fold). Here is an example of how to use one of the methods provided by this package (Block Cross-Validation):

```py
import numpy as np
from timecave.validation_methods.CV import BlockCV

ts = np.arange(0, 10)

# Split the data into 5 folds
splitter = BlockCV(5, ts);

for train, test in splitter.split():

    training_data = ts[train];
    validation_data = ts[test];

    # Train and validate your model
```

Methods to plot the partitioned time series and compute relevant statistics are also provided. 

For more information on how to use the package, check TimeCaVe's **DOCUMENTATION LINK**.

# Authors
The package was developed by Beatriz Lourenço and Miguel Loureiro, two graduate research assistants at [IST](https://tecnico.ulisboa.pt/en/), University of Lisbon, Portugal.

# Acknowledgements
This project would not have seen the light of day without the support of the Mechanical Engineering Institute ([IDMEC](https://www.idmec.tecnico.ulisboa.pt/)) and the 
Laboratory of Intelligent Systems for Data Analysis, Modeling and Optimization ([IS4](https://is4.tecnico.ulisboa.pt/)).

![IST_Logo](docs/images/IST_Logo_2_resized.png)

![IDMEC Logo](docs/images/IDMEC_PNG_resized.png)