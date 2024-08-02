# Getting Started

## Installation

TimeCaVe can be directly installed from PyPi using pip:

```
pip install timecave
```

To install the development version, simply type:

```
pip install "timecave[dev]"
```

This will install dependencies that have been used to develop TimeCaVe and its documentation, such as Black and MKDocs.

## Basic Usage

TimeCaVe is, above all, built to provide easy-to-use validation methods for time series forecasting models. The syntax is relatively similar to that of the methods provided by Scikit-learn (e.g. K-fold). Here is an example of how to use one of the methods provided by this package (Block Cross-Validation):

```py
import numpy as np
from timecave.validation_methods.CV import BlockCV

ts = np.arange(0, 10)

splitter = BlockCV(ts);

for train, test in splitter.split():

    training_data = data[train];
    validation_data = data[test];

    # Train and validate your model
```

For more information on how to use the package, please refer to our [API reference](API_ref/index.md), where detailed descriptions of every function and class are provided.