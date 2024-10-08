# Getting Started

## Installation

### Using pip

TimeCaVe can be directly installed from PyPi using pip:

```
pip install timecave
```

To install the development version, simply type:

```
pip install "timecave[dev]"
```

This will install dependencies that have been used to develop TimeCaVe and its documentation, such as Black and MKDocs.

### Using git

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

## Basic Usage

TimeCaVe is, above all else, built to provide easy-to-use validation methods for time series forecasting models. The syntax is relatively similar to that of the methods provided by Scikit-learn (e.g. K-fold). Here is an example of how to use one of the methods provided by this package (Block Cross-Validation):

```py
import numpy as np
from timecave.validation_methods.CV import BlockCV

ts = np.arange(0, 10)

# Split the data into 5 folds
splitter = BlockCV(5, ts);

for train, test, _ in splitter.split():

    training_data = ts[train];
    validation_data = ts[test];

    # Train and validate your model
```

For more information on how to use the package, please refer to our [API reference](API_ref/index.md), where detailed descriptions of every function and class are provided.