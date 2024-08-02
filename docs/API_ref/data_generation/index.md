---
hide:
    - toc
---

# Data generation

This subpackage contains all the routines that allow the user to generate synthetic time series data.

In most cases, using the time series generator class will suffice. In order to do so, type:

```py
from timecave.data_generation.time_series_generation import TimeSeriesGenerator
```

## Modules:
- [Frequency Modulation](frequency_modulation/index.md): Contains classes used to generate time-varying sinusoids.
- [Time series functions](time_series_functions/index.md): Implements the functions that can be used to generate data.
- [Time series generation](time_series_generation/index.md): Contains the generator class that can be used to generate synthetic time series data easily.