---
hide:
    - toc
---

# Validation methods

This subpackage contains all the model validation methods provided by this package. When importing a validation method, be sure to specify the correct module, like so:

``` py
from timecave.validation_methods.OOS import Holdout
```

Out-of-Sample and Prequential methods ensure the training set always precedes the validation set. This is not the case for cross-validation and Markov methods.

## Modules
- [Base](base/base.md): Contains the class that serves as the basis for every validation method.
- [OOS](OOS/index.md): Implements Out-of-Sample (OOS) methods.
- [Prequential](prequential/index.md): Implements prequential methods (also known as forward validation methods).
- [CV](CV/index.md): Implements cross-validation (CV) methods.
- [Markov](markov/index.md): Implements the Markov cross-validation method.
- [Weight functions](weights/index.md): Provides off-the-shelf weighting functions for use with CV and prequential methods.