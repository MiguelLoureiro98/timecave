from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

"""
This module contains the base class for all time series validation methods provided / supported by this package.
This class is simply an abstract class and should not be used directly (i.e. should not be made available to the user).
"""

# This should work with sklearn's Hyperparameter Search algorithms. If not, install sklearn and inherit from the BaseCrossValidator class (maybe?).
# For now, leave it as it is, as this approach will most likely work with said search algorithms and leads to fewer requirements.

class base_splitter(ABC):
    
    """
    Base class for all time series validation methods provided / supported by this package.

    This is simply an abstract class. As such, it should not be used directly.

    Attributes
    ----------
    _n_splits : int
        Number of splits for this instance of the validation method.

    Methods
    -------
    __init__(self, splits: int) -> None
        Class constructor.

    n_splits(self) -> int
        Returns the number of splits (set during initialisation).

    split(self) -> tuple[list]


    info(self) -> None


    statistics(self) -> None


    plot(self) -> None

    """

    def __init__(self, splits: int) -> None:

        """
        Class constructor.

        This is the constructor of the base_splitter class.

        Parameters
        ----------
        splits : int
            The number of splits.
        """

        super().__init__();
        self._n_splits = splits;
    
        return;

    @property
    def n_splits(self) -> int:
        
        """
        Get the number of splits for a given instance of a validation method.

        This method can be used to retrieve the number of splits for a given instance of a validation method (this is set on initialisation).
        Since the method is implemented as a property, this information can simply be accessed as an attribute using dot notation.

        Returns
        -------
        int
            The number of splits.
        """

        return self._n_splits;

    # If the sklearn-like interface is to be kept, then this version of the 'get_n_splits' should be implemented.

    #def get_n_splits(self) -> int:
        
        """
        Get the number of splits for a given instance of a validation method.

        This method can be used to retrieve the number of splits ... .

        Returns
        -------
        int
            The number of splits.
        """

        return self._n_splits;

    @abstractmethod
    def split(self) -> tuple[list]:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[list]
            _description_
        """

        pass

    @abstractmethod
    def info(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        pass

    @abstractmethod
    def statistics(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        pass

    @abstractmethod
    def plot(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        pass

if __name__ == "__main__":

    def split(X: np.ndarray, splits: int=5):

        samples = X.shape[0];
        indices = np.arange(0, samples);
        samples_per_index = int(np.round(samples / splits));

        for split in range(splits-1):

            train_ind = indices[split * samples_per_index:(split + 1) * samples_per_index];
            test_ind = indices[train_ind + np.fmin(samples_per_index, samples - train_ind[-1])];

            yield train_ind, test_ind;

    X = np.ones(10);

    for train, test in split(X):

        print(train);
        print(test);

    print(np.arange(13-5*2, 13, 2));