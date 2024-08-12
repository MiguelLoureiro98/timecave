"""
This module contains the base class for all time series validation methods provided / supported by this package.
This class is simply an abstract class and should not be used directly (i.e. should not be made available to the user).

Classes
-------
BaseSplitter
    Abstract base class for every validation method supported by this package.
"""

from abc import ABC, abstractmethod
from typing import Generator
import numpy as np
import pandas as pd

# This should work with sklearn's Hyperparameter Search algorithms. If not, install sklearn and inherit from the BaseCrossValidator class (maybe?).
# For now, leave it as it is, as this approach will most likely work with said search algorithms and leads to fewer requirements.


class BaseSplitter(ABC):
    """
    Base class for all time series validation methods supported by this package.

    This is simply an abstract class. As such, it should not be used directly.

    Parameters
    ----------
    splits : int
        The number of splits.

    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        The series' sampling frequency (Hz).

    Attributes
    ----------
    n_splits
        The number of splits.

    sampling_freq
        The series' sampling frequency (Hz).

    Methods
    -------
    split()
        Abstract method. The implementation differs for each validation method.

    info()            
        Abstract method. The implementation differs for each validation method.

    statistics() 
        Abstract method. The implementation differs for each validation method.

    plot()
        Abstract method. The implementation differs for each validation method.

    Raises
    ------
    TypeError
        If the number of splits is not an integer.

    ValueError
        If the number of splits is smaller than two.

    TypeError
        If ts is not a Numpy array nor a Pandas series.

    TypeError
        If the sampling frequency is neither a float nor an integer.
    
    ValueError
        If the sampling frequency is negative.

    ValueError
        If the time series is multivariate (i.e. not one-dimensional).

    ValueError
        If the number of splits is larger than the number of samples in the time series.
    """

    def __init__(
        self, splits: int, ts: np.ndarray | pd.Series, fs: float | int = 1
    ) -> None:
        
        """
        Class constructor.
        """

        super().__init__()
        self._splits_check(splits)
        self._ts_check(ts)
        self._fs_check(fs)
        self._n_splits = splits
        self._series = ts
        self._fs = fs
        self._n_samples = self._series.shape[0]
        self._dim_check()
        self._indices = np.arange(0, self._n_samples)

        return

    def _splits_check(self, splits: int) -> None:
        """
        Perform type and value checks on the 'splits' parameter.
        """

        if isinstance(splits, int) is False:

            raise TypeError("'splits' must be an integer.")

        elif splits < 2:

            raise ValueError("'splits' must be equal to or larger than two.")

        return

    def _ts_check(self, ts: np.ndarray | pd.Series) -> None:
        """
        Perform type and value checks on the 'ts' parameter.
        """

        if isinstance(ts, np.ndarray) is False and isinstance(ts, pd.Series) is False:

            raise TypeError("'ts' must be either a Numpy array or a Pandas series.")

        elif len(ts.shape) > 1:

            raise ValueError(
                "The time series must be univariate. Support for multivariate time series has yet to be implemented.\
                              If the input is supposed to be a one-dimensional numpy array, consider flattening the array."
            )

        return

        # def _ts_formatting(self, ts: np.ndarray | pd.Series) -> np.ndarray | pd.Series:

        time_series = ts.copy()

        if isinstance(time_series, np.ndarray) and len(time_series.shape) == 2:

            time_series = time_series.flatten()

        return time_series

    def _dim_check(self) -> None:
        """
        Perform a dimension check on the time series.
        """

        if self._n_splits > self._n_samples:

            raise ValueError(
                "The number of splits must be smaller than (or equal to) the number of samples in the time series."
            )

        return

    def _fs_check(self, fs: int | float) -> None:
        """
        Perform type and value checks on the series' sampling frequency.
        """

        if (isinstance(fs, float) or isinstance(fs, int)) is False:

            raise TypeError(
                "The sampling frequency should be either a float or an integer."
            )

        if fs <= 0:

            raise ValueError("The sampling frequency should be larger than zero.")

        return

        # This should only be defined for prequential and CV methods!
        # def _split_folds(self) -> list[np.ndarray]:

        samples_per_fold = self._n_samples / self._n_splits
        extra_samples = self._n_samples // self._n_splits

        indices = np.arange(0, self._n_samples)
        extra = np.zeros(self._n_splits)
        extra[:extra_samples] = 1
        folds = []
        curr_ind = 0

        for split in range(self._n_splits):

            next_ind = split * (samples_per_fold + 1) + extra[split]
            folds.append(indices[curr_ind:next_ind])
            curr_ind = next_ind

        return folds

    @property
    def n_splits(self) -> int:
        """
        Get the number of splits for a given instance of a validation method.

        This method can be used to retrieve the number of splits for a given
        instance of a validation method (this is set on initialisation).
        Since the method is implemented as a property, this information can
        simply be accessed as an attribute using dot notation.

        Returns
        -------
        int
            The number of splits.
        """

        return self._n_splits

    @property
    def sampling_freq(self) -> int | float:
        """
        Get the time series' sampling frequency.

        This method can be used to access the time series' sampling
        frequency, in Hertz (this is set on intialisation).
        Since the method is implemented as a property, this information can
        simply be accessed as an attribute using dot notation.

        Returns
        -------
        int | float
            The time series' sampling frequency (Hz).
        """

        return self._fs

        # If the sklearn-like interface is to be kept, then this version of the 'get_n_splits' should be implemented.

        # def get_n_splits(self) -> int:

        """
        Get the number of splits for a given instance of a validation method.

        This method can be used to retrieve the number of splits ... .

        Returns
        -------
        int
            The number of splits.
        """

        return self._n_splits

    @abstractmethod
    def split(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split the time series into training and validation sets.

        Abstract method. The implementation differs for each validation method.

        Yields
        ------
        np.ndarray
            Array of training indices.

        np.ndarray
            Array of validation indices.
        """

        pass

    @abstractmethod
    def info(self) -> None:
        """
        Provide additional information on the validation method.

        Abstract method. The implementation differs for each validation method.
        """

        pass

    @abstractmethod
    def statistics(self) -> tuple[pd.DataFrame]:
        """
        Compute and plot relevant statistics for both training and validation sets.

        Abstract method. The implementation differs for each validation method.

        Returns
        -------
        pd.DataFrame
            Relevant statistics and features for training sets.

        pd.DataFrame
            Relevant statistics and features for validation sets.
        """

        pass

    @abstractmethod
    def plot(self, height: int, width: int) -> None:
        """
        Plot the partitioned time series.

        Abstract method. The implementation differs for each validation method.

        Parameters
        ----------
        height : int
            Figure height.

        width : int
            Figure width.
        """

        pass


if __name__ == "__main__":

    def split(X: np.ndarray, splits: int = 5):

        samples = X.shape[0]
        indices = np.arange(0, samples)
        samples_per_index = int(np.round(samples / splits))

        for split in range(splits - 1):

            train_ind = indices[
                split * samples_per_index : (split + 1) * samples_per_index
            ]
            test_ind = indices[
                train_ind + np.fmin(samples_per_index, samples - train_ind[-1])
            ]

            yield train_ind, test_ind

    X = np.ones(10)

    for train, test in split(X):

        print(train)
        print(test)

    print(np.arange(13 - 5 * 2, 13, 2))
