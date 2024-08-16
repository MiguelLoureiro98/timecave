"""
This module contains all the Prequential ('Predictive Sequential') validation methods supported by this package. \
These methods are also known as Forward Validation methods.

Classes
-------
GrowingWindow
    Implements every variant of the Growing Window method.

RollingWindow
    Implements every variant of the Rolling Window method.

See also
--------
[Out-of-Sample methods](../prequential/index.md): Out-of-sample methods for time series data.

[Cross-validation methods](../CV/index.md): Cross-validation methods for time series data.

[Markov methods](../markov/index.md): Markov cross-validation method for time series data.

Notes
-----
Predictive Sequential, or "Prequential", methods are one of the three main classes of validation methods for time series data (the others being \
out-of-sample methods and cross-validation methods).
Unlike cross-validation methods, this class of methods preserves the temporal order of observations, although it differs from out-of-sample methods in that it \
partitions the series into equally sized folds.
For more details on this class of methods, the reader should refer to [[1]](#1).

References
----------
##1
Vitor Cerqueira, Luis Torgo, and Igor Mozetiˇc. Evaluating time series forecasting models: An empirical study on performance estimation methods.
Machine Learning, 109(11):1997–2028, 2020.
"""

from .base import BaseSplitter
from .weights import constant_weights
from ..data_characteristics import get_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator


class GrowingWindow(BaseSplitter):
    """
    GrowingWindow(splits: int, ts: np.ndarray | pd.Series, fs: float | int, gap: int = 0, weight_function: callable = constant_weights, params: dict = None)
    --------------------------------------------------------------------------------------------------------------------------------------------------------

    _summary_

    _extended_summary_

    Parameters
    ----------
    BaseSplitter : _type_
        _description_

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        splits: int,
        ts: np.ndarray | pd.Series,
        fs: float | int = 1,
        gap: int = 0,
        weight_function: callable = constant_weights,
        params: dict = None,
    ) -> None:

        super().__init__(splits, ts, fs)
        self._check_gap(gap)
        self._gap = gap
        self._splitting_ind = self._split_ind()
        self._weights = weight_function(self.n_splits, self._gap, 1, params)

        return

    def _check_gap(self, gap: int) -> None:
        """
        Perform type and value checks for the 'gap' parameter.
        """

        if isinstance(gap, int) is False:

            raise TypeError("'gap' must be an integer.")

        if gap < 0:

            raise ValueError("'gap' must be non-negative.")

        if self.n_splits - gap < 2:

            raise ValueError(
                f"With {self.n_splits}, the maximum allowable gap is {self._n_splits - 2} splits."
            )

        return

    def _split_ind(self) -> np.ndarray:
        """
        Compute the splitting indices.
        """

        remainder = int(self._n_samples % self.n_splits)
        split_size = int(np.floor(self._n_samples / self.n_splits))
        #print(split_size)
        #print(remainder)
        split_ind = np.arange(split_size, self._n_samples, split_size)
        #split_ind[:remainder] += 1
        #plit_ind[remainder:] += remainder

        if(remainder != 0):

            if(split_ind.shape[0] > self.n_splits):

                split_ind = split_ind[:-1];
            
            split_ind[:remainder] += np.array([i for i in range(1, remainder+1)]);
            split_ind[remainder:] += remainder;

        else:

            split_ind = np.append(split_ind, self._n_samples);
        
        if(split_ind.shape[0] > self.n_splits):

            split_ind = split_ind[:self.n_splits];

        #print(split_ind)

        return split_ind

    def split(self) -> Generator[tuple, None, None]:
        """
        _summary_

        _extended_summary_

        Yields
        ------
        Generator[tuple, None, None]
            _description_
        """

        for i, (ind, weight) in enumerate(zip(self._splitting_ind[:-1], self._weights)):

            gap_ind = self._splitting_ind[i + self._gap]
            gap_end_ind = self._splitting_ind[i + self._gap + 1]

            train = self._indices[:ind]
            validation = self._indices[gap_ind:gap_end_ind]

            yield (train, validation, weight)

    def info(self) -> None:
        """
        _summary_

        _extended_summary_
        """

        min_fold_size = int(np.floor(self._n_samples / self.n_splits))
        max_fold_size = min_fold_size

        remainder = self._n_samples % self.n_splits

        if remainder != 0:

            max_fold_size += 1

        min_fold_size_pct = np.round(min_fold_size / self._n_samples * 100, 2)
        max_fold_size_pct = np.round(max_fold_size / self._n_samples * 100, 2)

        max_train = (
            min_fold_size * (self.n_splits - remainder - 1) + max_fold_size * remainder
        )
        max_train_pct = np.round(max_train / self._n_samples * 100, 2)

        print("Growing Window method")
        print("---------------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Number of splits: {self.n_splits}")
        print(
            f"Fold size: {min_fold_size} to {max_fold_size} samples ({min_fold_size_pct} to {max_fold_size_pct} %)"
        )
        print(
            f"Minimum training set size: {max_fold_size} samples ({max_fold_size_pct} %)"
        )
        print(f"Maximum training set size: {max_train} samples ({max_train_pct} %)")
        print(f"Gap: {self._gap}")
        print(f"Weights: {np.round(self._weights, 3)}")

        return

    def statistics(self) -> tuple[pd.DataFrame]:
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[pd.DataFrame]
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        if int(np.floor(self._n_samples / self.n_splits)) < 2:

            raise ValueError(
                "The folds are too small to compute most meaningful features."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        full_features = get_features(self._series, self.sampling_freq)
        training_stats = []
        validation_stats = []

        for (training, validation, _) in self.split():

            training_feat = get_features(self._series[training], self.sampling_freq)
            training_stats.append(training_feat)

            validation_feat = get_features(self._series[validation], self.sampling_freq)
            validation_stats.append(validation_feat)

        training_features = pd.concat(training_stats)
        validation_features = pd.concat(validation_stats)

        return (full_features, training_features, validation_features)

    def plot(self, height: int, width: int) -> None:
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        height : int
            The figure's height.

        width : int
            The figure's width.
        """

        fig, axs = plt.subplots(self.n_splits - self._gap - 1, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Growing Window method")

        if(self.n_splits - self._gap - 1 > 1):

            for it, (training, validation, weight) in enumerate(self.split()):

                axs[it].scatter(training, self._series[training], label="Training set")
                axs[it].scatter(
                    validation, self._series[validation], label="Validation set"
                )
                axs[it].set_title("Iteration: {} Weight: {}".format(it + 1, np.round(weight, 3)))
                axs[it].set_ylim([self._series.min() - 1, self._series.max() + 1])
                axs[it].set_xlim([- 1, self._n_samples + 1])
                axs[it].legend()

        else:

            for (training, validation, weight) in self.split():

                axs.scatter(training, self._series[training], label="Training set")
                axs.scatter(
                    validation, self._series[validation], label="Validation set"
                )
                axs.set_title("Iteration: {} Weight: {}".format(1, np.round(weight, 3)))
                axs.legend()

        plt.show()

        return


class RollingWindow(BaseSplitter):
    """
    RollingWindow(splits: int, ts: np.ndarray | pd.Series, fs: float | int, gap: int = 0, weight_function: callable = constant_weights, params: dict = None)
    --------------------------------------------------------------------------------------------------------------------------------------------------------

    _summary_

    _extended_summary_

    Parameters
    ----------
    splits : int
        Number of splits.

    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int
        Sampling frequency (Hz).

    gap : int, default=0
        __description__

    weight_function : callable, default=constant_weights
        __description__

    params : dict, optional
        __description__
    """

    def __init__(
        self,
        splits: int,
        ts: np.ndarray | pd.Series,
        fs: float | int = 1,
        gap: int = 0,
        weight_function: callable = constant_weights,
        params: dict = None,
    ) -> None:

        super().__init__(splits, ts, fs)
        self._check_gap(gap)
        self._gap = gap
        self._splitting_ind = self._split_ind()
        self._weights = weight_function(self.n_splits, self._gap, 1, params)

        return

    def _check_gap(self, gap: int) -> None:
        """
        Perform type and value checks for the 'gap' parameter.
        """

        if isinstance(gap, int) is False:

            raise TypeError("'gap' must be an integer.")

        if gap < 0:

            raise ValueError("'gap' must be non-negative.")

        if self.n_splits - gap < 2:

            raise ValueError(
                f"With {self.n_splits}, the maximum allowable gap is {self._n_splits - 2} splits."
            )

        return

    def _split_ind(self) -> np.ndarray:
        """
        Compute the splitting indices.
        """

        remainder = int(self._n_samples % self.n_splits)
        split_size = int(np.floor(self._n_samples / self.n_splits))
        split_ind = np.arange(0, self._n_samples, split_size)
        #split_ind[:remainder] += 1

        #if remainder != 0:

        #    split_ind[remainder:] += remainder

        #split_ind = np.append(split_ind, self._n_samples)

        if(remainder != 0):

            #if(split_ind.shape[0] > self.n_splits):

            #    split_ind = split_ind[:-1];
            
            split_ind[1:remainder+1] += np.array([i for i in range(1, remainder+1)]);
            split_ind[remainder+1:] += remainder;

        else:

            split_ind = np.append(split_ind, self._n_samples);

        #print(split_ind)

        if(split_ind.shape[0] > self.n_splits + 1):

            split_ind = split_ind[:self.n_splits+1];

        return split_ind

    def split(self) -> Generator[tuple, None, None]:
        """
        _summary_

        _extended_summary_

        Yields
        ------
        Generator[tuple, None, None]
            _description_
        """

        #print(self._splitting_ind)

        for i, (ind, weight) in enumerate(zip(self._splitting_ind[1:-1], self._weights)):

            gap_ind = self._splitting_ind[i + 1 + self._gap]
            gap_end_ind = self._splitting_ind[i + 1 + self._gap + 1]
            start_training_ind = self._splitting_ind[i]

            train = self._indices[start_training_ind:ind]
            validation = self._indices[gap_ind:gap_end_ind]

            #print("Bollocks");
            #print(train)
            #print(validation)

            yield (train, validation, weight)

    def info(self) -> None:
        """
        _summary_

        _extended_summary_
        """

        min_fold_size = int(np.floor(self._n_samples / self.n_splits))
        max_fold_size = min_fold_size

        remainder = self._n_samples % self.n_splits

        if remainder != 0:

            max_fold_size += 1

        min_fold_size_pct = np.round(min_fold_size / self._n_samples * 100, 2)
        max_fold_size_pct = np.round(max_fold_size / self._n_samples * 100, 2)

        print("Rolling Window method")
        print("---------------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Number of splits: {self.n_splits}")
        print(
            f"Fold size: {min_fold_size} to {max_fold_size} samples ({min_fold_size_pct} to {max_fold_size_pct} %)"
        )
        print(f"Gap: {self._gap}")
        print(f"Weights: {np.round(self._weights, 3)}")

        return

    def statistics(self) -> tuple[pd.DataFrame]:
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[pd.DataFrame]
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        if int(np.round(self._n_samples / self.n_splits)) < 2:

            raise ValueError(
                "The folds are too small to compute most meaningful features."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        full_features = get_features(self._series, self.sampling_freq)
        training_stats = []
        validation_stats = []

        for (training, validation, _) in self.split():

            training_feat = get_features(self._series[training], self.sampling_freq)
            training_stats.append(training_feat)

            validation_feat = get_features(self._series[validation], self.sampling_freq)
            validation_stats.append(validation_feat)

        training_features = pd.concat(training_stats)
        validation_features = pd.concat(validation_stats)

        return (full_features, training_features, validation_features)

    def plot(self, height: int, width: int) -> None:
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        height : int
            The figure's height.

        width : int
            The figure's width.
        """

        fig, axs = plt.subplots(self.n_splits - self._gap - 1, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Rolling Window method")

        if(self.n_splits - self._gap - 1 > 1):

            for it, (training, validation, weight) in enumerate(self.split()):

                axs[it].scatter(training, self._series[training], label="Training set")
                axs[it].scatter(
                    validation, self._series[validation], label="Validation set"
                )
                axs[it].set_title("Iteration: {} Weight: {}".format(it + 1, np.round(weight, 3)))
                axs[it].set_ylim([self._series.min() - 1, self._series.max() + 1])
                axs[it].set_xlim([- 1, self._n_samples + 1])
                axs[it].legend()

        else:

            for (training, validation, weight) in self.split():

                axs.scatter(training, self._series[training], label="Training set")
                axs.scatter(
                    validation, self._series[validation], label="Validation set"
                )
                axs.set_title("Iteration: {} Weight: {}".format(1, np.round(weight, 3)))
                axs.legend()

        plt.show()

        return
