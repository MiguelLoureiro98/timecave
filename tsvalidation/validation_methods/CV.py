"""
This module contains all the CV ('Cross-Validation') validation methods supported by this package.

Classes
-------
Block_CV

hv_Block_CV

"""

from ._base import base_splitter
from .weights import constant_weights
from ..data_characteristics import get_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator


class Block_CV(base_splitter):
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    base_splitter : _type_
        _description_
    """

    def __init__(
        self,
        splits: int,
        ts: np.ndarray | pd.Series,
        fs: float | int,
        weight_function: callable = constant_weights,
    ) -> None:
        super().__init__(splits, ts, fs)
        self._splitting_ind = self._split_ind()
        self._weights = weight_function(self.n_splits)

    def _split_ind(self) -> np.ndarray:
        """
        Compute the splitting indices.
        """

        remainder = int(self._n_samples % self.n_splits)
        split_size = int(np.round(self._n_samples / self.n_splits))
        split_ind = np.arange(0, self._n_samples, split_size)
        split_ind[:remainder] += 1

        if remainder != 0:

            split_ind[remainder:] += remainder

        split_ind = np.append(split_ind, self._n_samples)

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

        for i, ind, weight in enumerate(zip(self._splitting_ind[:-1]), self._weights):

            next_ind = self._splitting_ind[i + 1]

            validation = self._indices[ind:next_ind]
            train = np.array([el for el in self._indices if el not in validation])

            yield (train, validation, weight)

    def info(self) -> None:
        """
        _summary_

        _extended_summary_
        """

        min_fold_size = int(np.round(self._n_samples / self.n_splits))
        max_fold_size = min_fold_size

        remainder = self._n_samples % self.n_splits

        if remainder != 0:

            max_fold_size += 1

        min_fold_size_pct = np.round(min_fold_size / self._n_samples, 4) * 100
        max_fold_size_pct = np.round(max_fold_size / self._n_samples, 4) * 100

        print("Block CV method")
        print("---------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Number of splits: {self.n_splits}")
        print(
            f"Fold size: {min_fold_size} to {max_fold_size} samples ({min_fold_size_pct} to {max_fold_size_pct} %)"
        )
        print(f"Weights: {self._weights}")

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

        full_features = get_features(self._series, self.sampling_freq)
        training_stats = []
        validation_stats = []

        for training, validation in self.split():

            if self._series[training].shape[0] >= 2:

                training_feat = get_features(self._series[training], self.sampling_freq)
                training_stats.append(training_feat)

            else:

                print(
                    "The training set is too small to compute most meaningful features."
                )

            if self._series[validation].shape[0] >= 2:

                validation_feat = get_features(
                    self._series[validation], self.sampling_freq
                )
                validation_stats.append(validation_feat)

            else:

                print(
                    "The validation set is too small to compute most meaningful features."
                )

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
        fig.suptitle("Block CV method")

        for it, (training, validation) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Iteration {}".format(it + 1))
            axs[it].legend()

        plt.show()

        return
