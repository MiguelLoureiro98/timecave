"""
This module contains all the Markov cross-validation method.

Classes
-------
MarkovCV

"""

from tsvalidation.validation_methods._base import base_splitter
from tsvalidation.data_characteristics import get_features

import numpy as np
import pandas as pd
from typing import Generator
import matplotlib.pyplot as plt


class MarkovCV(base_splitter):
    def __init__(
        self, ts: np.ndarray | pd.Series, fs: float | int, p: int, seed: int
    ) -> None:

        if p % 3 == 0:
            self._m = int(2 * p / 3) + 1
        else:
            self._m = int(2 * p / 3) + 2

        self.n_subsets = (
            2 * self._m
        )  # total number of subsets (training + tests subsets)
        splits = 2 * self._m  # due to 2-fold CV
        super().__init__(splits, ts, fs)
        self._p = p
        self._seed = seed

    def _markov_iteration(self, n):
        np.random.seed(self._seed)

        d = np.zeros(n, dtype=int)

        i, j = 1, -1
        if np.random.rand() < 0.25:
            d[0], d[1] = i, i + 1
        elif np.random.rand() < 0.5:
            d[0], d[1] = i, j - 1
        elif np.random.rand() < 0.75:
            d[0], d[1] = j - 1, i
        else:
            d[0], d[1] = j - 1, j - 1

        for t in range(2, n):
            rd = np.random.rand()
            if (d[t - 1] > 0) and (d[t - 2] > 0):
                d[t] = j
                j -= 1
            elif (d[t - 1] < 0) and (d[t - 2] < 0):
                d[t] = i
                i += 1
            elif rd > 0.5:
                d[t] = j
                j -= 1
            else:
                d[t] = i
                i += 1

        return d

    def _markov_partitions(self):
        n = self._n_samples

        d = self._markov_iteration(n)

        Id = np.mod(d, self._m) + 1 + np.where(d > 0, 1, 0) * self._m

        Su = {}
        for u in range(1, 2 * self._m + 1):
            Su[u] = np.where(Id == u)[0]

        self._suo = {}
        self._sue = {}
        for u in range(1, self._m + 1):
            self._suo[u] = Su[u * 2 - 1]
            self._sue[u] = Su[u * 2]

        pass

    def split(self) -> Generator[tuple, None, None]:
        self._markov_partitions()
        for i in range(1, len(self._suo.items()) + 1):
            train, validation = self._suo[i], self._sue[i]
            yield (train, validation)
            train, validation = self._sue[i], self._suo[i]
            yield (train, validation)  # two-fold cross validation

    def info(self) -> None:
        lengths = []
        for i in range(1, len(self._suo.items()) + 1):
            lengths.extend([len(self._suo[i]), len(self._sue[i])])

        print("Markov CV method")
        print("---------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Number of splits: {self.n_splits}")
        print(f"Number of observations per set: {min(lengths)} to {max(lengths)}")
        pass

    def statistics(self) -> tuple[pd.DataFrame]:

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

        fig, axs = plt.subplots(self.n_splits, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Markov CV method")

        for it, (training, validation) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Fold {}".format(it + 1))
            axs[it].legend()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return


if __name__ == "__main__":
    mcv = MarkovCV(ts=np.ones(50), fs=0.2, p=4, seed=1)
    mcv.split()
    mcv.plot(2, 10)
    mcv.info()
    mcv.statistics()
    print()
