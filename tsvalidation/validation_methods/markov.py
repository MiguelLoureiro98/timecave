"""
This module contains all the Markov cross-validation method.

Classes
-------
MarkovCV

"""

from tsvalidation.validation_methods._base import base_splitter
import numpy as np
import pandas as pd
from typing import Generator
import matplotlib.pyplot as plt


class MarkovCV(base_splitter):
    def __init__(self, ts: np.ndarray | pd.Series, fs: float | int, p: int) -> None:

        if self._p % 3 == 0:
            self._m = int(2 * self._p / 3) + 1
        else:
            self._m = int(2 * self._p / 3) + 2

        splits = 2 * self._m
        super().__init__(splits, ts, fs)
        self._p = p

    def _markov_iteration(self, n):
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

        Suo = {}
        Sue = {}
        for u in range(1, 2 * self._m + 1):
            Suo[u] = Su[u][Su[u] % 2 != 0]
            Sue[u] = Su[u][Su[u] % 2 == 0]

        return Suo, Sue

    def split(self) -> Generator[tuple, None, None]:
        Suo, Sue = self._markov_partitions()

        for i in range(1, len(Suo.items()) + 1):
            train, validation = Suo[i], Sue[i]
            yield (train, validation)
            train, validation = Sue[i], Suo[i]
            yield (train, validation)

    def info(self) -> None:
        pass

    def statistics(self) -> tuple[pd.DataFrame]:
        return

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

        fig, axs = plt.subplots(self.n_splits - 1, 1, sharex=True)
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

        plt.show()

        return


if __name__ == "__main":
    mcv = MarkovCV(ts=np.arange(0, 100), fs=0.2, p=3)
    mcv.split()

    print()
