"""
This module contains all the Markov cross-validation method.

Classes
-------
MarkovCV

"""

from ._base import base_splitter
import numpy as np
import pandas as pd
from typing import Generator


class MarkovCV(base_splitter):
    def __init__(
        self,
        splits: int,
        ts: np.ndarray | pd.Series,
        fs: float | int,
    ) -> None:

        super().__init__(splits, ts, fs)

    def split(self) -> Generator[tuple, None, None]:
        yield

    def info(self) -> None:
        pass

    def statistics(self) -> tuple[pd.DataFrame]:
        return

    def plot(self, height: int, width: int) -> None:
        pass
