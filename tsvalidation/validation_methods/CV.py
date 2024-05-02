"""

"""

from ._base import base_splitter
from .weights import constant_weights
from ..data_characteristics import get_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator


class CV(base_splitter):

    def __init__(
        self,
        splits: int,
        ts: np.ndarray | pd.Series,
        fs: float | int,
        h: int = 0,
        weight_function: callable = constant_weights,
    ) -> None:
        super().__init__(splits, ts, fs)

    def split(self) -> Generator[tuple, None, None]:

        pass

    def info(self) -> None:

        pass

    def statistics(self) -> tuple[pd.DataFrame]:

        pass

    def plot(self, height: int, width: int) -> None:

        pass
