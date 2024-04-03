from ._base import base_splitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This module contains all the Out-of-Sample (OOS) validation methods supported by this package.

Classes
-------
Holdout
    Implements the classic Holdout method.

Repeated_Holdout
    Implements the Repeated Holdout approach.
"""

class Holdout(base_splitter):

    def __init__(self, ts: np.ndarray | pd.Series, validation_size: float=0.3) -> None:
        
        """
        Class constructor.

        This is the constructor of the Holdout class.

        Parameters
        ----------
        ts : np.ndarray | pd.Series
            Univariate time series.

        validation_size : float, optional
            Validation set size (relative to the time series size), by default 0.3.
        """

        super().__init__(2, ts);
        self._val_size = validation_size;

        return;

    def _check_validation_size(self, validation_size: float) -> None:
        
        """
        Perform type and value checks on the 'validation_size' parameter.

        This method checks whether the 'validation_size' parameter is a float and whether it
        lies in the interval of 0 to 1.

        Parameters
        ----------
        validation_size : float
            Validation set size (relative to the time series size).

        Raises
        ------
        TypeError
            If the validation size is not a float.
        
        ValueError
            If the validation size does not lie in the ]0, 1[ interval. 
        """

        if(isinstance(validation_size, float) is False):

            raise TypeError("'validation_size' must be a float.");

        elif(validation_size >= 1 or validation_size <= 0):

            raise ValueError("'validation_size' must be greater than zero and less than one.");

        return;

    def split(self) -> np.Generator[np.ndarray, np.ndarray]:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        np.Generator[np.ndarray, np.ndarray]
            _description_

        Yields
        ------
        Iterator[np.Generator[np.ndarray, np.ndarray]]
            _description_
        """

        split_ind = int(np.round((1 - self._val_size) * self._n_samples));

        train = self._indices[:split_ind];
        validation = self._indices[split_ind:];

        yield train, validation;

    def info(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        print(f"Holdout method\
                --------------\
                Time series size: {self._n_samples}\
                Training size: {1 - self._val_size} ({(1 - self._val_size) * self._n_samples} samples)\
                Validation size: {self._val_size} ({self._val_size * self._n_samples} samples)");

        return;

    def statistics(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        split = self.split();
        training, validation = next(split);

        return;

    def plot(self, height: int, width: int) -> None:
        
        """
        _summary_

        _extended_summary_
        """
        
        split = self.split();
        training, validation = next(split);

        fig = plt.figure(size=(height, width));

        return;