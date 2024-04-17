"""
This module contains all the Out-of-Sample (OOS) validation methods supported by this package.

Classes
-------
Holdout
    Implements the classic Holdout method.

Repeated_Holdout
    Implements the Repeated Holdout approach.
"""

from ._base import base_splitter
from ..data_characteristics import get_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator

class Holdout(base_splitter):

    def __init__(self, ts: np.ndarray | pd.Series, fs: float | int, validation_size: float=0.3) -> None:
        
        """
        Class constructor.

        This is the constructor of the Holdout class.

        Parameters
        ----------
        ts : np.ndarray | pd.Series
            Univariate time series.

        validation_size : float, optional
            Validation set size (relative to the time series size), by default 0.3.

        Raises
        ------
        TypeError
            If the validation size is not a float.
        
        ValueError
            If the validation size does not lie in the ]0, 1[ interval.
        """

        super().__init__(2, ts, fs);
        self._check_validation_size(validation_size);
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
        Provide some basic information on the training and validation sets.

        This method ... .
        """

        print(f"Holdout method\
                --------------\
                Time series size: {self._n_samples}\
                Training size: {np.round(1 - self._val_size, 2)} ({(1 - self._val_size) * self._n_samples} samples)\
                Validation size: {np.round(self._val_size, 2)} ({self._val_size * self._n_samples} samples)");

        return;

    def statistics(self) -> tuple[pd.DataFrame]:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[pd.DataFrame]
            Relevant features for the entire time series, the training set, and the validation set.
        """

        split = self.split();
        training, validation = next(split);

        full_feat = get_features(self._series, self.sampling_freq);
        training_feat = get_features(self._series[training], self.sampling_freq);
        validation_feat = get_features(self._series[validation], self.sampling_freq);

        return (full_feat, training_feat, validation_feat);

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
        
        split = self.split();
        training, validation = next(split);

        fig = plt.figure(figsize=(height, width));
        ax = fig.add_subplot(1, 1, 1);
        ax.plot(training, self._series[training], label="Training set");
        ax.plot(validation, self._series[validation], label="Validation set");
        ax.set_xlabel("Samples");
        ax.set_ylabel("Time Series");
        ax.set_title("Holdout method");
        ax.legend();
        plt.show();

        return;

class Repeated_Holdout(base_splitter):

    def __init__(self, ts: np.ndarray | pd.Series, fs: float | int, iterations: int, splitting_interval: list=[0.7, 0.8]) -> None:

        super().__init__(2, ts, fs);
        self._check_iterations(iterations);
        self._check_splits(splitting_interval);
        self._iter = iterations;
        self._interval = self._convert_interval(splitting_interval);

        return;

    def _check_iterations(self, iterations: int) -> None:

        """
        Perform type and value checks on the number of iterations.
        """

        if(isinstance(iterations, int) is False):

            raise TypeError("The number of iterations must be an integer.");

        if(iterations <= 0):

            raise ValueError("The number of iterations must be positive.");

    def _check_splits(self, splitting_interval: list) -> None:

        if(isinstance(splitting_interval, list) is False):

            raise TypeError("The splitting interval must be a list.");

        if(len(splitting_interval) > 2):

            raise ValueError("The splitting interval should be composed of two elements.");

        for element in splitting_interval:
            
            if((isinstance(element, int) or isinstance(element, float)) is False):
                
                raise TypeError("The interval must be entirely composed of integers or floats.");
    
        if(splitting_interval[0] > splitting_interval[1]):

            raise ValueError("'splitting_interval' should have the [smaller_value, larger_value] format.");

        return;

    def _convert_interval(self, splitting_interval: list) -> list:
        
        """
        Convert intervals of floats (percentages) into intervals of integers (indices).
        """

        for ind, element in enumerate(splitting_interval):

            if(isinstance(element, float) is True):

                splitting_interval[ind] = int(np.round(element * self._n_samples));
            
        return splitting_interval;

    def split(self) -> np.Generator[np.ndarray, np.ndarray]:

        pass

    def info(self) -> None:

        pass

    def statistics(self) -> tuple[pd.DataFrame]:
        
        pass

    def plot(self, height: int, width: int) -> None:
        
        pass