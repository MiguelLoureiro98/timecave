"""
This module contains all the Out-of-Sample (OOS) validation methods supported by this package.

Classes
-------
Holdout
    Implements the classic Holdout method.

Repeated_Holdout
    Implements the Repeated Holdout approach.

Rolling_Origin_Update
    Implements the Rolling Origin Update method.

Rolling_Origin_Recalibration
    Implements the Rolling Origin Recalibration method.

Fixed_Size_Rolling_Window
    Implements the Fixed-size Rolling Window method.

TODO: Add Tashman reference to the last 3/4 methods.
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

        fs : float | int
            Sampling frequency (Hz).

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

    def split(self) -> Generator[tuple, None, None]:
        
        """
        _summary_

        _extended_summary_

        Yields
        ------
        Generator[tuple, None, None]
            _description_
        """

        split_ind = int(np.round((1 - self._val_size) * self._n_samples));

        train = self._indices[:split_ind];
        validation = self._indices[split_ind:];

        yield (train, validation);

    def info(self) -> None:
        
        """
        Provide some basic information on the training and validation sets.

        This method ... .
        """

        print("Holdout method");
        print("--------------");
        print(f"Time series size: {self._n_samples}");
        print(f"Training size: {np.round(1 - self._val_size, 2)} ({(1 - self._val_size) * self._n_samples} samples)");
        print(f"Validation size: {np.round(self._val_size, 2)} ({self._val_size * self._n_samples} samples)");

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

    def __init__(self, ts: np.ndarray | pd.Series, fs: float | int, iterations: int, splitting_interval: list[int | float]=[0.7, 0.8], seed: int=0) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        ts : np.ndarray | pd.Series
            _description_

        fs : float | int
            _description_

        iterations : int
            _description_

        splitting_interval : list[int | float], optional
            _description_, by default [0.7, 0.8]

        seed : int, optional
            _description_, by default 0
        """

        self._check_iterations(iterations);
        self._check_splits(splitting_interval);
        super().__init__(iterations, ts, fs);
        self._iter = iterations;
        self._interval = self._convert_interval(splitting_interval);
        self._seed = seed;
        self._splitting_ind = self._get_splitting_ind();

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

        """
        Perform several type and value checks on the splitting interval.
        """

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

    def _check_seed(self, seed: int) -> None:

        """
        Perform a type check on the seed.
        """

        if(isinstance(seed, int) is False):

            raise TypeError("'seed' should be an integer.");

        return;

    def _get_splitting_ind(self) -> np.ndarray:

        """
        Generate the splitting indices.
        """

        np.random.seed(self._seed);
        rand_ind = np.random.randint(low=self._interval[0], high=self._interval[1], size=self._iter);

        return rand_ind;

    def split(self) -> Generator[tuple, None, None]:
        
        """
        _summary_

        _extended_summary_

        Yields
        ------
        Generator[tuple, None, None]
            _description_
        """

        for ind in self._splitting_ind:

            training = self._indices[:ind];
            validation = self._indices[ind:];

            yield (training, validation);

    def info(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        mean_size = self._n_samples - self._splitting_ind.mean();
        max_size = self._n_samples - self._splitting_ind.max();
        min_size = self._n_samples - self._splitting_ind.min();

        mean_pct = np.round(mean_size / self._n_samples, 2) * 100;
        max_pct = np.round(max_size / self._n_samples, 2) * 100;
        min_pct = np.round(min_size / self._n_samples, 2) * 100;

        print("Repeated Holdout method");
        print("-----------------------");
        print(f"Time series size: {self._n_samples} samples");
        print(f"Average validation set size: {mean_size} samples ({mean_pct} %)");
        print(f"Maximum validation set size: {max_size} samples ({max_pct} %)");
        print(f"Minimum validation set size: {min_size} samples ({min_pct} %)");

    def statistics(self) -> tuple[pd.DataFrame]:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[pd.DataFrame]
            _description_
        """

        full_features = get_features(self._series, self.sampling_freq);

        training_stats = [];
        validation_stats = [];

        #for ind in self._splitting_ind:

        #    training_feat = get_features(self._series[:ind], self.sampling_freq);
        #    validation_feat = get_features(self._series[ind:], self.sampling_freq);
        #    training_stats.append(training_feat);
        #    validation_stats.append(validation_feat);

        for (training, validation) in self.split():

            training_feat = get_features(self._series[training], self.sampling_freq);
            validation_feat = get_features(self._series[validation], self.sampling_freq);
            training_stats.append(training_feat);
            validation_stats.append(validation_feat);
        
        training_features = pd.concat(training_stats);
        validation_features = pd.concat(validation_stats);

        return (full_features, training_features, validation_features);

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
        
        fig, axs = plt.subplots(self._iter, 1, sharex=True);
        fig.set_figheight(height);
        fig.set_figwidth(width);
        fig.supxlabel("Samples");
        fig.supylabel("Time Series");
        fig.suptitle("Repeated Holdout method");

        for it, (training, validation) in enumerate(self.split()):

            axs[it, 0].plot(training, self._series[training], label="Training set");
            axs[it, 0].plot(validation, self._series[validation], label="Validation set");
            axs[it, 0].set_title("Iteration {}".format(it+1));
            axs[it, 0].legend();
        
        plt.show();

        return;

class Rolling_Origin_Update(base_splitter):

    def __init__(self, ts: np.ndarray | pd.Series, fs: float | int, origin: int | float=0.7) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        ts : np.ndarray | pd.Series
            _description_

        fs : float | int
            _description_

        origin : int | float, optional
            _description_, by default 0.7
        """

        super().__init__(2, ts, fs);
        self._check_origin(origin);
        self._origin = self._convert_origin(origin);
        self._splitting_ind = np.arange(self._origin + 1, self._n_samples);
        self._n_splits = self._splitting_ind.shape[0];

        return;

    def _check_origin(self, origin: int | float) -> None:

        """
        Perform type and value checks on the origin.
        """

        is_int = isinstance(origin, int);
        is_float = isinstance(origin, float);

        if((is_int or is_float) is False):

            raise TypeError("'origin' must be an integer or a float.");
    
        if(is_float and (origin >= 1 or origin <= 0)):

            raise ValueError("If 'origin' is a float, it must lie in the interval of ]0, 1[.");
    
        if(is_int and (origin >= self._n_samples or origin <= 0)):

            raise ValueError("If 'origin' is an integer, it must lie in the interval of ]0, n_samples[.");

        return;

    def _convert_origin(self, origin: int | float) -> int:

        """
        Cast the origin from float (proportion) to integer (index).
        """

        if(isinstance(origin, float) is True):

            origin = int(np.round(origin * self._n_samples)) - 1;

        return origin;

    def split(self) -> Generator[np.ndarray, None, None]:
        
        """
        _summary_

        _extended_summary_

        Yields
        ------
        Generator[tuple, None, None]
            _description_
        """

        for ind in self._splitting_ind:

            training = self._indices[:self._origin + 1];
            validation = self._indices[ind:];
            
            yield (training, validation);

    def info(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        training_size = self._origin;
        max_size = self._n_samples - self._origin;
        min_size = 1;

        training_pct = np.round(training_size / self._n_samples, 2) * 100;
        max_pct = np.round(max_size / self._n_samples, 2) * 100;
        min_pct = np.round(1 / self._n_samples, 2) * 100;

        print("Rolling Origin Update method");
        print("----------------------------");
        print(f"Time series size: {self._n_samples} samples");
        print(f"Training set size (fixed parameter): {training_size} samples ({training_pct} %)");
        print(f"Maximum validation set size: {max_size} samples ({max_pct} %)");
        print(f"Minimum validation set size: {min_size} sample ({min_pct} %)");

        return;

    def statistics(self) -> tuple[pd.DataFrame]:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[pd.DataFrame]
            _description_
        """

        full_features = get_features(self._series, self.sampling_freq);
        it_1 = True;
        validation_stats = [];

        for (training, validation) in self.split():

            if(it_1 is True):

                training_features = get_features(self._series[training], self.sampling_freq);
                it_1 = False;
            
            validation_feat = get_features(self._series[validation], self.sampling_freq);
            validation_stats.append(validation_feat);
        
        validation_features = pd.concat(validation_stats);

        return (full_features, training_features, validation_features);

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

        fig, axs = plt.subplots(self._n_samples - self._origin, 1, sharex=True);
        fig.set_figheight(height);
        fig.set_figwidth(width);
        fig.supxlabel("Samples");
        fig.supylabel("Time Series");
        fig.suptitle("Rolling Origin Update method");

        for it, (training, validation) in enumerate(self.split()):

            axs[it, 0].plot(training, self._series[training], label="Training set");
            axs[it, 0].plot(validation, self._series[validation], label="Validation set");
            axs[it, 0].set_title("Iteration {}".format(it+1));
            axs[it, 0].legend();
        
        plt.show();

        return;

class Rolling_Origin_Recalibration(base_splitter):

    def __init__(self, ts: np.ndarray | pd.Series, fs: float | int, origin: int | float=0.7) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        ts : np.ndarray | pd.Series
            _description_

        fs : float | int
            _description_

        origin : int | float, optional
            _description_, by default 0.7
        """

        super().__init__(2, ts, fs);
        self._check_origin(origin);
        self._origin = self._convert_origin(origin);
        self._splitting_ind = np.arange(self._origin + 1, self._n_samples);
        self._n_splits = self._splitting_ind.shape[0];

        return;

    def _check_origin(self, origin: int | float) -> None:

        """
        Perform type and value checks on the origin.
        """

        is_int = isinstance(origin, int);
        is_float = isinstance(origin, float);

        if((is_int or is_float) is False):

            raise TypeError("'origin' must be an integer or a float.");
    
        if(is_float and (origin >= 1 or origin <= 0)):

            raise ValueError("If 'origin' is a float, it must lie in the interval of ]0, 1[.");
    
        if(is_int and (origin >= self._n_samples or origin <= 0)):

            raise ValueError("If 'origin' is an integer, it must lie in the interval of ]0, n_samples[.");

        return;

    def _convert_origin(self, origin: int | float) -> int:

        """
        Cast the origin from float (proportion) to integer (index).
        """

        if(isinstance(origin, float) is True):

            origin = int(np.round(origin * self._n_samples)) - 1;

        return origin;

    def split(self) -> Generator[tuple, None, None]:
        
        """
        _summary_

        _extended_summary_

        Yields
        ------
        Generator[tuple, None, None]
            _description_
        """

        for ind in self._splitting_ind:

            training = self._indices[:ind];
            validation = self._indices[ind:];
        
            yield (training, validation);

    def info(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        max_training_size = self._n_samples - 1;
        min_training_size = self._origin;
        max_validation_size = self._n_samples - self._origin;
        min_validation_size = 1;

        max_training_pct = np.round(max_training_size / self._n_samples, 2) * 100;
        min_training_pct = np.round(min_training_size / self._n_samples, 2) * 100;
        max_validation_pct = np.round(max_validation_size / self._n_samples, 2) * 100;
        min_validation_pct = np.round(min_validation_size / self._n_samples, 2) * 100;

        print("Rolling Origin Recalibration method");
        print("-----------------------------------");
        print(f"Time series size: {self._n_samples} samples");
        print(f"Minimum training set size: {min_training_size} samples ({min_training_pct} %)");
        print(f"Maximum validation set size: {max_validation_size} samples ({max_validation_pct} %)");
        print(f"Maximum training set size: {max_training_size} samples ({max_training_pct} %)");
        print(f"Minimum validation set size: {min_validation_size} samples ({min_validation_pct} %)");

        return;

    def statistics(self) -> tuple[pd.DataFrame]:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[pd.DataFrame]
            _description_
        """
        
        full_features = get_features(self._series, self.sampling_freq);
        training_stats = [];
        validation_stats = [];

        for (training, validation) in self.split():

            training_feat = get_features(self._series[training], self.sampling_freq);
            training_stats.append(training_feat);
            validation_feat = get_features(self._series[validation], self.sampling_freq);
            validation_stats.append(validation_feat);
        
        training_features = pd.concat(training_stats);
        validation_features = pd.concat(validation_stats);

        return (full_features, training_features, validation_features);

    def plot(self, height: int, width: int) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        height : int
            _description_

        width : int
            _description_
        """

        fig, axs = plt.subplots(self._n_samples - self._origin, 1, sharex=True);
        fig.set_figheight(height);
        fig.set_figwidth(width);
        fig.supxlabel("Samples");
        fig.supylabel("Time Series");
        fig.suptitle("Rolling Origin Recalibration method");

        for it, (training, validation) in enumerate(self.split()):

            axs[it, 0].plot(training, self._series[training], label="Training set");
            axs[it, 0].plot(validation, self._series[validation], label="Validation set");
            axs[it, 0].set_title("Iteration {}".format(it+1));
            axs[it, 0].legend();
        
        plt.show();

        return;

class Fixed_Size_Rolling_Window(base_splitter):

    def __init__(self, ts: np.ndarray | pd.Series, fs: float | int, origin: int | float=0.7) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        ts : np.ndarray | pd.Series
            Univariate time series.

        fs : float | int
            Sampling frequency (Hz).

        origin : int | float, optional
            _description_, by default 0.7
        """

        super().__init__(2, ts, fs);
        self._check_origin(origin);
        self._origin = self._convert_origin(origin);
        self._splitting_ind = np.arange(self._origin + 1, self._n_samples);
        self._n_splits = self._splitting_ind.shape[0];

        return;

    def _check_origin(self, origin: int | float) -> None:

        """
        Perform type and value checks on the origin.
        """

        is_int = isinstance(origin, int);
        is_float = isinstance(origin, float);

        if((is_int or is_float) is False):

            raise TypeError("'origin' must be an integer or a float.");
    
        if(is_float and (origin >= 1 or origin <= 0)):

            raise ValueError("If 'origin' is a float, it must lie in the interval of ]0, 1[.");
    
        if(is_int and (origin >= self._n_samples or origin <= 0)):

            raise ValueError("If 'origin' is an integer, it must lie in the interval of ]0, n_samples[.");

        return;

    def _convert_origin(self, origin: int | float) -> int:

        """
        Cast the origin from float (proportion) to integer (index).
        """

        if(isinstance(origin, float) is True):

            origin = int(np.round(origin * self._n_samples)) - 1;

        return origin;

    def split(self) -> Generator[tuple, None, None]:
        
        """
        _summary_

        _extended_summary_

        Yields
        ------
        Generator[tuple, None, None]
            _description_
        """
        start_training_ind = self._splitting_ind - self._origin - 1;
        
        for start_ind, end_ind in zip(start_training_ind, self._splitting_ind):

            training = self._indices[start_ind:end_ind];
            validation = self._indices[end_ind:];

            yield (training, validation);

    def info(self) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        training_size = self._origin;
        max_size = self._n_samples - self._origin;
        min_size = 1;

        training_pct = np.round(training_size / self._n_samples, 2) * 100;
        max_pct = np.round(max_size / self._n_samples, 2) * 100;
        min_pct = np.round(1 / self._n_samples, 2) * 100;

        print("Fixed-size Rolling Window method");
        print("--------------------------------");
        print(f"Time series size: {self._n_samples} samples");
        print(f"Training set size (fixed parameter): {training_size} samples ({training_pct} %)");
        print(f"Maximum validation set size: {max_size} samples ({max_pct} %)");
        print(f"Minimum validation set size: {min_size} sample ({min_pct} %)");

        return;

    def statistics(self) -> tuple[pd.DataFrame]:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        tuple[pd.DataFrame]
            _description_
        """

        full_features = get_features(self._series, self.sampling_freq);
        training_stats = [];
        validation_stats = [];

        for (training, validation) in self.split():

            training_feat = get_features(self._series[training], self.sampling_freq);
            training_stats.append(training_feat);
            validation_feat = get_features(self._series[validation], self.sampling_freq);
            validation_stats.append(validation_feat);
        
        training_features = pd.concat(training_stats);
        validation_features = pd.concat(validation_stats);

        return (full_features, training_features, validation_features);

    def plot(self, height: int, width: int) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        height : int
            _description_

        width : int
            _description_
        """

        fig, axs = plt.subplots(self._n_samples - self._origin, 1, sharex=True);
        fig.set_figheight(height);
        fig.set_figwidth(width);
        fig.supxlabel("Samples");
        fig.supylabel("Time Series");
        fig.suptitle("Fixed-size Rolling Window method");

        for it, (training, validation) in enumerate(self.split()):

            axs[it, 0].plot(training, self._series[training], label="Training set");
            axs[it, 0].plot(validation, self._series[validation], label="Validation set");
            axs[it, 0].set_title("Iteration {}".format(it+1));
            axs[it, 0].legend();
        
        plt.show();

        return;