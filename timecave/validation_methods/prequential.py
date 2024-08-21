#   Copyright 2024 Beatriz Lourenço, Miguel Loureiro, IS4
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
[Out-of-Sample methods](../OOS/index.md): Out-of-sample methods for time series data.

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
    Implements every variant of the Growing Window method.

    This class implements the Growing Window method. It also supports every variant of this method, including Gap Growing Window and 
    Weighted Growing Window. The `gap` parameter can be used to implement the former, while the `weight_function` argument allows the user 
    to implement the latter in a convenient way.

    Parameters
    ----------
    splits : int
        The number of folds used to partition the data.

    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    gap : int, default=0
        Number of folds separating the validation set from the training set. 
        If this value is set to zero, the validation set will be adjacent to the training set.

    weight_function : callable, default=constant_weights
        Fold weighting function. Check the [weights](../weights/index.md) module for more details.

    params : dict, optional
        Parameters to be passed to the weighting functions.

    Attributes
    ----------
    n_splits : int
        The number of splits.

    sampling_freq : int | float
        The series' sampling frequency (Hz).

    Methods
    -------
    split()
        Split the time series into training and validation sets.

    info()            
        Provide additional information on the validation method.

    statistics() 
        Compute relevant statistics for both training and validation sets.

    plot(height: int, width: int)
        Plot the partitioned time series.

    Raises
    ------
    TypeError
        If `gap` is not an integer.

    ValueError
        If `gap` is a negative number.

    ValueError
        If `gap` surpasses the limit imposed by the number of folds.

    See also
    --------
    [Rolling Window](roll.md): Similar to Growing Window, but the amount of samples in the training set is kept constant.

    Notes
    -----
    The Growing Window method splits the data into $N$ different folds. Then, in every iteration $i$, the model is trained on data
    from the first $i$ folds and validated on the $i+1^{th}$ fold (assuming no gap is specified). The average error on the validation sets 
    is then taken as the estimate of the model's true error. This method preserves the temporal 
    order of observations, as the training set always precedes the validation set. If a gap is specified, the procedure runs for $N-1-N_{gap}$ 
    iterations, where $N_{gap}$ is the number of folds separating the training and validation sets.

    ![grow](../../../images/GrowWindow.png)

    Note that the amount of data used to train the model varies significantly from fold to fold. Therefore, it seems natural to assume that the models trained \
    on more data will better mimic the situation where the model is trained using all the available data, thus yielding a more accurate estimate of the model's true error. 
    To address this issue, one may use a weighted average to compute the final estimate of the error, with larger weights being assigned to the estimates obtained \
    using models trained on larger amounts of data.
    For more details on this method, the reader should refer to [[1]](#1).
    
    References
    ----------
    ##1
    Vitor Cerqueira, Luis Torgo, and Igor Mozetiˇc. Evaluating time series forecasting models: An empirical study on performance estimation methods.
    Machine Learning, 109(11):1997–2028, 2020.
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

    def split(self) -> Generator[tuple[np.ndarray, np.ndarray, float], None, None]:
        """
        Split the time series into training and validation sets.

        This method splits the series' indices into disjoint sets containing the training and validation indices.
        At every iteration, an array of training indices and another one containing the validation indices are generated.
        Note that this method is a generator. To access the indices, use the `next()` method or a `for` loop.

        Yields
        ------
        np.ndarray
            Array of training indices.

        np.ndarray
            Array of validation indices.

        float
            Weight assigned to the error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import GrowingWindow
        >>> ts = np.ones(10);
        >>> splitter = GrowingWindow(5, ts); # Split the data into 5 different folds
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1]
        Validation set indices: [2 3]
        Iteration 2
        Training set indices: [0 1 2 3]
        Validation set indices: [4 5]
        Iteration 3
        Training set indices: [0 1 2 3 4 5]
        Validation set indices: [6 7]
        Iteration 4
        Training set indices: [0 1 2 3 4 5 6 7]
        Validation set indices: [8 9]
        
        If the number of samples is not divisible by the number of folds, the first folds will contain more samples:

        >>> ts2 = np.ones(17);
        >>> splitter = GrowingWindow(5, ts2);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1 2 3]
        Validation set indices: [4 5 6 7]
        Iteration 2
        Training set indices: [0 1 2 3 4 5 6 7]
        Validation set indices: [ 8  9 10]
        Iteration 3
        Training set indices: [ 0  1  2  3  4  5  6  7  8  9 10]
        Validation set indices: [11 12 13]
        Iteration 4
        Training set indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13]
        Validation set indices: [14 15 16]

        If a gap is specified (Gap Growing Window), the validation set will no longer be adjacent to the training set.
        Keep in mind that, the larger the gap between these two sets, the fewer iterations are run:

        >>> splitter = GrowingWindow(5, ts, gap=1);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1]
        Validation set indices: [4 5]
        Iteration 2
        Training set indices: [0 1 2 3]
        Validation set indices: [6 7]
        Iteration 3
        Training set indices: [0 1 2 3 4 5]
        Validation set indices: [8 9]

        Weights can be assigned to the error estimates (Weighted Growing Window method). 
        The parameters for the weighting functions must be passed to the class constructor:

        >>> from timecave.validation_methods.weights import exponential_weights
        >>> splitter = GrowingWindow(5, ts, weight_function=exponential_weights, params={"base": 2});
        >>> for ind, (train, val, weight) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        ...     print(f"Weight: {np.round(weight, 3)}");
        Iteration 1
        Training set indices: [0 1]
        Validation set indices: [2 3]
        Weight: 0.067
        Iteration 2
        Training set indices: [0 1 2 3]
        Validation set indices: [4 5]
        Weight: 0.133
        Iteration 3
        Training set indices: [0 1 2 3 4 5]
        Validation set indices: [6 7]
        Weight: 0.267
        Iteration 4
        Training set indices: [0 1 2 3 4 5 6 7]
        Validation set indices: [8 9]
        Weight: 0.533
        """

        for i, (ind, weight) in enumerate(zip(self._splitting_ind[:-1], self._weights)):

            gap_ind = self._splitting_ind[i + self._gap]
            gap_end_ind = self._splitting_ind[i + self._gap + 1]

            train = self._indices[:ind]
            validation = self._indices[gap_ind:gap_end_ind]

            yield (train, validation, weight)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the number of splits, the fold size, the maximum and minimum training set sizes, the gap, 
        and the weights that will be used to compute the error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import GrowingWindow
        >>> ts = np.ones(10);
        >>> splitter = GrowingWindow(5, ts);
        >>> splitter.info();
        Growing Window method
        ---------------------
        Time series size: 10 samples
        Number of splits: 5
        Fold size: 2 to 2 samples (20.0 to 20.0 %)
        Minimum training set size: 2 samples (20.0 %)
        Maximum training set size: 8 samples (80.0 %)
        Gap: 0
        Weights: [1. 1. 1. 1.]
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
        Compute relevant statistics for both training and validation sets.

        This method computes relevant time series features, such as mean, strength-of-trend, etc. for both the whole time series, the training set and the validation set.
        It can and should be used to ensure that the characteristics of both the training and validation sets are, statistically speaking, similar to those of the time series one wishes to forecast.
        If this is not the case, using the validation method will most likely lead to a poor assessment of the model's performance.

        Returns
        -------
        pd.DataFrame
            Relevant features for the entire time series.

        pd.DataFrame
            Relevant features for the training set.

        pd.DataFrame
            Relevant features for the validation set.

        Raises
        ------
        ValueError
            If the time series is composed of less than three samples.
        
        ValueError
            If the folds comprise less than two samples.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import GrowingWindow
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = GrowingWindow(5, ts);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
               Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0  1.000000     1.0  1.0  1.0  0.000000            0.0 -7.850462e-17           0.000000               0.0          0.000000                inf            0.000000              0.000000
        0  1.000000     1.0  1.0  1.0  0.000000            0.0 -8.214890e-17           0.000000               0.0          0.000000                inf            0.000000              0.000000
        0  0.833333     1.0  0.0  1.0  0.138889            1.0 -1.428571e-01           0.125000               0.5          0.792481           0.931695            0.200000              0.200000
        0  0.625000     1.0  0.0  1.0  0.234375            1.0 -1.785714e-01           0.122818               0.5          0.600876           1.383496            0.142857              0.142857
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   1.0     1.0  1.0  1.0      0.00            0.0 -7.850462e-17               0.00               0.0               0.0                inf                 0.0                   0.0
        0   0.5     0.5  0.0  1.0      0.25            1.0 -1.000000e+00               0.25               0.5               0.0                inf                 1.0                   1.0
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00               0.00               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00               0.00               0.0               0.0                inf                 0.0                   0.0
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
        Plot the partitioned time series.

        This method allows the user to plot the partitioned time series. The training and validation sets are plotted using different colours.

        Parameters
        ----------
        height : int
            The figure's height.

        width : int
            The figure's width.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import GrowingWindow
        >>> ts = np.ones(100);
        >>> splitter = GrowingWindow(5, ts);
        >>> splitter.plot(10, 10);

        ![grow_plot](../../../images/Grow_plot.png)
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
    Implements every variant of the Rolling Window method.

    This class implements the Rolling Window method. It also supports every variant of this method, including Gap Rolling Window and 
    Weighted Rolling Window. The `gap` parameter can be used to implement the former, while the `weight_function` argument allows the user 
    to implement the latter in a convenient way.

    Parameters
    ----------
    splits : int
        The number of folds used to partition the data.

    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    gap : int, default=0
        Number of folds separating the validation set from the training set. 
        If this value is set to zero, the validation set will be adjacent to the training set.

    weight_function : callable, default=constant_weights
        Fold weighting function. Check the [weights](../weights/index.md) module for more details.

    params : dict, optional
        Parameters to be passed to the weighting functions.

    Attributes
    ----------
    n_splits : int
        The number of splits.

    sampling_freq : int | float
        The series' sampling frequency (Hz).

    Methods
    -------
    split()
        Split the time series into training and validation sets.

    info()            
        Provide additional information on the validation method.

    statistics() 
        Compute relevant statistics for both training and validation sets.

    plot(height: int, width: int)
        Plot the partitioned time series.

    Raises
    ------
    TypeError
        If `gap` is not an integer.

    ValueError
        If `gap` is a negative number.

    ValueError
        If `gap` surpasses the limit imposed by the number of folds.

    See also
    --------
    [Growing Window](roll.md): Similar to Rolling Window, but the training set size gradually increases.

    Notes
    -----
    The Rolling Window method splits the data into $N$ different folds. Then, in every iteration $i$, the model is trained on data
    from the $i^{th}$ fold and validated on the $i+1^{th}$ fold (assuming no gap is specified). The average error on the validation sets 
    is then taken as the estimate of the model's true error. This method preserves the temporal 
    order of observations, as the training set always precedes the validation set. If a gap is specified, the procedure runs for $N-1-N_{gap}$ 
    iterations, where $N_{gap}$ is the number of folds separating the training and validation sets.

    ![roll](../../../images/RollWindow.png)

    Note that, even though the size of the training set is kept constant throughout the validation procedure, the models from the last iterations are trained on more 
    recent data. It is therefore reasonable to assume that these models will have an advantage over the ones trained on older data, yielding a less biased estimate of the 
    model's true error. 
    To address this issue, one may use a weighted average to compute the final estimate of the error, with larger weights being assigned to the estimates obtained \
    using models trained on more recent data.
    For more details on this method, the reader should refer to [[1]](#1).
    
    References
    ----------
    ##1
    Vitor Cerqueira, Luis Torgo, and Igor Mozetiˇc. Evaluating time series forecasting models: An empirical study on performance estimation methods.
    Machine Learning, 109(11):1997–2028, 2020.
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

    def split(self) -> Generator[tuple[np.ndarray, np.ndarray, float], None, None]:
        """
        Split the time series into training and validation sets.

        This method splits the series' indices into disjoint sets containing the training and validation indices.
        At every iteration, an array of training indices and another one containing the validation indices are generated.
        Note that this method is a generator. To access the indices, use the `next()` method or a `for` loop.

        Yields
        ------
        np.ndarray
            Array of training indices.

        np.ndarray
            Array of validation indices.

        float
            Weight assigned to the error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import RollingWindow
        >>> ts = np.ones(10);
        >>> splitter = RollingWindow(5, ts); # Split the data into 5 different folds
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1]
        Validation set indices: [2 3]
        Iteration 2
        Training set indices: [2 3]
        Validation set indices: [4 5]
        Iteration 3
        Training set indices: [4 5]
        Validation set indices: [6 7]
        Iteration 4
        Training set indices: [6 7]
        Validation set indices: [8 9]
        
        If the number of samples is not divisible by the number of folds, the first folds will contain more samples:

        >>> ts2 = np.ones(17);
        >>> splitter = RollingWindow(5, ts2);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1 2 3]
        Validation set indices: [4 5 6 7]
        Iteration 2
        Training set indices: [4 5 6 7]
        Validation set indices: [ 8  9 10]
        Iteration 3
        Training set indices: [ 8  9 10]
        Validation set indices: [11 12 13]
        Iteration 4
        Training set indices: [11 12 13]
        Validation set indices: [14 15 16]

        If a gap is specified (Gap Rolling Window), the validation set will no longer be adjacent to the training set.
        Keep in mind that, the larger the gap between these two sets, the fewer iterations are run:

        >>> splitter = RollingWindow(5, ts, gap=1);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1]
        Validation set indices: [4 5]
        Iteration 2
        Training set indices: [2 3]
        Validation set indices: [6 7]
        Iteration 3
        Training set indices: [4 5]
        Validation set indices: [8 9]

        Weights can be assigned to the error estimates (Weighted Rolling Window method). 
        The parameters for the weighting functions must be passed to the class constructor:

        >>> from timecave.validation_methods.weights import exponential_weights
        >>> splitter = RollingWindow(5, ts, weight_function=exponential_weights, params={"base": 2});
        >>> for ind, (train, val, weight) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        ...     print(f"Weight: {np.round(weight, 3)}");
        Iteration 1
        Training set indices: [0 1]
        Validation set indices: [2 3]
        Weight: 0.067
        Iteration 2
        Training set indices: [2 3]
        Validation set indices: [4 5]
        Weight: 0.133
        Iteration 3
        Training set indices: [4 5]
        Validation set indices: [6 7]
        Weight: 0.267
        Iteration 4
        Training set indices: [6 7]
        Validation set indices: [8 9]
        Weight: 0.533
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
        Provide some basic information on the training and validation sets.

        This method displays the number of splits, the fold size, the gap, 
        and the weights that will be used to compute the error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import RollingWindow
        >>> ts = np.ones(10);
        >>> splitter = RollingWindow(5, ts);
        >>> splitter.info();
        Rolling Window method
        ---------------------
        Time series size: 10 samples
        Number of splits: 5
        Fold size: 2 to 2 samples (20.0 to 20.0 %)
        Gap: 0
        Weights: [1. 1. 1. 1.]
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
        Compute relevant statistics for both training and validation sets.

        This method computes relevant time series features, such as mean, strength-of-trend, etc. for both the whole time series, the training set and the validation set.
        It can and should be used to ensure that the characteristics of both the training and validation sets are, statistically speaking, similar to those of the time series one wishes to forecast.
        If this is not the case, using the validation method will most likely lead to a poor assessment of the model's performance.

        Returns
        -------
        pd.DataFrame
            Relevant features for the entire time series.

        pd.DataFrame
            Relevant features for the training set.

        pd.DataFrame
            Relevant features for the validation set.

        Raises
        ------
        ValueError
            If the time series is composed of less than three samples.
        
        ValueError
            If the folds comprise less than two samples.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import RollingWindow
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = RollingWindow(5, ts);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   1.0     1.0  1.0  1.0      0.00            0.0 -7.850462e-17               0.00               0.0               0.0                inf                 0.0                   0.0
        0   1.0     1.0  1.0  1.0      0.00            0.0 -7.850462e-17               0.00               0.0               0.0                inf                 0.0                   0.0
        0   0.5     0.5  0.0  1.0      0.25            1.0 -1.000000e+00               0.25               0.5               0.0                inf                 1.0                   1.0
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00               0.00               0.0               0.0                inf                 0.0                   0.0
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   1.0     1.0  1.0  1.0      0.00            0.0 -7.850462e-17               0.00               0.0               0.0                inf                 0.0                   0.0
        0   0.5     0.5  0.0  1.0      0.25            1.0 -1.000000e+00               0.25               0.5               0.0                inf                 1.0                   1.0
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00               0.00               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00               0.00               0.0               0.0                inf                 0.0                   0.0
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
        Plot the partitioned time series.

        This method allows the user to plot the partitioned time series. The training and validation sets are plotted using different colours.

        Parameters
        ----------
        height : int
            The figure's height.

        width : int
            The figure's width.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.prequential import RollingWindow
        >>> ts = np.ones(100);
        >>> splitter = RollingWindow(5, ts);
        >>> splitter.plot(10, 10);

        ![roll_plot](../../../images/Roll_plot.png)
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

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);