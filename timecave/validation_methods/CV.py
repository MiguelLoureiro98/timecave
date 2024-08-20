"""
This module contains all the CV ('Cross-Validation') validation methods supported by this package.

Classes
-------
BlockCV
    Implements the Block CV method, along with its weighted version.

hvBlockCV
    Implements the hv Block method.

AdaptedhvBlockCV
    Implements the Adapted hv Block CV method, along with its weighted version.

See also
--------
[Out-of-Sample methods](../OOS/index.md): Out-of-sample methods for time series data.

[Prequential methods](../prequential/index.md): Prequential or forward validation methods for time series data.

[Markov methods](../markov/index.md): Markov cross-validation method for time series data.

Notes
-----
Cross-validation methods are one of the three main classes of validation methods for time series data (the others being \
out-of-sample methods and prequential methods).
Like prequential methods, CV methods partition the series into equally sized folds (with the exception of the hv Block variant).
However, CV methods do not preserve the temporal order of observations, meaning that a model can be trained on later data and tested \
on earlier data. CV methods also differ from Out-of-Sample methods, as the latter do not partition the series in the same way.
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


class BlockCV(BaseSplitter):
    """
    Implements the Block Cross-validation method, as well as its weighted variant.

    This class implements both the Block Cross-validation method and the Weighted Block Cross-validation method. 
    The `weight_function` argument allows the user to implement the latter in a convenient way.

    Parameters
    ----------
    splits : int
        The number of folds used to partition the data.

    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

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

    See also
    --------
    [hv Block CV](hv.md): A blend of Block CV and leave-one-out CV.

    [Adapted hv Block CV](adapted_hv.md): Similar to Block CV, but the training samples that lie closest to the validation set are removed.

    Notes
    -----
    The Block Cross-validation method splits the data into $N$ different folds. Then, in every iteration $i$, the model is validated on data
    from the $i^{th}$ folds and trained on data from the remaining folds. The average error on the validation sets 
    is then taken as the estimate of the model's true error. This method does not preserve the temporal order of the observations.

    ![block](../../../images/BlockCV.png)

    It is reasonable to assume that when the model is validated on more recent data, the error estimate will be more accurate. 
    To address this issue, one may use a weighted average to compute the final estimate of the error, with larger weights being assigned to the estimates obtained \
    using models validated on more recent data.
    For more details on this method, the reader should refer to [[1]](#1) or [[2]](#2).
    
    References
    ----------
    ##1
    Christoph Bergmeir and José M Benítez. On the use of cross-validation for
    time series predictor evaluation. Information Sciences, 191:192–213, 2012.
    
    ##2
    Vitor Cerqueira, Luis Torgo, and Igor Mozetiˇc. Evaluating time series forecasting models: An empirical study on performance estimation methods.
    Machine Learning, 109(11):1997–2028, 2020.
    """

    def __init__(
        self,
        splits: int,
        ts: np.ndarray | pd.Series,
        fs: float | int = 1,
        weight_function: callable = constant_weights,
        params: dict = None,
    ) -> None:

        super().__init__(splits, ts, fs)
        self._splitting_ind = self._split_ind()
        self._weights = weight_function(self.n_splits, params=params)

    def _split_ind(self) -> np.ndarray:
        """
        Compute the splitting indices.
        """

        remainder = int(self._n_samples % self.n_splits)
        split_size = int(np.floor(self._n_samples / self.n_splits))
        split_ind = np.arange(0, self._n_samples, split_size)

        if(remainder != 0):
            
            split_ind[1:remainder+1] += np.array([i for i in range(1, remainder+1)]);
            split_ind[remainder+1:] += remainder;

        else:

            split_ind = np.append(split_ind, self._n_samples);

        if(split_ind.shape[0] > self.n_splits + 1):

            split_ind = split_ind[:self.n_splits+1];

        #split_ind[:remainder] += 1

        #if remainder != 0:

        #    split_ind[remainder:] += remainder

        #split_ind = np.append(split_ind, self._n_samples)

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
        >>> from timecave.validation_methods.CV import BlockCV
        >>> ts = np.ones(10);
        >>> splitter = BlockCV(5, ts); # Split the data into 5 different folds
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [2 3 4 5 6 7 8 9]
        Validation set indices: [0 1]
        Iteration 2
        Training set indices: [0 1 4 5 6 7 8 9]
        Validation set indices: [2 3]
        Iteration 3
        Training set indices: [0 1 2 3 6 7 8 9]
        Validation set indices: [4 5]
        Iteration 4
        Training set indices: [0 1 2 3 4 5 8 9]
        Validation set indices: [6 7]
        Iteration 5
        Training set indices: [0 1 2 3 4 5 6 7]
        Validation set indices: [8 9]

        If the number of samples is not divisible by the number of folds, the first folds will contain more samples:

        >>> ts2 = np.ones(17);
        >>> splitter = BlockCV(5, ts2);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [ 4  5  6  7  8  9 10 11 12 13 14 15 16]
        Validation set indices: [0 1 2 3]
        Iteration 2
        Training set indices: [ 0  1  2  3  8  9 10 11 12 13 14 15 16]
        Validation set indices: [4 5 6 7]
        Iteration 3
        Training set indices: [ 0  1  2  3  4  5  6  7 11 12 13 14 15 16]
        Validation set indices: [ 8  9 10]
        Iteration 4
        Training set indices: [ 0  1  2  3  4  5  6  7  8  9 10 14 15 16]
        Validation set indices: [11 12 13]
        Iteration 5
        Training set indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13]
        Validation set indices: [14 15 16]

        Weights can be assigned to the error estimates (Weighted Rolling Window method). 
        The parameters for the weighting functions must be passed to the class constructor:

        >>> from timecave.validation_methods.weights import exponential_weights
        >>> splitter = BlockCV(5, ts, weight_function=exponential_weights, params={"base": 2});
        >>> for ind, (train, val, weight) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        ...     print(f"Weight: {np.round(weight, 3)}");
        Iteration 1
        Training set indices: [2 3 4 5 6 7 8 9]
        Validation set indices: [0 1]
        Weight: 0.032
        Iteration 2
        Training set indices: [0 1 4 5 6 7 8 9]
        Validation set indices: [2 3]
        Weight: 0.065
        Iteration 3
        Training set indices: [0 1 2 3 6 7 8 9]
        Validation set indices: [4 5]
        Weight: 0.129
        Iteration 4
        Training set indices: [0 1 2 3 4 5 8 9]
        Validation set indices: [6 7]
        Weight: 0.258
        Iteration 5
        Training set indices: [0 1 2 3 4 5 6 7]
        Validation set indices: [8 9]
        Weight: 0.516
        """

        for i, (ind, weight) in enumerate(zip(self._splitting_ind[:-1], self._weights)):

            next_ind = self._splitting_ind[i + 1]

            validation = self._indices[ind:next_ind]
            train = np.array([el for el in self._indices if el not in validation])

            yield (train, validation, weight)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the number of splits, the fold size, 
        and the weights that will be used to compute the error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.CV import BlockCV
        >>> ts = np.ones(10);
        >>> splitter = BlockCV(5, ts);
        >>> splitter.info();
        Block CV method
        ---------------
        Time series size: 10 samples
        Number of splits: 5
        Fold size: 2 to 2 samples (20.0 to 20.0 %)
        Weights: [1. 1. 1. 1. 1.]
        """

        min_fold_size = int(np.floor(self._n_samples / self.n_splits))
        max_fold_size = min_fold_size

        remainder = self._n_samples % self.n_splits

        if remainder != 0:

            max_fold_size += 1

        min_fold_size_pct = np.round(min_fold_size / self._n_samples * 100, 2)
        max_fold_size_pct = np.round(max_fold_size / self._n_samples * 100, 2)

        print("Block CV method")
        print("---------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Number of splits: {self.n_splits}")
        print(
            f"Fold size: {min_fold_size} to {max_fold_size} samples ({min_fold_size_pct} to {max_fold_size_pct} %)"
        )
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
        >>> from timecave.validation_methods.CV import BlockCV
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = BlockCV(5, ts);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
            Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0  0.375     0.0  0.0  1.0  0.234375            1.0    -0.178571           0.154195             0.500          0.600876           1.383496            0.142857              0.142857
        0  0.375     0.0  0.0  1.0  0.234375            1.0    -0.178571           0.154195             0.500          0.600876           1.383496            0.142857              0.142857
        0  0.500     0.5  0.0  1.0  0.250000            1.0    -0.190476           0.095190             0.375          0.600876           1.428869            0.142857              0.142857
        0  0.625     1.0  0.0  1.0  0.234375            1.0    -0.178571           0.122818             0.500          0.600876           1.383496            0.142857              0.142857
        0  0.625     1.0  0.0  1.0  0.234375            1.0    -0.178571           0.122818             0.500          0.600876           1.383496            0.142857              0.142857
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   1.0     1.0  1.0  1.0      0.00            0.0 -7.850462e-17               0.00               0.0               0.0                inf                 0.0                   0.0
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
        >>> from timecave.validation_methods.CV import BlockCV
        >>> ts = np.ones(100);
        >>> splitter = BlockCV(5, ts);
        >>> splitter.plot(10, 10);

        ![block_plot](../../../images/BlockCV_plot.png)
        """

        fig, axs = plt.subplots(self.n_splits, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Block CV method")

        for it, (training, validation, w) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Fold: {} Weight: {}".format(it + 1, np.round(w, 3)))
            axs[it].set_ylim([self._series.min() - 1, self._series.max() + 1])
            axs[it].set_xlim([- 1, self._n_samples + 1])
            axs[it].legend()

        plt.show()

        return


class hvBlockCV(BaseSplitter):
    """
    Implements the hv Block Cross-validation method.

    This class implements the hv Block Cross-validation method. It is similar to the [BlockCV](block.md) class, 
    but it does not support weight generation. Consequently, in order to implement a weighted version of this method, 
    the user must implement their own derived class or compute the weights separately.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    h : int, default=1
        Controls the amount of samples that will be removed from the training set. 
        The `h` samples immediately following and preceding the validation set are not used for training.

    v : int, default=1
        Controls the size of the validation set. $2v + 1$ samples will be used for validation.

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
        If either `h` or `v` are not integers.

    ValueError
        If either `h` or `v` are smaller than or equal to zero.

    ValueError
        If the sum of `h` and `v` is larger than half the amount of samples in the series. 

    Warning
    -------
    Being a variant of the leave-one-out CV procedure, this method is computationally intensive.

    See also
    --------
    [Block CV](block.md): The original Block CV method, which partitions the series into equally sized folds. No training samples are removed.

    Notes
    -----
    The hv Block Cross-validation method is essentially a leave-one-out version of the BlockCV method.
    There are, however, two nuances: the first one is that the $h$ samples immediately following and preceding the validation set 
    are removed from the training set; the second one is that more than one sample can be used for validation. More specifically, the validation set
    comprises $2v + 1$ samples. Note that, if $h = v = 0$, the method boils down to the classic leave-one-out cross-validation procedure.
    The average error on the validation sets is taken as the estimate of the model's true error. This method does not preserve the temporal order of the observations.

    The method was first proposed by Racine [[1]](#1).
    
    References
    ----------
    ##1
    Jeff Racine. Consistent cross-validatory model-selection for dependent data: 
    hv-block cross-validation. Journal of econometrics, 99(1):39–61, 2000
    """

    def __init__(
        self,
        ts: np.ndarray | pd.Series,
        fs: float | int = 1,
        h: int = 0,
        v: int = 0,
    ) -> None:

        super().__init__(ts.shape[0], ts, fs)
        self._check_hv(h, v)
        self._h = h
        self._v = v

        return

    def _check_hv(self, h: int, v: int) -> None:
        """
        Perform type and value checks on both h and v.
        """

        if (isinstance(h, int) and isinstance(v, int)) is False:

            raise TypeError("Both 'h' and 'v' must be integers.")

        if h < 0 or v < 0:

            raise ValueError("Both 'h' and 'v' must be non-negative.")

        if h + v >= np.floor(self._n_samples / 2):

            raise ValueError(
                "The sum of h and v should be less than half the amount of samples that compose the time series."
            )

        return

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
        >>> from timecave.validation_methods.CV import hvBlockCV
        >>> ts = np.ones(10);
        >>> splitter = hvBlockCV(ts, h=2, v=1); # Use 3 samples for validation; remove 2-4 samples from the training set
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [4 5 6 7 8 9]
        Validation set indices: [0 1]
        Iteration 2
        Training set indices: [5 6 7 8 9]
        Validation set indices: [0 1 2]
        Iteration 3
        Training set indices: [6 7 8 9]
        Validation set indices: [1 2 3]
        Iteration 4
        Training set indices: [7 8 9]
        Validation set indices: [2 3 4]
        Iteration 5
        Training set indices: [0 8 9]
        Validation set indices: [3 4 5]
        Iteration 6
        Training set indices: [0 1 9]
        Validation set indices: [4 5 6]
        Iteration 7
        Training set indices: [0 1 2]
        Validation set indices: [5 6 7]
        Iteration 8
        Training set indices: [0 1 2 3]
        Validation set indices: [6 7 8]
        Iteration 9
        Training set indices: [0 1 2 3 4]
        Validation set indices: [7 8 9]
        Iteration 10
        Training set indices: [0 1 2 3 4 5]
        Validation set indices: [8 9]
        """

        for i, _ in enumerate(self._indices):

            validation = self._indices[
                np.fmax(i - self._v, 0) : np.fmin(i + self._v + 1, self._n_samples)
            ]
            h_ind = self._indices[
                np.fmax(i - self._v - self._h, 0) : np.fmin(
                    i + self._v + self._h + 1, self._n_samples
                )
            ]
            train = np.array([el for el in self._indices if el not in h_ind])

            yield (train, validation, 1.0)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the number of splits, the values of the `h` and `v` 
        parameters, and the maximum and minimum sizes of both the training and validation sets.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.CV import hvBlockCV
        >>> ts = np.ones(10);
        >>> splitter = hvBlockCV(ts, h=2, v=2);
        >>> splitter.info();
        hv-Block CV method
        ------------------
        Time series size: 10 samples
        Number of splits: 10
        Minimum training set size: 1 samples (10.0 %)
        Maximum training set size: 5 samples (50.0 %)
        Minimum validation set size: 3 samples (30.0 %)
        Maximum validation set size: 5 samples (50.0 %)
        h: 2
        v: 2
        """

        min_train_size = self._n_samples - 2 * (self._h + self._v + 1) + 1
        max_train_size = self._n_samples - self._h - self._v - 1
        min_val_size = self._v + 1
        max_val_size = 2 * self._v + 1

        min_train_pct = np.round(min_train_size / self._n_samples * 100, 2)
        max_train_pct = np.round(max_train_size / self._n_samples * 100, 2)
        min_val_pct = np.round(min_val_size / self._n_samples * 100, 2)
        max_val_pct = np.round(max_val_size / self._n_samples * 100, 2)

        print("hv-Block CV method")
        print("------------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Number of splits: {self.n_splits}")
        print(
            f"Minimum training set size: {min_train_size} samples ({min_train_pct} %)"
        )
        print(
            f"Maximum training set size: {max_train_size} samples ({max_train_pct} %)"
        )
        print(f"Minimum validation set size: {min_val_size} samples ({min_val_pct} %)")
        print(f"Maximum validation set size: {max_val_size} samples ({max_val_pct} %)")
        print(f"h: {self._h}")
        print(f"v: {self._v}")

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
        >>> from timecave.validation_methods.CV import hvBlockCV
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = hvBlockCV(ts, h=2, v=2);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        The training set is too small to compute most meaningful features.
        The training set is too small to compute most meaningful features.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.0     0.0  0.0  0.0       0.0            0.0  0.000000e+00                0.0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0  0.000000e+00                0.0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0  0.000000e+00                0.0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0  0.000000e+00                0.0               0.0               0.0                inf                 0.0                   0.0
        0   1.0     1.0  1.0  1.0       0.0            0.0 -7.850462e-17                0.0               0.0               0.0                inf                 0.0                   0.0
        0   1.0     1.0  1.0  1.0       0.0            0.0  8.985767e-17                0.0               0.0               0.0                inf                 0.0                   0.0
        0   1.0     1.0  1.0  1.0       0.0            0.0 -8.214890e-17                0.0               0.0               0.0                inf                 0.0                   0.0
        0   1.0     1.0  1.0  1.0       0.0            0.0 -1.050792e-16                0.0               0.0               0.0                inf                 0.0                   0.0
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   1.0     1.0  1.0  1.0      0.00            0.0  8.985767e-17           0.000000               0.0          0.000000                inf                0.00                  0.00
        0   1.0     1.0  1.0  1.0      0.00            0.0 -8.214890e-17           0.000000               0.0          0.000000                inf                0.00                  0.00
        0   1.0     1.0  1.0  1.0      0.00            0.0 -1.050792e-16           0.000000               0.0          0.000000                inf                0.00                  0.00
        0   0.8     1.0  0.0  1.0      0.16            1.0 -2.000000e-01           0.100000               0.4          0.630930           0.923760                0.25                  0.25
        0   0.6     1.0  0.0  1.0      0.24            1.0 -3.000000e-01           0.109017               0.4          0.347041           1.131371                0.25                  0.25
        0   0.4     0.0  0.0  1.0      0.24            1.0 -3.000000e-01           0.134752               0.4          0.347041           1.131371                0.25                  0.25
        0   0.2     0.0  0.0  1.0      0.16            1.0 -2.000000e-01           0.200000               0.4          1.000000           0.923760                0.25                  0.25
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00           0.000000               0.0          0.000000                inf                0.00                  0.00
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00           0.000000               0.0          0.000000                inf                0.00                  0.00
        0   0.0     0.0  0.0  0.0      0.00            0.0  0.000000e+00           0.000000               0.0          0.000000                inf                0.00                  0.00
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        full_features = get_features(self._series, self.sampling_freq)
        training_stats = []
        validation_stats = []

        for training, validation, _ in self.split():

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
        >>> from timecave.validation_methods.CV import hvBlockCV
        >>> ts = np.ones(6);
        >>> splitter = hvBlockCV(ts, h=1, v=1);
        >>> splitter.plot(10, 10);

        ![hv](../../../images/hvBlock_plot.png)
        """

        fig, axs = plt.subplots(self.n_splits, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("hv-Block CV method")

        for it, (training, validation, _) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Fold {}".format(it + 1))
            axs[it].set_ylim([self._series.min() - 1, self._series.max() + 1])
            axs[it].set_xlim([- 1, self._n_samples + 1])
            axs[it].legend()

        plt.show()

        return

class AdaptedhvBlockCV(BaseSplitter):
    """
    Implements the Adapted hv Block Cross-validation method.

    This class implements the Adapted hv Block Cross-validation method. It is similar to the [BlockCV](block.md) class, 
    but it does not support weight generation. Consequently, in order to implement a weighted version of this method, 
    the user must implement their own derived class or compute the weights separately.

    Parameters
    ----------
    splits : int
        The number of folds used to partition the data.

    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    h : int, default=1
        Controls the amount of samples that will be removed from the training set. 
        The `h` samples immediately following and preceding the validation set are not used for training.

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
        If `h` is not an integer.

    ValueError
        If `h` is smaller than or equal to zero.

    ValueError
        If `h` is larger than the number of samples in a fold.

    See also
    --------
    [Block CV](block.md): The original Block CV method, where no training samples are removed.

    Notes
    -----
    The Adapted hv Block Cross-validation method splits the data into $N$ different folds. Then, in every iteration $i$, the model is validated on data
    from the $i^{th}$ folds and trained on data from the remaining folds. There is, however, one subtle difference from the original Block Cross-validation 
    method: the $h$ training samples that lie closest to the validation set are removed, thereby reducing the correlation between the training set and the 
    validation set.
    The average error on the validation sets is then taken as the estimate of the model's true error. This method does not preserve the temporal order of the observations.

    ![block](../../../images/AdaptedhbBlockCV.png)

    This method was proposed by Cerqueira et al. [[1]](#1).
    
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
        h: int = 1,
    ) -> None:

        super().__init__(splits, ts, fs)
        self._check_h(h);
        self._h = h;
        self._splitting_ind = self._split_ind()

    def _check_h(self, h: int) -> None:
        """
        Perform type and value checks on h.
        """

        if (isinstance(h, int)) is False:

            raise TypeError("'h' must be an integer.")

        if h < 0:

            raise ValueError("'h' must be non-negative.")

        if h >= int(np.floor(self._n_samples / self.n_splits)):

            raise ValueError(
                "h should be smaller than the number of samples in a fold."
            )

        return

    def _split_ind(self) -> np.ndarray:
        """
        Compute the splitting indices.
        """

        remainder = int(self._n_samples % self.n_splits)
        split_size = int(np.floor(self._n_samples / self.n_splits))
        split_ind = np.arange(0, self._n_samples, split_size)

        if(remainder != 0):
            
            split_ind[1:remainder+1] += np.array([i for i in range(1, remainder+1)]);
            split_ind[remainder+1:] += remainder;

        else:

            split_ind = np.append(split_ind, self._n_samples);

        if(split_ind.shape[0] > self.n_splits + 1):

            split_ind = split_ind[:self.n_splits+1];

        #split_ind[:remainder] += 1

        #if remainder != 0:

        #    split_ind[remainder:] += remainder

        #split_ind = np.append(split_ind, self._n_samples)

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
            Weight assigned the error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.CV import AdaptedhvBlockCV
        >>> ts = np.ones(10);
        >>> splitter = AdaptedhvBlockCV(5, ts); # Split the data into 5 different folds
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [3 4 5 6 7 8 9]
        Validation set indices: [0 1]
        Iteration 2
        Training set indices: [0 5 6 7 8 9]
        Validation set indices: [2 3]
        Iteration 3
        Training set indices: [0 1 2 7 8 9]
        Validation set indices: [4 5]
        Iteration 4
        Training set indices: [0 1 2 3 4 9]
        Validation set indices: [6 7]
        Iteration 5
        Training set indices: [0 1 2 3 4 5 6]
        Validation set indices: [8 9]

        If the number of samples is not divisible by the number of folds, the first folds will contain more samples:

        >>> ts2 = np.ones(17);
        >>> splitter = AdaptedhvBlockCV(5, ts2, h=2);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ... 
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [ 6  7  8  9 10 11 12 13 14 15 16]
        Validation set indices: [0 1 2 3]
        Iteration 2
        Training set indices: [ 0  1 10 11 12 13 14 15 16]
        Validation set indices: [4 5 6 7]
        Iteration 3
        Training set indices: [ 0  1  2  3  4  5 13 14 15 16]
        Validation set indices: [ 8  9 10]
        Iteration 4
        Training set indices: [ 0  1  2  3  4  5  6  7  8 16]
        Validation set indices: [11 12 13]
        Iteration 5
        Training set indices: [ 0  1  2  3  4  5  6  7  8  9 10 11]
        Validation set indices: [14 15 16]
        """

        for i, (ind) in enumerate(self._splitting_ind[:-1]):

            next_ind = self._splitting_ind[i + 1]

            validation = self._indices[ind:next_ind]
            h_ind = self._indices[
                np.fmax(ind - self._h, 0) : np.fmin(
                    next_ind + self._h, self._n_samples
                )
            ]
            train = np.array([el for el in self._indices if el not in h_ind])

            yield (train, validation, 1.0)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the number of splits and the fold size.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.CV import AdaptedhvBlockCV
        >>> ts = np.ones(10);
        >>> splitter = AdaptedhvBlockCV(5, ts);
        >>> splitter.info();
        Adapted hv Block CV method
        ---------------
        Time series size: 10 samples
        Number of splits: 5
        Fold size: 1 to 2 samples (10.0 to 20.0 %)
        """

        min_fold_size = int(np.floor(self._n_samples / self.n_splits)) - self._h
        max_fold_size = int(np.floor(self._n_samples / self.n_splits))

        remainder = self._n_samples % self.n_splits

        if remainder != 0:

            max_fold_size += 1

        min_fold_size_pct = np.round(min_fold_size / self._n_samples * 100, 2)
        max_fold_size_pct = np.round(max_fold_size / self._n_samples * 100, 2)

        print("Adapted hv Block CV method")
        print("---------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Number of splits: {self.n_splits}")
        print(
            f"Fold size: {min_fold_size} to {max_fold_size} samples ({min_fold_size_pct} to {max_fold_size_pct} %)"
        )

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
        >>> from timecave.validation_methods.CV import AdaptedhvBlockCV
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = AdaptedhvBlockCV(5, ts);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
               Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0  0.285714     0.0  0.0  1.0  0.204082            1.0    -0.178571           0.146421          0.428571          0.702232           1.212183            0.166667              0.166667
        0  0.166667     0.0  0.0  1.0  0.138889            1.0    -0.142857           0.250000          0.500000          1.000000           0.931695            0.200000              0.200000
        0  0.500000     0.5  0.0  1.0  0.250000            1.0    -0.257143           0.138889          0.500000          0.455486           1.250000            0.200000              0.200000
        0  0.833333     1.0  0.0  1.0  0.138889            1.0    -0.142857           0.125000          0.500000          0.792481           0.931695            0.200000              0.200000
        0  0.714286     1.0  0.0  1.0  0.204082            1.0    -0.178571           0.094706          0.428571          0.556506           1.212183            0.166667              0.166667
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   1.0     1.0  1.0  1.0      0.00            0.0 -7.850462e-17               0.00               0.0               0.0                inf                 0.0                   0.0
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
        >>> from timecave.validation_methods.CV import AdaptedhvBlockCV
        >>> ts = np.ones(100);
        >>> splitter = AdaptedhvBlockCV(5, ts, h=5);
        >>> splitter.plot(10, 10);

        ![block_plot](../../../images/AdaptedHV_plot.png)
        """

        fig, axs = plt.subplots(self.n_splits, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Adapted hv Block CV method")

        for it, (training, validation, w) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Fold: {}".format(it + 1))
            axs[it].set_ylim([self._series.min() - 1, self._series.max() + 1])
            axs[it].set_xlim([- 1, self._n_samples + 1])
            axs[it].legend()

        plt.show()

        return
    
if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);