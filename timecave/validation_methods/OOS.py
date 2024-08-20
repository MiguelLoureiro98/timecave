"""
This module contains all the Out-of-Sample (OOS) validation methods supported by this package.

Classes
-------
Holdout
    Implements the classic Holdout method.

RepeatedHoldout
    Implements the Repeated Holdout approach.

RollingOriginUpdate
    Implements the Rolling Origin Update method.

RollingOriginRecalibration
    Implements the Rolling Origin Recalibration method.

FixedSizeRollingWindow
    Implements the Fixed-size Rolling Window method.

See also
--------
[Prequential methods](../prequential/index.md): Prequential or forward validation methods for time series data.

[Cross-validation methods](../CV/index.md): Cross-validation methods for time series data.

[Markov methods](../markov/index.md): Markov cross-validation method for time series data.

Notes
-----
Out-of-sample methods are one of the three main classes of validation methods for time series data (the others being prequential methods and cross-validation methods).
Unlike cross-validation methods, this class of methods preserves the temporal order of observations, although it differs from prequential methods in that it does not \
partition the series into equally sized folds.
For a comprehensive review of this class of methods, the reader should refer to [[1]](#1).

References
----------
##1
Leonard J Tashman. Out-of-sample tests of forecasting accuracy: an analysis
and review. International journal of forecasting, 16(4):437–450, 2000.
"""

from .base import BaseSplitter
from ..data_characteristics import get_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator
from warnings import warn


class Holdout(BaseSplitter):
    """    
    Implements the classic Holdout method.

    This class implements the classic Holdout method, which splits the time series into two disjoint sets: one used for training, and another one used for validation purposes.
    Note that the larger the validation set, the smaller the training set, and vice-versa.
    As this is an Out-of-Sample method, the training indices precede the validation ones.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    validation_size : float, default=0.3
        Validation set size (relative to the time series size).

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
        If the validation size is not a float.

    ValueError
        If the validation size does not lie in the ]0, 1[ interval.

    See also
    --------
    [RepeatedHoldout](rep_holdout.md): Perform several iterations of the Holdout method with a randomised validation set size.

    Notes
    -----
    The classic Holdout method consists of splitting the time series in two different sets: one for training and one for validation.
    This method preserves the temporal order of observations: \
    the oldest set of observations is used for training, while the most recent data is used for validating the model.
    The model's error on the validation set data is used as an estimate of its true error.

    ![OOS_image](../../../images/OOS.png)
    
    This method's computational cost is negligible.
    """

    def __init__(
        self, ts: np.ndarray | pd.Series, fs: float | int = 1, validation_size: float = 0.3
    ) -> None:

        super().__init__(2, ts, fs)
        self._check_validation_size(validation_size)
        self._val_size = validation_size

        return

    def _check_validation_size(self, validation_size: float) -> None:
        """
        Perform type and value checks on the 'validation_size' parameter.
        """

        if isinstance(validation_size, float) is False:

            raise TypeError("'validation_size' must be a float.")

        elif validation_size >= 1 or validation_size <= 0:

            raise ValueError(
                "'validation_size' must be greater than zero and less than one."
            )

        return

    def split(self) -> Generator[tuple[np.ndarray, np.ndarray, float], None, None]:
        """
        Split the time series into training and validation sets.

        This method splits the series' indices into two disjoint sets: one containing the training indices, and another one with the validation indices.
        Note that this method is a generator. To access the indices, use the `next()` method or a `for` loop.

        Yields
        ------
        np.ndarray
            Array of training indices.

        np.ndarray
            Array of validation indices.

        float
            Used for compatibility reasons. Irrelevant for this method.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import Holdout
        >>> ts = np.ones(10);
        >>> splitter = Holdout(ts);
        >>> for train, val, _ in splitter.split():
        ...     
        ...     # Print the training indices and their respective values
        ...     print(f"Training indices: {train}");
        ...     print(f"Training values: {ts[train]}");
        ...     
        ...     # Do the same for the validation indices
        ...     print(f"Validation indices: {val}");
        ...     print(f"Validation values: {ts[val]}");
        Training indices: [0 1 2 3 4 5 6]
        Training values: [1. 1. 1. 1. 1. 1. 1.]
        Validation indices: [7 8 9]
        Validation values: [1. 1. 1.]
        """

        split_ind = int(np.round((1 - self._val_size) * self._n_samples))

        train = self._indices[:split_ind]
        validation = self._indices[split_ind:]

        yield (train, validation, 1.0)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the time series size along with those of the training and validation sets.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import Holdout
        >>> ts = np.ones(10);
        >>> splitter = Holdout(ts);
        >>> splitter.info();
        Holdout method
        --------------
        Time series size: 10 samples
        Training set size: 7 samples (70.0 %)
        Validation set size: 3 samples (30.0 %)
        """

        print("Holdout method")
        print("--------------")
        print(f"Time series size: {self._n_samples} samples")
        print(
            f"Training set size: {int(np.round((1 - self._val_size) * self._n_samples))} samples ({np.round(1 - self._val_size, 4) * 100} %)"
        )
        print(
            f"Validation set size: {int(np.round(self._val_size * self._n_samples))} samples ({np.round(self._val_size, 4) * 100} %)"
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

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import Holdout
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = Holdout(ts, validation_size=0.5);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude   Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   1.0     1.0  1.0  1.0       0.0            0.0 -1.050792e-16                0.0               0.0               0.0                inf                 0.0                   0.0
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        split = self.split()
        training, validation, _ = next(split)

        full_feat = get_features(self._series, self.sampling_freq)

        if self._series[training].shape[0] >= 2:

            training_feat = get_features(self._series[training], self.sampling_freq)

        else:

            training_feat = pd.DataFrame(columns=full_feat.columns)
            warn("Training and validation set statistics can only be computed if each of these comprise two or more samples.")

        if self._series[validation].shape[0] >= 2:

            validation_feat = get_features(self._series[validation], self.sampling_freq)

        else:

            validation_feat = pd.DataFrame(columns=full_feat.columns)
            warn("Training and validation set statistics can only be computed if each of these comprise two or more samples.")

        return (full_feat, training_feat, validation_feat)

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
        >>> from timecave.validation_methods.OOS import Holdout
        >>> ts = np.arange(1, 11);
        >>> splitter = Holdout(ts);
        >>> splitter.plot(10, 10);

        ![Holdout_plot_image](../../../images/Holdout_plot.png)
        """

        split = self.split()
        training, validation, _ = next(split)

        fig = plt.figure(figsize=(height, width))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(training, self._series[training], label="Training set")
        ax.scatter(validation, self._series[validation], label="Validation set")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Time Series")
        ax.set_title("Holdout method")
        ax.legend()
        plt.show()

        return


class RepeatedHoldout(BaseSplitter):
    """
    Implements the Repeated Holdout method.

    This class implements the Repeated Holdout method. This is essentially an extension of the classic Holdout method, as it simply applies \
    this method multiple times with a randomised splitting point. At every iteration, this point is chosen at random from an interval of values \
    specified by the user. For this purpose, our implementation uses a uniform distribution, though, in theory, any continuous distribution could be used.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    iterations : int, default=5
        Number of iterations that should be performed.

    splitting_interval : list[int | float], default=[0.7, 0.8]
        Interval from which the splitting point will be drawn. \
        If the values are integers, they are interpreted as indices. \
        Otherwise, they are regarded as the minimum and maximum allowable \
        sizes for the training set.

    seed : int, default=0
        Random seed.

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
        If the `iterations` parameter is not an integer.

    ValueError
        If the `iterations` parameter is not positive.

    TypeError
        If the splitting interval is not a list.

    ValueError
        If the splitting interval list does not contain two values.
    
    See also
    --------
    [Holdout](holdout.md): The classic Holdout method.

    Notes
    -----
    The Repeated Holdout method is an extension of the classic Holdout method. Essentially, the Holdout method is applied \
    multiple times, and an average of the error on the validation set is used as an estimate of the model's true error.
    At every iteration, the splitting point (and therefore the training and validation set sizes) is computed randomly from \
    an interval of values specified by the user.

    ![Rep_holdout](../../../images/RepHoldout.png)

    Compared to the classic Holdout method, it has a greater computational cost, though, depending on the number \
    of iterations and the prediction model, this may be negligible. For more details on this method, the reader should \
    refer to [[1]](#1).

    References
    ----------
    ##1
    Vitor Cerqueira, Luis Torgo, and Igor Mozetiˇc. Evaluating time series fore-
    casting models: An empirical study on performance estimation methods.
    Machine Learning, 109(11):1997–2028, 2020.
    """

    def __init__(
        self,
        ts: np.ndarray | pd.Series,
        fs: float | int = 1,
        iterations: int = 5,
        splitting_interval: list[int | float] = [0.7, 0.8],
        seed: int = 0,
    ) -> None:

        self._check_iterations(iterations)
        self._check_splits(splitting_interval)
        super().__init__(iterations, ts, fs)
        self._iter = iterations
        self._interval = self._convert_interval(splitting_interval)
        self._seed = seed
        self._splitting_ind = self._get_splitting_ind()

        return

    def _check_iterations(self, iterations: int) -> None:
        """
        Perform type and value checks on the number of iterations.
        """

        if isinstance(iterations, int) is False:

            raise TypeError("The number of iterations must be an integer.")

        if iterations <= 0:

            raise ValueError("The number of iterations must be positive.")

    def _check_splits(self, splitting_interval: list) -> None:
        """
        Perform several type and value checks on the splitting interval.
        """

        if isinstance(splitting_interval, list) is False:

            raise TypeError("The splitting interval must be a list.")

        if len(splitting_interval) > 2:

            raise ValueError(
                "The splitting interval should be composed of two elements."
            )

        for element in splitting_interval:

            if (isinstance(element, int) or isinstance(element, float)) is False:

                raise TypeError(
                    "The interval must be entirely composed of integers or floats."
                )

        if splitting_interval[0] > splitting_interval[1]:

            raise ValueError(
                "'splitting_interval' should have the [smaller_value, larger_value] format."
            )

        return

    def _convert_interval(self, splitting_interval: list) -> list:
        """
        Convert intervals of floats (percentages) into intervals of integers (indices).
        """

        for ind, element in enumerate(splitting_interval):

            if isinstance(element, float) is True:

                splitting_interval[ind] = int(np.round(element * self._n_samples))

        return splitting_interval

    def _check_seed(self, seed: int) -> None:
        """
        Perform a type check on the seed.
        """

        if isinstance(seed, int) is False:

            raise TypeError("'seed' should be an integer.")

        return

    def _get_splitting_ind(self) -> np.ndarray:
        """
        Generate the splitting indices.
        """

        np.random.seed(self._seed)
        rand_ind = np.random.randint(
            low=self._interval[0], high=self._interval[1], size=self._iter
        )

        return rand_ind

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
            Used for compatibility reasons. Irrelevant for this method.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RepeatedHoldout
        >>> ts = np.ones(100);

        If the splitting interval consists of two floats, the method assumes they define the minimum and maximum training set sizes:

        >>> splitter = RepeatedHoldout(ts, splitting_interval=[0.6, 0.8]);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ...
        ...     print(f"Iteration {ind+1}");
        ...     print(f"# training samples: {train.shape[0]}");
        ...     print(f"# validation samples: {val.shape[0]}");
        Iteration 1
        # training samples: 72
        # validation samples: 28
        Iteration 2
        # training samples: 75
        # validation samples: 25
        Iteration 3
        # training samples: 60
        # validation samples: 40
        Iteration 4
        # training samples: 63
        # validation samples: 37
        Iteration 5
        # training samples: 63
        # validation samples: 37

        If two integers are specified instead, they will be regarded as indices.

        >>> splitter = RepeatedHoldout(ts, splitting_interval=[80, 95]);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ...
        ...     print(f"Iteration {ind+1}");
        ...     print(f"# training samples: {train.shape[0]}");
        ...     print(f"# validation samples: {val.shape[0]}");
        Iteration 1
        # training samples: 92
        # validation samples: 8
        Iteration 2
        # training samples: 85
        # validation samples: 15
        Iteration 3
        # training samples: 80
        # validation samples: 20
        Iteration 4
        # training samples: 83
        # validation samples: 17
        Iteration 5
        # training samples: 91
        # validation samples: 9
        """

        for ind in self._splitting_ind:

            training = self._indices[:ind]
            validation = self._indices[ind:]

            yield (training, validation, 1.0)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the average, minimum and maximum validation set sizes.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RepeatedHoldout
        >>> ts = np.ones(10);
        >>> splitter = RepeatedHoldout(ts, splitting_interval=[0.6, 0.9]);
        >>> splitter.info();
        Repeated Holdout method
        -----------------------
        Time series size: 10 samples
        Average validation set size: 3.4 samples (34.0 %)
        Maximum validation set size: 4 samples (40.0 %)
        Minimum validation set size: 3 samples (30.0 %)
        """

        mean_size = self._n_samples - self._splitting_ind.mean()
        min_size = self._n_samples - self._splitting_ind.max()
        max_size = self._n_samples - self._splitting_ind.min()

        mean_pct = np.round(mean_size / self._n_samples, 4) * 100
        max_pct = np.round(max_size / self._n_samples, 4) * 100
        min_pct = np.round(min_size / self._n_samples, 4) * 100

        print("Repeated Holdout method")
        print("-----------------------")
        print(f"Time series size: {self._n_samples} samples")
        print(f"Average validation set size: {np.round(mean_size, 4)} samples ({mean_pct} %)")
        print(f"Maximum validation set size: {max_size} samples ({max_pct} %)")
        print(f"Minimum validation set size: {min_size} samples ({min_pct} %)")

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

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RepeatedHoldout
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = RepeatedHoldout(ts, splitting_interval=[0.6, 0.9]);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
               Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0  0.833333     1.0  0.0  1.0  0.138889            1.0    -0.142857           0.125000          0.500000          0.792481           0.931695            0.200000              0.200000
        0  0.714286     1.0  0.0  1.0  0.204082            1.0    -0.178571           0.094706          0.428571          0.556506           1.212183            0.166667              0.166667
        0  0.833333     1.0  0.0  1.0  0.138889            1.0    -0.142857           0.125000          0.500000          0.792481           0.931695            0.200000              0.200000
        0  0.714286     1.0  0.0  1.0  0.204082            1.0    -0.178571           0.094706          0.428571          0.556506           1.212183            0.166667              0.166667
        0  0.714286     1.0  0.0  1.0  0.204082            1.0    -0.178571           0.094706          0.428571          0.556506           1.212183            0.166667              0.166667
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        full_features = get_features(self._series, self.sampling_freq)

        training_stats = []
        validation_stats = []

        # for ind in self._splitting_ind:

        #    training_feat = get_features(self._series[:ind], self.sampling_freq);
        #    validation_feat = get_features(self._series[ind:], self.sampling_freq);
        #    training_stats.append(training_feat);
        #    validation_stats.append(validation_feat);

        for training, validation, _ in self.split():

            if self._series[training].shape[0] >= 2:

                training_feat = get_features(self._series[training], self.sampling_freq)
                training_stats.append(training_feat)

            else:

                warn(
                    "The training set is too small to compute most meaningful features."
                )

            if self._series[validation].shape[0] >= 2:

                validation_feat = get_features(
                    self._series[validation], self.sampling_freq
                )
                validation_stats.append(validation_feat)

            else:

                warn(
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
        >>> from timecave.validation_methods.OOS import RepeatedHoldout
        >>> ts = np.ones(100);
        >>> splitter = RepeatedHoldout(ts, splitting_interval=[0.6, 0.9]);
        >>> splitter.plot(10, 10);

        ![Holdout_plot_image](../../../images/RepHoldout_plot.png)
        """

        fig, axs = plt.subplots(self._iter, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Repeated Holdout method")

        for it, (training, validation, _) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Iteration {}".format(it + 1))
            axs[it].legend()

        plt.show()

        return


class RollingOriginUpdate(BaseSplitter):
    """
    Implements the Rolling Origin Update method.

    This class implements the Rolling Origin Update method. This method splits the data into a single training set and several validation sets.
    The validation sets are NOT disjoint, and, even though the model is validated several times, it only undergoes the training process once.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    origin : int | float, default=0.7
        The point from which the data is split. \
        If an integer is passed, it is interpreted as an index. \
        If a float is passed instead, it is treated as the percentage of samples \
        that should be used for training.

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
        If `origin` is neither an integer nor a float.

    ValueError
        If `origin` is a float that does not lie in the ]0, 1[ interval.

    ValueError
        If `origin` is an integer that does not lie in the ]0, n_samples[ interval.

    Warning
    -------
    The model should only be trained ONCE.

    See also
    --------
    [Rolling Origin Recalibration](roll_recal.md): Similar to the Rolling Origin Update method, but the training set is updated along with the validation set.
    
    Notes
    -----
    The Rolling Origin Update method consists of splitting the data into a training set and a validation set, \
    with the former preceding the latter. The model is only trained once. \
    Then, at every iteration, a single data point (the one closest to the training set) is dropped from the \
    validation set, and the model's performance is assessed using the new validation set. This process ends once \
    the validation set consists of a single data point. The estimate of the true model error is the average validation \
    error.
    
    ![RollUpdate](../../../images/RollUpdate.png)

    For more details on this method, the reader should refer to [[1]](#1).

    References
    ----------
    ##1
    Leonard J Tashman. Out-of-sample tests of forecasting accuracy: an analysis
    and review. International journal of forecasting, 16(4):437–450, 2000.
    """

    def __init__(
        self, ts: np.ndarray | pd.Series, fs: float | int = 1, origin: int | float = 0.7
    ) -> None:

        super().__init__(2, ts, fs)
        self._check_origin(origin)
        self._origin = self._convert_origin(origin)
        self._splitting_ind = np.arange(self._origin + 1, self._n_samples)
        self._n_splits = self._splitting_ind.shape[0]

        return

    def _check_origin(self, origin: int | float) -> None:
        """
        Perform type and value checks on the origin.
        """

        is_int = isinstance(origin, int)
        is_float = isinstance(origin, float)

        if (is_int or is_float) is False:

            raise TypeError("'origin' must be an integer or a float.")

        if is_float and (origin >= 1 or origin <= 0):

            raise ValueError(
                "If 'origin' is a float, it must lie in the interval of ]0, 1[."
            )

        if is_int and (origin >= self._n_samples or origin <= 0):

            raise ValueError(
                "If 'origin' is an integer, it must lie in the interval of ]0, n_samples[."
            )

        return

    def _convert_origin(self, origin: int | float) -> int:
        """
        Cast the origin from float (proportion) to integer (index).
        """

        if isinstance(origin, float) is True:

            origin = int(np.round(origin * self._n_samples)) - 1

        return origin

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
            Used for compatibility reasons. Irrelevant for this method.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RollingOriginUpdate
        >>> ts = np.ones(10);
        >>> splitter = RollingOriginUpdate(ts);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ...
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Validation set indices: {val}");
        ...     print(f"Validation set size: {val.shape[0]}");
        Iteration 1
        Validation set indices: [7 8 9]
        Validation set size: 3
        Iteration 2
        Validation set indices: [8 9]
        Validation set size: 2
        Iteration 3
        Validation set indices: [9]
        Validation set size: 1
        """

        for ind in self._splitting_ind:

            training = self._indices[: self._origin + 1]
            validation = self._indices[ind:]

            yield (training, validation, 1.0)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the training set size and both the minimum and maximum validation set size.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RollingOriginUpdate
        >>> ts = np.ones(10);
        >>> splitter = RollingOriginUpdate(ts);
        >>> splitter.info();
        Rolling Origin Update method
        ----------------------------
        Time series size: 10 samples
        Training set size (fixed parameter): 7 samples (70.0 %)
        Maximum validation set size: 3 samples (30.0 %)
        Minimum validation set size: 1 sample (10.0 %)
        """

        training_size = self._origin + 1
        max_size = self._n_samples - self._origin - 1
        min_size = 1

        training_pct = np.round(training_size / self._n_samples, 4) * 100
        max_pct = np.round(max_size / self._n_samples, 4) * 100
        min_pct = np.round(1 / self._n_samples, 4) * 100

        print("Rolling Origin Update method")
        print("----------------------------")
        print(f"Time series size: {self._n_samples} samples")
        print(
            f"Training set size (fixed parameter): {training_size} samples ({training_pct} %)"
        )
        print(f"Maximum validation set size: {max_size} samples ({max_pct} %)")
        print(f"Minimum validation set size: {min_size} sample ({min_pct} %)")

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

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RollingOriginUpdate
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = RollingOriginUpdate(ts);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        Training and validation set features can only computed if each set is composed of two or more samples.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
               Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0  0.714286     1.0  0.0  1.0  0.204082            1.0    -0.178571           0.094706          0.428571          0.556506           1.212183            0.166667              0.166667
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        full_features = get_features(self._series, self.sampling_freq)
        it_1 = True
        validation_stats = []

        print(
            "Training and validation set features can only computed if each set is composed of two or more samples."
        )

        for training, validation, _ in self.split():

            if it_1 is True and self._series[training].shape[0] >= 2:

                training_features = get_features(
                    self._series[training], self.sampling_freq
                )
                it_1 = False

            if self._series[validation].shape[0] >= 2:

                validation_feat = get_features(
                    self._series[validation], self.sampling_freq
                )
                validation_stats.append(validation_feat)

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
        >>> from timecave.validation_methods.OOS import RollingOriginUpdate
        >>> ts = np.arange(1, 11);
        >>> splitter = RollingOriginUpdate(ts);
        >>> splitter.plot(10, 10);

        ![Update_plot_image](../../../images/RollUpdate_plot.png)
        """

        fig, axs = plt.subplots(self._n_samples - self._origin - 1, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Rolling Origin Update method")

        for it, (training, validation, _) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Iteration {}".format(it + 1))
            axs[it].legend()

        plt.show()

        return


class RollingOriginRecalibration(BaseSplitter):
    """
    Implements the Rolling Origin Recalibration method.

    This class implements the Rolling Origin Recalibration method. This method splits the data into various training and validation sets.
    Neither the training sets nor the validation sets are disjoint. At every iteration, a single data point is dropped from the validation set and added \
    to the training set.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int
        Sampling frequency (Hz).

    origin : int | float, default=0.7
        The point from which the data is split. \
        If an integer is passed, it is interpreted as an index. \
        If a float is passed instead, it is treated as the percentage of samples \
        that should be used for training.

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
        If `origin` is neither an integer nor a float.

    ValueError
        If `origin` is a float that does not lie in the ]0, 1[ interval.

    ValueError
        If `origin` is an integer that does not lie in the ]0, n_samples[ interval.

    Warning
    -------
    Depending on the time series' size, this method can have a large computational cost.

    See also
    --------
    [Rolling Origin Update](roll_update.md): Similar to the Rolling Origin Recalibration method, but the model is only trained once.
    
    Notes
    -----
    The Rolling Origin Recalibration method consists of splitting the data into a training set and a validation set, \
    with the former preceding the latter. At every iteration, a single data point (the one closest to the training set) is dropped from the \
    validation set and added to the training set. The model's performance is then assessed on the new validation set. This process ends once \
    the validation set consists of a single data point. The estimate of the true model error is the average validation \
    error across [over] all iterations.
    
    ![RollRecal](../../../images/RollRecal.png)

    For more details on this method, the reader should refer to [[1]](#1).

    References
    ----------
    ##1
    Leonard J Tashman. Out-of-sample tests of forecasting accuracy: an analysis
    and review. International journal of forecasting, 16(4):437–450, 2000.
    """

    def __init__(
        self, ts: np.ndarray | pd.Series, fs: float | int = 1, origin: int | float = 0.7
    ) -> None:

        super().__init__(2, ts, fs)
        self._check_origin(origin)
        self._origin = self._convert_origin(origin)
        self._splitting_ind = np.arange(self._origin + 1, self._n_samples)
        self._n_splits = self._splitting_ind.shape[0]

        return

    def _check_origin(self, origin: int | float) -> None:
        """
        Perform type and value checks on the origin.
        """

        is_int = isinstance(origin, int)
        is_float = isinstance(origin, float)

        if (is_int or is_float) is False:

            raise TypeError("'origin' must be an integer or a float.")

        if is_float and (origin >= 1 or origin <= 0):

            raise ValueError(
                "If 'origin' is a float, it must lie in the interval of ]0, 1[."
            )

        if is_int and (origin >= self._n_samples or origin <= 0):

            raise ValueError(
                "If 'origin' is an integer, it must lie in the interval of ]0, n_samples[."
            )

        return

    def _convert_origin(self, origin: int | float) -> int:
        """
        Cast the origin from float (proportion) to integer (index).
        """

        if isinstance(origin, float) is True:

            origin = int(np.round(origin * self._n_samples)) - 1

        return origin

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
            Used for compatibility reasons. Irrelevant for this method.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RollingOriginRecalibration
        >>> ts = np.ones(10);
        >>> splitter = RollingOriginRecalibration(ts);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ...
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1 2 3 4 5 6]
        Validation set indices: [7 8 9]
        Iteration 2
        Training set indices: [0 1 2 3 4 5 6 7]
        Validation set indices: [8 9]
        Iteration 3
        Training set indices: [0 1 2 3 4 5 6 7 8]
        Validation set indices: [9]
        """

        for ind in self._splitting_ind:

            training = self._indices[:ind]
            validation = self._indices[ind:]

            yield (training, validation, 1.0)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the minimum and maximum sizes for both the training and validation sets.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RollingOriginRecalibration
        >>> ts = np.ones(10);
        >>> splitter = RollingOriginRecalibration(ts);
        >>> splitter.info();
        Rolling Origin Recalibration method
        -----------------------------------
        Time series size: 10 samples
        Minimum training set size: 7 samples (70.0 %)
        Maximum validation set size: 3 samples (30.0 %)
        Maximum training set size: 9 samples (90.0 %)
        Minimum validation set size: 1 samples (10.0 %)
        """

        max_training_size = self._n_samples - 1
        min_training_size = self._origin + 1
        max_validation_size = self._n_samples - self._origin - 1
        min_validation_size = 1

        max_training_pct = np.round(max_training_size / self._n_samples, 4) * 100
        min_training_pct = np.round(min_training_size / self._n_samples, 4) * 100
        max_validation_pct = np.round(max_validation_size / self._n_samples, 4) * 100
        min_validation_pct = np.round(min_validation_size / self._n_samples, 4) * 100

        print("Rolling Origin Recalibration method")
        print("-----------------------------------")
        print(f"Time series size: {self._n_samples} samples")
        print(
            f"Minimum training set size: {min_training_size} samples ({min_training_pct} %)"
        )
        print(
            f"Maximum validation set size: {max_validation_size} samples ({max_validation_pct} %)"
        )
        print(
            f"Maximum training set size: {max_training_size} samples ({max_training_pct} %)"
        )
        print(
            f"Minimum validation set size: {min_validation_size} samples ({min_validation_pct} %)"
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

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import RollingOriginRecalibration
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = RollingOriginRecalibration(ts);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        Training and validation set features can only computed if each set is composed of two or more samples.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
               Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0  0.714286     1.0  0.0  1.0  0.204082            1.0    -0.178571           0.094706          0.428571          0.556506           1.212183            0.166667              0.166667
        0  0.625000     1.0  0.0  1.0  0.234375            1.0    -0.178571           0.122818          0.500000          0.600876           1.383496            0.142857              0.142857
        0  0.555556     1.0  0.0  1.0  0.246914            1.0    -0.166667           0.105483          0.444444          0.385860           1.502496            0.125000              0.125000
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        full_features = get_features(self._series, self.sampling_freq)
        training_stats = []
        validation_stats = []

        print(
            "Training and validation set features can only computed if each set is composed of two or more samples."
        )

        for training, validation, _ in self.split():

            if self._series[training].shape[0] >= 2:

                training_feat = get_features(self._series[training], self.sampling_freq)
                training_stats.append(training_feat)

            if self._series[validation].shape[0] >= 2:

                validation_feat = get_features(
                    self._series[validation], self.sampling_freq
                )
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
        >>> from timecave.validation_methods.OOS import RollingOriginRecalibration
        >>> ts = np.arange(1, 11);
        >>> splitter = RollingOriginRecalibration(ts);
        >>> splitter.plot(10, 10);

        ![Holdout_plot_image](../../../images/RollRecal_plot.png)
        """

        fig, axs = plt.subplots(self._n_samples - self._origin - 1, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Rolling Origin Recalibration method")

        for it, (training, validation, _) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Iteration {}".format(it + 1))
            axs[it].legend()

        plt.show()

        return


class FixedSizeRollingWindow(BaseSplitter):
    """
    Implements the Fixed Size Rolling Window method.

    This class implements the Fixed Size Rolling Window method. This method splits the data into a single training set and several validation sets.
    Neither the training sets nor the validation sets are disjoint. At every iteration, a single data point is dropped from the validation set and added \
    to the training set. The oldest data point belonging to the training set is also discarded, so that the amount of training samples remains constant.

    Parameters
    ----------
    ts : np.ndarray | pd.Series
        Univariate time series.

    fs : float | int, default=1
        Sampling frequency (Hz).

    origin : int | float, default=0.7
        The point from which the data is split. \
        If an integer is passed, it is interpreted as an index. \
        If a float is passed instead, it is treated as the percentage of samples \
        that should be used for training.

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
        If `origin` is neither an integer nor a float.

    ValueError
        If `origin` is a float that does not lie in the ]0, 1[ interval.

    ValueError
        If `origin` is an integer that does not lie in the ]0, n_samples[ interval.
    
    Warning
    -------
    Depending on the time series' size, this method can have a large computational cost.

    Notes
    -----
    The Fixed Size Rolling Origin method consists of splitting the data into a training set and a validation set, \
    with the former preceding the latter. At every iteration, a single data point (the one closest to the training set) is dropped from the \
    validation set and added to the training set. Additionally, the oldest data point belonging to the training set is discarded, so that the amount of training samples remains constant. \
    The model is then trained on the new training set and tested on the new validation set. This process ends once \
    the validation set consists of a single data point. The estimate of the true model error is the average validation \
    error.
    
    ![FixedSizeRoll](../../../images/FixedRoll.png)

    For more details on this method, the reader should refer to [[1]](#1).

    References
    ----------
    ##1
    Leonard J Tashman. Out-of-sample tests of forecasting accuracy: an analysis
    and review. International journal of forecasting, 16(4):437–450, 2000.
    """

    def __init__(
        self, ts: np.ndarray | pd.Series, fs: float | int = 1, origin: int | float = 0.7
    ) -> None:

        super().__init__(2, ts, fs)
        self._check_origin(origin)
        self._origin = self._convert_origin(origin)
        self._splitting_ind = np.arange(self._origin + 1, self._n_samples)
        self._n_splits = self._splitting_ind.shape[0]

        return

    def _check_origin(self, origin: int | float) -> None:
        """
        Perform type and value checks on the origin.
        """

        is_int = isinstance(origin, int)
        is_float = isinstance(origin, float)

        if (is_int or is_float) is False:

            raise TypeError("'origin' must be an integer or a float.")

        if is_float and (origin >= 1 or origin <= 0):

            raise ValueError(
                "If 'origin' is a float, it must lie in the interval of ]0, 1[."
            )

        if is_int and (origin >= self._n_samples or origin <= 0):

            raise ValueError(
                "If 'origin' is an integer, it must lie in the interval of ]0, n_samples[."
            )

        return

    def _convert_origin(self, origin: int | float) -> int:
        """
        Cast the origin from float (proportion) to integer (index).
        """

        if isinstance(origin, float) is True:

            origin = int(np.round(origin * self._n_samples)) - 1

        return origin

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
            Used for compatibility reasons. Irrelevant for this method.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import FixedSizeRollingWindow
        >>> ts = np.ones(10);
        >>> splitter = FixedSizeRollingWindow(ts);
        >>> for ind, (train, val, _) in enumerate(splitter.split()):
        ...
        ...     print(f"Iteration {ind+1}");
        ...     print(f"Training set indices: {train}");
        ...     print(f"Validation set indices: {val}");
        Iteration 1
        Training set indices: [0 1 2 3 4 5 6]
        Validation set indices: [7 8 9]
        Iteration 2
        Training set indices: [1 2 3 4 5 6 7]
        Validation set indices: [8 9]
        Iteration 3
        Training set indices: [2 3 4 5 6 7 8]
        Validation set indices: [9]
        """
        start_training_ind = self._splitting_ind - self._origin - 1

        for start_ind, end_ind in zip(start_training_ind, self._splitting_ind):

            training = self._indices[start_ind:end_ind]
            validation = self._indices[end_ind:]

            yield (training, validation, 1.0)

    def info(self) -> None:
        """
        Provide some basic information on the training and validation sets.

        This method displays the minimum and maximum validation set size, as well as the training set size.

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import FixedSizeRollingWindow
        >>> ts = np.ones(10);
        >>> splitter = FixedSizeRollingWindow(ts);
        >>> splitter.info();
        Fixed-size Rolling Window method
        --------------------------------
        Time series size: 10 samples
        Training set size (fixed parameter): 7 samples (70.0 %)
        Maximum validation set size: 3 samples (30.0 %)
        Minimum validation set size: 1 sample (10.0 %)
        """

        training_size = self._origin + 1
        max_size = self._n_samples - self._origin - 1
        min_size = 1

        training_pct = np.round(training_size / self._n_samples, 4) * 100
        max_pct = np.round(max_size / self._n_samples, 4) * 100
        min_pct = np.round(1 / self._n_samples, 4) * 100

        print("Fixed-size Rolling Window method")
        print("--------------------------------")
        print(f"Time series size: {self._n_samples} samples")
        print(
            f"Training set size (fixed parameter): {training_size} samples ({training_pct} %)"
        )
        print(f"Maximum validation set size: {max_size} samples ({max_pct} %)")
        print(f"Minimum validation set size: {min_size} sample ({min_pct} %)")

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

        Examples
        --------
        >>> import numpy as np
        >>> from timecave.validation_methods.OOS import FixedSizeRollingWindow
        >>> ts = np.hstack((np.ones(5), np.zeros(5)));
        >>> splitter = FixedSizeRollingWindow(ts);
        >>> ts_stats, training_stats, validation_stats = splitter.statistics();
        Frequency features are only meaningful if the correct sampling frequency is passed to the class.
        Training and validation set features can only computed if each set is composed of two or more samples.
        >>> ts_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.5     0.5  0.0  1.0      0.25            1.0    -0.151515           0.114058               0.5           0.38717            1.59099            0.111111              0.111111
        >>> training_stats
               Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0  0.714286     1.0  0.0  1.0  0.204082            1.0    -0.178571           0.094706          0.428571          0.556506           1.212183            0.166667              0.166667
        0  0.571429     1.0  0.0  1.0  0.244898            1.0    -0.214286           0.108266          0.428571          0.387375           1.327880            0.166667              0.166667
        0  0.428571     0.0  0.0  1.0  0.244898            1.0    -0.214286           0.124661          0.428571          0.387375           1.327880            0.166667              0.166667
        >>> validation_stats
           Mean  Median  Min  Max  Variance  P2P_amplitude  Trend_slope  Spectral_centroid  Spectral_rolloff  Spectral_entropy  Strength_of_trend  Mean_crossing_rate  Median_crossing_rate
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        0   0.0     0.0  0.0  0.0       0.0            0.0          0.0                  0               0.0               0.0                inf                 0.0                   0.0
        """

        if self._n_samples <= 2:

            raise ValueError(
                "Basic statistics can only be computed if the time series comprises more than two samples."
            )

        print("Frequency features are only meaningful if the correct sampling frequency is passed to the class.")

        full_features = get_features(self._series, self.sampling_freq)
        training_stats = []
        validation_stats = []

        print(
            "Training and validation set features can only computed if each set is composed of two or more samples."
        )

        for training, validation, _ in self.split():

            if self._series[training].shape[0] >= 2:

                training_feat = get_features(self._series[training], self.sampling_freq)
                training_stats.append(training_feat)

            if self._series[validation].shape[0] >= 2:

                validation_feat = get_features(
                    self._series[validation], self.sampling_freq
                )
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
        >>> from timecave.validation_methods.OOS import FixedSizeRollingWindow
        >>> ts = np.ones(10);
        >>> splitter = FixedSizeRollingWindow(ts);
        >>> splitter.plot(10, 10);

        ![Holdout_plot_image](../../../images/FixedRoll_plot.png)
        """

        fig, axs = plt.subplots(self._n_samples - self._origin - 1, 1, sharex=True)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.supxlabel("Samples")
        fig.supylabel("Time Series")
        fig.suptitle("Fixed-size Rolling Window method")

        for it, (training, validation, _) in enumerate(self.split()):

            axs[it].scatter(training, self._series[training], label="Training set")
            axs[it].scatter(
                validation, self._series[validation], label="Validation set"
            )
            axs[it].set_title("Iteration {}".format(it + 1))
            axs[it].legend()

        plt.show()

        return

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);