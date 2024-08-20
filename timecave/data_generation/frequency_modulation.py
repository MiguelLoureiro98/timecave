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
This module contains helper functions that can be used to generate time series with time-varying frequency characteristics.

Classes
-------
BaseFrequency
    Base class for frequency modulation.

FrequencyModulationWithStep
    Frequency modulation where the dominant frequency changes abruptly.

FrequencyModulationLinear
    Frequency modulation where the dominant frequency changes linearly.
"""

import numpy as np
from abc import abstractmethod


class BaseFrequency:
    """
    Base class for frequency modulation.

    This class provides a base for implementing frequency modulation techniques.

    Methods
    -------
    modulate(time)
        Adjusts the frequency based on the given time.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def modulate(self, time: int) -> np.array:
        """
        Adjusts the frequency based on the given time.

        This method should be implemented by subclasses to perform frequency modulation.

        Parameters
        ----------
        time : int
            Number of timesteps for which modulation is meant to be performed.

        Returns
        -------
        np.array
            Each entry of the array corresponds to the frequency at the given timestep.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement modulate method!")


class FrequencyModulationWithStep(BaseFrequency):
    """
    Frequency modulation with step.

    This class implements frequency modulation with a step change.

    Parameters
    ----------
    freq_init : float or int
        The initial frequency value.

    t_split : int
        The timestep at which the frequency changes.

    Methods
    ----------
    modulate(time)
        Adjusts the frequency based on the given time.

    Raises
    ----------
    TypeError
        If 'freq_init' is not a float or int. If 't_split' is not an int.

    ValueError
        If 'freq_init' is not greater than zero. If 't_split' is not greater than zero.
    """

    def __init__(self, freq_init: float or int, t_split: float or int):
        self._check_freq_init(freq_init)
        self._check_t_split(t_split)

        super().__init__()
        self._freq_init = freq_init
        self._t_split = t_split

    def _check_freq_init(self, freq_init: float or int) -> None:
        """
        Checks if the 'freq_init' is a non-negative float or int.
        """
        if isinstance(freq_init, (float, int)) is False:

            raise TypeError("'freq_init' must be a float or int.")

        elif freq_init <= 0:

            raise ValueError("'freq_init' must be greater than zero.")
        return

    def _check_t_split(self, t_split: int or float) -> None:
        """
        Checks if the 't_split' is a non-negative float or int.
        """
        if isinstance(t_split, (int, float)) is False:

            raise TypeError("'t_split' must be int.")

        elif t_split <= 0:

            raise ValueError("'t_split' must be greater than zero.")
        return

    def modulate(self, time: int) -> np.array:
        """
        Adjusts the frequency based on the given time.

        This method calculates the frequency modulation based on the given time
        and the initial frequency value.

        Parameters
        ----------
        time : int
            Number of timesteps for which modulation is meant to be performed.

        Returns
        ----------
        np.array
            Each entry of the array corresponds to the frequency at a given timestep.

        Examples
        --------
        >>> from timecave.data_generation.frequency_modulation import FrequencyModulationWithStep
        >>> mod = FrequencyModulationWithStep(1, 50);
        >>> mod.modulate(25)
        array(1)
        >>> mod.modulate(100)
        array(2)
        >>> mod.modulate(50)
        array(1)
        """
        initial_period = 1 / self._freq_init
        t_split_adjusted = (self._t_split // initial_period) * initial_period
        return np.where(time > t_split_adjusted, self._freq_init * 2, self._freq_init)


class FrequencyModulationLinear(BaseFrequency):
    """
    Represents a linear frequency modulation.

    Parameters
    ----------
    freq_init : float or int
        The initial frequency value.

    slope : float
        Slope of the frequency modulation over time.

    Methods
    ----------
    modulate(time)
        Adjusts the frequency based on the given time.
    """

    def __init__(self, freq_init: int or float, slope: int or float):
        self._check_freq_init(freq_init)
        self._check_slope(slope)
        super().__init__()
        self._slope = slope
        self._freq_init = freq_init

    def _check_freq_init(self, freq_init: float or int) -> None:
        """
        Checks if the 'freq_init' is a non-negative float or int.
        """
        if isinstance(freq_init, (float, int)) is False:

            raise TypeError("'freq_init' must be a float or int.")

        elif freq_init <= 0:

            raise ValueError("'freq_init' must be greater than zero.")
        return

    def _check_slope(self, slope: float or int) -> None:
        """
        Checks if the 'slope' is a non-negative float or int.
        """
        if isinstance(slope, (float, int)) is False:

            raise TypeError("'slope' must be a float or int.")

        elif slope < 0:

            raise ValueError("'slope' must be equal or greater than ozero.")
        return

    def modulate(self, time: int):
        """
        Adjusts the frequency based on the given time.

        This method should be implemented by subclasses to perform frequency modulation.

        Parameters
        ----------
        time : int
            Number of timesteps for which modulation is meant to be performed.

        Returns
        -------
        float or int
            The modulated frequency value at the given time instant.

        Examples
        --------
        >>> from timecave.data_generation.frequency_modulation import FrequencyModulationLinear
        >>> mod = FrequencyModulationLinear(1, 10);
        >>> mod.modulate(5)
        51
        >>> mod.modulate(10)
        101
        """
        return self._freq_init + self._slope * time

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True);