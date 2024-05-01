import numpy as np
from abc import abstractmethod


class BaseFrequency:
    """
    Base class for frequency modulation.

    This class provides a base for implementing frequency modulation techniques.

    Methods
    ----------
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
        ----------
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

    def __init__(self, freq_init, t_split):
        self._check_freq_init(freq_init)
        self._check_t_split(t_split)

        super().__init__()
        self._freq_init = freq_init
        self._t_split = t_split

    def _check_freq_init(self, freq_init: float or int) -> None:
        """
        Checks if the 'freq_init' is a non-negative float or int.
        """
        if isinstance(freq_init, (float, int)):

            raise TypeError("'freq_init' must be a float or int.")

        elif freq_init <= 0:

            raise ValueError("'freq_init' must be greater than zero.")
        return

    def _check_t_split(self, t_split) -> None:
        """
        Checks if the 't_split' is a non-negative float or int.
        """
        if isinstance(t_split, int):

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
        -------
        float or int
            The modulated frequency value at the given time.
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
        if isinstance(freq_init, (float, int)):

            raise TypeError("'freq_init' must be a float or int.")

        elif freq_init <= 0:

            raise ValueError("'freq_init' must be greater than zero.")
        return

    def _check_slope(self, slope: float or int) -> None:
        """
        Checks if the 'slope' is a non-negative float or int.
        """
        if isinstance(slope, (float, int)):

            raise TypeError("'slope' must be a float or int.")

        elif slope <= 0:

            raise ValueError("'slope' must be greater than zero.")
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
        ----------
        np.array
            Each entry of the array corresponds to the frequency at a given timestep.
        """
        return self._freq_init + self._slope * time
