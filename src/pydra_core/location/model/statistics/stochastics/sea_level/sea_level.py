import numpy as np

from abc import ABC


class SeaLevel(ABC):
    """
    Class to describe the sea (water) level statistics.
    Sea level statistics are conditional on the wind direction.
    """

    def __init__(self):
        """
        Constructor class for the SeaLevel statistics.
        """
        self.nm = 0
        self.m = None
        self.epm = None

    def __len__(self):
        """
        Return the number of sea level discretisations.

        Returns
        -------
        int
            Number of discretisations
        """
        return self.nm

    def get_discretisation(self) -> np.ndarray:
        """
        Return the sea level discretisation.

        Returns
        -------
        np.ndarray
            1D array with discretisation
        """
        return self.m

    def get_exceedance_probability(self) -> np.ndarray:
        """
        Return exceedance probility of the sea level, conditional on the wind
        direction.

        Returns
        -------
        np.ndarray
            A 2D array with the sea level exceedance probability conditional on
            the wind direction (r x m)
        """
        return self.epm
