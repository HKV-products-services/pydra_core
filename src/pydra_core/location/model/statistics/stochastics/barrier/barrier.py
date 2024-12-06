import numpy as np

from abc import ABC, abstractmethod

from ..discrete_probability import DiscreteProbability
from .....settings.settings import Settings


class Barrier(ABC):
    def __init__(self, settings: Settings, wind_direction: DiscreteProbability):
        # Save key information
        self.settings = settings
        self.wind_direction = wind_direction
        self.k = None
        self.nk = None

    def __len__(self):
        """
        Return the number of barrier discretisations.

        Returns
        -------
        int
            Number of discretisations
        """
        return self.nk

    def get_discretisation(self) -> np.ndarray:
        """
        Return the barrier discretisation.

        Returns
        -------
        np.ndarray
            1D array with discretisation
        """
        return self.k

    @abstractmethod
    def calculate_closing_probability(
        self, wind_direction: float, closing_situation: int
    ) -> np.ndarray:
        pass
