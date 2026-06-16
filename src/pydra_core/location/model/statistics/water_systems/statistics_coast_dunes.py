import numpy as np

from scipy.stats import norm

from typing import Optional

from ..statistics import Statistics
# from ..stochastics.discrete_probability import DiscreteProbability
# from ..stochastics.model_uncertainty import ModelUncertainty
# from ..stochastics.sea_level.sea_level_point import SeaLevelPoint
# from ..stochastics.sea_level.sea_level_triangular import SeaLevelTriangular
# from ..stochastics.sigma_function import SigmaFunction
# from ..stochastics.wind_speed import WindSpeed
from ....settings.settings import Settings
from .....common.interpolate import Interpolate
from .....common.probability import ProbabilityFunctions


class StatisticsCoastDunes(Statistics):
    """
    Statistics class for the Coast Dunes
    Water systems: Coast Dunes
    """

    def __init__(self, settings: Settings):
        """
        Init the Statistics class

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit initialisation method from parent
        super().__init__(settings)

        self.sea_level: Optional[SeaLevel] = None

    def calculate_probability(self, wind_direction: float, closing_situation: int = 1, given: list = []):
        """
        Calculate the probability of occurence for the discretisation given the wind direction.

        Parameters
        ----------
        wind_direction : float
            Wind direction
        closing_situation : int
            Closing situation, (irrelevant for Coast)
        given : list
            Given stochasts
        """

        # Combine all probabilities
        probability = 0

        # Return probability
        return probability