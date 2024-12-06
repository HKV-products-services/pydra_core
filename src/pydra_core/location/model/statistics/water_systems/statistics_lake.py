import numpy as np

from ..statistics import Statistics
from ..stochastics.discrete_probability import DiscreteProbability
from ..stochastics.lake_level import LakeLevel
from ..stochastics.model_uncertainty import ModelUncertainty
from ..stochastics.wind_speed import WindSpeed
from ....settings.settings import Settings
from .....common.probability import ProbabilityFunctions


class StatisticsLake(Statistics):
    """
    Statistics class for the Lakes
    Water systems: IJssel Lake, Marker Lake, Veluwe Lakes and Grevelingen
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

        # Wind
        self.wind_direction = DiscreteProbability(settings.wind_direction_probability)
        self.wind_speed = WindSpeed(settings)

        # Lake level
        self.lake_level = LakeLevel(settings)

        # Model uncertainty
        self.model_uncertainties = ModelUncertainty(settings)

        # Discrete, slow, fast stochatics
        self.stochastics_discrete = {
            "r": self.wind_direction.get_discretisation(),
            "k": [1],
        }
        self.stochastics_fast = {"u": self.wind_speed.get_discretisation()}
        self.stochastics_slow = {"a": self.lake_level.get_discretisation()}

    def calculate_probability(
        self, wind_direction: float, closing_situation: int = 1, given: list = []
    ):
        """
        Calculate the probability of occurence for the discretisation given the wind direction.

        Parameters
        ----------
        direction : float
            Wind direction
        closing_situation : int
            Closing situation, (irrelevant for Coast)
        given : list
            Given stochasts
        """
        # Wind speed
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
        p_wind = ProbabilityFunctions.probability_density(
            self.wind_speed.get_discretisation(),
            self.wind_speed.get_exceedance_probability()[:, ir],
        ).probability[:, None]
        if "u" in given:
            p_wind[:] = 1.0

        # Lake level
        if "a" in given:
            p_lake_level = np.ones_like(self.lake_level.get_discretisation())
        else:
            # If not given, use the instantaneous probability
            p_lake_level = ProbabilityFunctions.probability_density(
                self.lake_level.get_discretisation(),
                1 - self.lake_level.get_exceedance_probability(),
            ).probability

        # Probability of wind direction
        p_direction = 1.0 if "r" in given else self.wind_direction.get_probability()[ir]

        # Combine probabilities
        return p_wind * p_lake_level[None, :] * p_direction
