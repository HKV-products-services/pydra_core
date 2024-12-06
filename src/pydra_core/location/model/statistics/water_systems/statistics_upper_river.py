import numpy as np

from ..statistics import Statistics
from ..stochastics.discharge import Discharge
from ..stochastics.discrete_probability import DiscreteProbability
from ..stochastics.model_uncertainty import ModelUncertainty
from ..stochastics.wind_speed import WindSpeed
from ....settings.settings import Settings
from .....common.probability import ProbabilityFunctions


class StatisticsUpperRiver(Statistics):
    """
    Statistics class for the Upper Rivers
    Water systems: Rhine Non Tidal and Meuse (Valley) Non Tidal
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

        # Discharge
        self.discharge = Discharge(settings)

        # Model uncertainty
        self.model_uncertainties = ModelUncertainty(settings)

        # Discrete, slow, fast stochatics
        self.stochastics_discrete = {
            "r": self.wind_direction.get_discretisation(),
            "k": [1],
        }
        self.stochastics_fast = {"u": self.wind_speed.get_discretisation()}
        self.stochastics_slow = {"q": self.discharge.get_discretisation()}

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

        # Discharge
        if "q" in given:
            p_discharge = np.ones_like(self.discharge.get_discretisation())
        else:
            # If not given, use the instantaneous probability
            p_discharge = ProbabilityFunctions.probability_density(
                self.discharge.get_discretisation(),
                1 - self.discharge.get_exceedance_probability(),
            ).probability

        # Probability of wind direction
        p_direction = 1.0 if "r" in given else self.wind_direction.get_probability()[ir]

        # Combine probabilities
        return p_wind * p_discharge[None, :] * p_direction
