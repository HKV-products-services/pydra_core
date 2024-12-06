import numpy as np

from scipy.stats import norm

from ..statistics import Statistics
from ..stochastics.discrete_probability import DiscreteProbability
from ..stochastics.model_uncertainty import ModelUncertainty
from ..stochastics.sea_level.sea_level_point import SeaLevelPoint
from ..stochastics.sea_level.sea_level_triangular import SeaLevelTriangular
from ..stochastics.sigma_function import SigmaFunction
from ..stochastics.wind_speed import WindSpeed
from ....settings.settings import Settings
from .....common.interpolate import Interpolate
from .....common.probability import ProbabilityFunctions


class StatisticsCoast(Statistics):
    """
    Statistics class for the Coast
    Water systems: Coast (North, Central, South), Waddensea (West, East) and Western scheldt
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

        # Sea level (Most parts require triangular interpolation, some one point (Den Helder))
        if settings.sea_level_probability is not None:
            self.sea_level = SeaLevelPoint(settings)
        else:
            self.sea_level = SeaLevelTriangular(settings)

        # Sigmafunctie
        self.sigma_function = SigmaFunction(settings)

        # Wind
        self.wind_direction = DiscreteProbability(settings.wind_direction_probability)
        self.wind_speed = WindSpeed(settings)
        self.wind_speed.correct_with_sigma_function(
            self.sigma_function, self.wind_direction
        )

        # Calculate the probability P(m, u, r)
        self.__calculate_combined_probabilities()

        # Model uncertainty
        self.model_uncertainties = ModelUncertainty(settings)

        # Discrete, slow, fast stochatics
        # TODO: Replace with framework
        self.stochastics_discrete = {
            "r": self.wind_direction.get_discretisation(),
            "k": [1],
        }
        self.stochastics_fast = {
            "u": self.wind_speed.get_discretisation(),
            "m": self.sea_level.get_discretisation(),
        }
        self.stochastics_slow = {}

    def calculate_probability(
        self, wind_direction: float, closing_situation: int = 1, given: list = []
    ):
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
        # Probability of wind direction
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
        kanswr = 1.0 if "r" in given else self.wind_direction.get_probability()[ir]

        # Probability of a sea level and wind speed given a wind direction
        kans_um_r = self.p_mur[:, :, ir]

        # If given, calculate the conditional probabilities
        if "u" in given:
            kans_um_r[:] = ProbabilityFunctions.conditional_probability(
                kans_um_r, axis=0
            )
        if "m" in given:
            kans_um_r[:] = ProbabilityFunctions.conditional_probability(
                kans_um_r, axis=1
            )

        # Combine all probabilities
        probability = kans_um_r[:, :] * kanswr

        # Return probability
        return probability

    def __calculate_combined_probabilities(self):
        # Statistics
        m = self.sea_level
        s = self.sigma_function
        r = self.wind_direction
        u = self.wind_speed

        # Initialize an empty matrix
        self.p_mur = np.zeros((len(u), len(m), len(r)))

        # Per wind direction
        for ir in range(len(r)):
            # Calculate the probability density of the sea level given the wind direction
            pd_m = ProbabilityFunctions.probability_density(
                m.get_discretisation(), m.get_exceedance_probability()[:, ir]
            )

            # If there is correlation (sigma > 0)
            if s.correlation[ir]:
                # Calculate sigma
                sigma = Interpolate.inextrp1d(
                    x=m.get_discretisation(), xp=s.sigma_sea_level, fp=s.sigma[:, ir]
                )
                if np.min(sigma) < 0.0:
                    raise ValueError()

                # Exceedance probability of the wind speed given the sea water level epm_r[Nwind, Nswl]
                snorm = (u.k_u[:, ir][:, None] - m.epm_exp[:, ir][None, :]) / sigma[
                    None, :
                ]
                epm_r = 1 - norm.cdf(snorm)

                # Per sea level
                for im in range(len(m)):
                    pd_u = ProbabilityFunctions.probability_density(
                        u.get_discretisation(), epm_r[:, im]
                    )
                    self.p_mur[:, im, ir] = pd_u.probability * pd_m.probability[im]

            # If there is no correlation (sigma <= 0)
            else:
                pd_u = ProbabilityFunctions.probability_density(
                    u.get_discretisation(), u.get_exceedance_probability()[:, ir]
                )
                self.p_mur[:, :, ir] = (
                    pd_u.probability[:, None] * pd_m.probability[None, :]
                )
