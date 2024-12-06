import numpy as np

from scipy.stats import norm

from ..statistics import Statistics
from ..stochastics.barrier.barrier_easternscheldt import BarrierEasternScheldt
from ..stochastics.discrete_probability import DiscreteProbability
from ..stochastics.model_uncertainty import ModelUncertainty
from ..stochastics.sea_level.sea_level_point import SeaLevelPoint
from ..stochastics.sigma_function import SigmaFunction
from ..stochastics.wind_speed import WindSpeed
from ....settings.settings import Settings
from .....common.interpolate import Interpolate
from .....common.probability import ProbabilityFunctions


class StatisticsEasternScheldt(Statistics):
    """
    Statistics class for the Eastern Scheldt
    Water systems: Eastern Scheldt
    """

    def __init__(self, settings: Settings):
        """
        Init the Statistics class for the Eastern Scheldt

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit initialisation method from parent
        super().__init__(settings)

        # Sea level
        self.sea_level = SeaLevelPoint(settings)

        # Storm surge duration
        self.storm_surge_duration = DiscreteProbability(
            settings.storm_surge_duration_probability
        )

        # Fase differences (between surge and tide)
        self.phase_surge_tide = DiscreteProbability(
            settings.phase_surge_tide_probability
        )

        # Sigma function
        self.sigma_function = SigmaFunction(settings)

        # Wind
        self.wind_direction = DiscreteProbability(settings.wind_direction_probability)
        self.wind_speed = WindSpeed(settings)
        self.wind_speed.correct_with_sigma_function(
            self.sigma_function, self.wind_direction
        )

        # Calculate P(u, m, r)
        self.__calculate_combined_probabilities()

        # Eastern Scheldt Barrier
        self.barrier = BarrierEasternScheldt(
            settings, self.wind_direction, self.wind_speed, self.sea_level
        )

        # Model uncertainty
        self.model_uncertainties = ModelUncertainty(settings)

        # Discrete, slow, fast stochatics
        self.stochastics_discrete = {
            "r": self.wind_direction.get_discretisation(),
            "k": [1, 2, 3, 4, 5, 6, 7, 8],
        }
        self.stochastics_fast = {
            "u": self.wind_speed.get_discretisation(),
            "m": self.sea_level.get_discretisation(),
            "d": self.storm_surge_duration.get_discretisation(),
            "p": self.phase_surge_tide.get_discretisation(),
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
        pwr = 1.0 if "r" in given else self.wind_direction.get_probability()[ir]

        # Probability of closing given a wind direction
        p_k = (
            1.0
            if "k" in given
            else self.barrier.calculate_closing_probability(
                wind_direction, closing_situation
            )
        )

        # Probability of a sea level and wind speed given a wind direction
        p_um_r = self.p_mur[:, :, ir]

        # If given, calculate the conditional probabilities
        if "u" in given:
            p_um_r[:] = ProbabilityFunctions.conditional_probability(p_um_r, axis=0)
        if "m" in given:
            p_um_r[:] = ProbabilityFunctions.conditional_probability(p_um_r, axis=1)

        # Combine all probabilities
        probability = (
            p_um_r[:, :, None, None]
            * self.storm_surge_duration.get_probability()[None, None, :, None]
            * self.phase_surge_tide.get_probability()[None, None, None, :]
            * p_k
            * pwr
        )

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
                snorm = (
                    self.wind_speed.k_u[:, ir][:, None] - m.epm_exp[:, ir][None, :]
                ) / sigma[None, :]
                ovkansen = 1 - norm.cdf(snorm)

                # Per sea level
                for im in range(len(m)):
                    pd_u = ProbabilityFunctions.probability_density(
                        u.get_discretisation(), ovkansen[:, im]
                    )
                    self.p_mur[:, im, ir] = pd_u.probability * pd_m.probability[im]

            # If there is no correlation (sigma <= 0)
            else:
                pd_u = ProbabilityFunctions.probability_density(
                    u.get_discretisation(), u.get_exceedance_probability()[:, ir]
                )
                self.p_mur[:, :, ir] = (
                    pd_m.probability[None, :] * pd_u.probability[:, None]
                )
