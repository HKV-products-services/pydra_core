from abc import ABC, abstractmethod

from .stochastics.barrier.barrier import Barrier
from .stochastics.barrier.no_barrier import NoBarrier
from .stochastics.discharge import Discharge
from .stochastics.discrete_probability import DiscreteProbability
from .stochastics.lake_level import LakeLevel
from .stochastics.model_uncertainty import ModelUncertainty
from .stochastics.sea_level.sea_level import SeaLevel
from .stochastics.sigma_function import SigmaFunction

# from .stochastics.wave_shape import WaveShape
from .stochastics.wind_speed import WindSpeed
from ...settings.settings import Settings


class Statistics(ABC):
    """
    Statistics abstract base class
    Used to define statistic classes for different water systems
    """

    @abstractmethod
    def __init__(self, settings: Settings):
        # Save settings
        self.settings: Settings = settings

        # Statistics, required
        self.wind_direction: DiscreteProbability = None
        self.wind_speed: WindSpeed = None
        self.model_uncertainties: ModelUncertainty = None

        # Coast
        self.barrier: Barrier = NoBarrier()
        self.phase_surge_tide: DiscreteProbability = None
        self.sea_level: SeaLevel = None
        self.storm_surge_duration: DiscreteProbability = None

        # Lake
        self.lake_level: LakeLevel = None

        # Other
        self.sigma_function: SigmaFunction = None

        # River
        self.discharge: Discharge = None

        # Discrete, slow, fast stochatics
        # TODO: Replace with framework
        self.stochastics_discrete = {}
        self.stochastics_fast = {}
        self.stochastics_slow = {}

    @abstractmethod
    def calculate_probability(
        self, wind_direction: float, closing_situation: int = 1, given: list = []
    ):
        """
        TODO
        """
        pass

    def get_wind_directions(self) -> DiscreteProbability:
        """
        Returns the wind directions statistics object.

        Returns
        -------
        WindDirection
            The wind direction statistics
        """
        return self.wind_direction

    def get_wind_speed(self) -> WindSpeed:
        """
        Returns the wind statistics object.

        Returns
        -------
        WindSpeed
            The wind speed statistics
        """
        return self.wind_speed

    def get_model_uncertainties(self) -> ModelUncertainty:
        """
        Returns the model uncertainty statistics object.

        Returns
        -------
        ModelUncertainty
            The ModelUncertainty statistics
        """
        return self.model_uncertainties

    def get_barrier(self) -> Barrier:
        """
        Returns the barrier statistics object.

        Returns
        -------
        Barrier
            The barrier statistics object
        """
        return self.barrier

    def get_discharge(self) -> Discharge:
        """
        Returns the discharge statistics object.

        Returns
        -------
        Discharge
            The discharge statistics object
        """
        return self.discharge

    def get_lake_level(self) -> LakeLevel:
        """
        Returns the lake level statistics object.

        Returns
        -------
        LakeLevel
            The LakeLevel statistics
        """
        return self.lake_level

    def get_phase_surge_tide(self) -> DiscreteProbability:
        """
        Returns the fase surge tide statistics object.

        Returns
        -------
        FaseSurgeTide
            The FaseSurgeTide statistics
        """
        return self.phase_surge_tide

    def get_sea_level(self) -> SeaLevel:
        """
        Returns the sea level statistics object.

        Returns
        -------
        SeaLevel
            The SeaLevel statistics
        """
        return self.sea_level

    def get_sigma_function(self) -> SigmaFunction:
        """
        Returns the sigma function statistics object.

        Returns
        -------
        SigmaFunction
            The SigmaFunction statistics
        """
        return self.sigma_function

    def get_storm_surge_duration(self) -> DiscreteProbability:
        """
        Returns the storm surge duration statistics object.

        Returns
        -------
        StormSurgeDuration
            The StormSurgeDuration statistics
        """
        return self.storm_surge_duration
