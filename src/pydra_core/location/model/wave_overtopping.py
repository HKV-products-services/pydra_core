import numpy as np

from .base_model import BaseModel
from .loading.other_systems.loading_wave_overtopping import LoadingWaveOvertopping
from .statistics.other_systems.statistics_wave_overtopping import (
    StatisticsWaveOvertopping,
)
from ..location import Location


class WaveOvertopping(BaseModel):
    def __init__(
        self, location: Location, water_levels: np.ndarray, probability: np.ndarray
    ):
        """
        Wave Overtopping model
        """
        # Inherit
        super().__init__(location.get_settings())

        # Init statistics
        self.statistics = StatisticsWaveOvertopping(location, water_levels, probability)

        # Init loading
        self.loading = LoadingWaveOvertopping(location, water_levels)
