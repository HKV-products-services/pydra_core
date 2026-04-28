import numpy as np

from .model_base import ModelBase
from .loading.other_systems.loading_overtopping import LoadingOvertopping
from .statistics.other_systems.statistics_overtopping import StatisticsOvertopping
from ..location import Location


class ModelOvertopping(ModelBase):
    def __init__(self, location: Location, water_levels: np.ndarray, probability: np.ndarray):
        """
        Overtopping model
        """
        # Inherit
        super().__init__(location.get_settings())

        # Init statistics
        self.statistics = StatisticsOvertopping(location, water_levels, probability)

        # Init loading
        self.loading = LoadingOvertopping(location, water_levels)
