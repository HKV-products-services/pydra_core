import numpy as np

from .model_base import ModelBase
from .loading.other_systems.loading_hbn import LoadingHBN
from .statistics.other_systems.statistics_hbn import StatisticsHBN
from ..location import Location


class ModelHBN(ModelBase):
    def __init__(self, location: Location, water_levels: np.ndarray, probability: np.ndarray):
        """
        HBN model
        """
        # Inherit
        super().__init__(location.get_settings())

        # Init statistics
        self.statistics = StatisticsHBN(location, water_levels, probability)

        # Init loading
        self.loading = LoadingHBN(location, water_levels)
