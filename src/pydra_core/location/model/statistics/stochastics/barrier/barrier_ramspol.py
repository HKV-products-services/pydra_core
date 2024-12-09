import numpy as np

# from scipy.stats import norm

from .barrier import Barrier

# from ..discharge import Discharge
from ..discrete_probability import DiscreteProbability

# from ..sea_level.sea_level import SeaLevel
# from ..wind_speed import WindSpeed
# from ....loading.loading_model.loading_model import LoadingModel
from .....settings.settings import Settings


class BarrierRamspol(Barrier):
    def __init__(self, settings: Settings, wind_direction: DiscreteProbability):
        # Inherit
        super().__init__(settings, wind_direction)

        # Init standard variables
        self.k = [1, 2]
        self.nk = len(self.k)

        # Save failure probability of the Ramspol
        self.failure_probability_ramspol = settings.failure_probability_ramspol

    def calculate_closing_probability(
        self, wind_direction: float, closing_situation: int
    ) -> np.ndarray:
        """
        Calculate the failure probability of the Ramspol
        """
        if closing_situation == 1:
            return self.failure_probability_ramspol
        elif closing_situation == 2:
            return 1 - self.failure_probability_ramspol
        else:
            raise KeyError(
                f"[ERROR] Unknown closing situation: {closing_situation}, expecting 1 or 2."
            )
