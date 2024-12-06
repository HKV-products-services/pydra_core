import numpy as np

from .barrier import Barrier


class NoBarrier(Barrier):
    def __init__(self):
        # Inherit
        super().__init__(None, None)

        # Only 1 barrier state (= open)
        self.k = [1]
        self.nk = len(self.k)

    def calculate_closing_probability(
        self, wind_direction: float, closing_situation: int
    ) -> np.ndarray:
        """
        Irrelevant for no barrier
        """
        return None
