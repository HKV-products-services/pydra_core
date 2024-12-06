import numpy as np

from ..statistics import Statistics
from ....location import Location


class StatisticsWaveOvertopping(Statistics):
    def __init__(
        self, location: Location, water_levels: np.ndarray, probability: np.ndarray
    ):
        """
        Init the Statistics class for the Eastern Scheldt

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit initialisation method from parent
        super().__init__(location.get_settings())

        # Init [wlev, u, xtr1, xtr2, ..., r]
        self.probability = probability

        #  Discrete, slow, fast stochatics
        self.stochastics_discrete = {
            "r": location.get_model().get_statistics().stochastics_discrete["r"]
        }
        self.stochastics_fast = {
            "wlev": water_levels,
            "u": location.get_model().get_statistics().stochastics_fast["u"],
        }
        self.stochastics_slow = location.get_model().get_statistics().stochastics_slow

        # Discretisation
        discretisation = {
            **self.stochastics_fast,
            **self.stochastics_slow,
            **self.stochastics_discrete,
        }
        self.dimensions = list(discretisation.keys())

        # Check dimensions
        for i, dim in enumerate(self.dimensions):
            if probability.shape[i] != len(discretisation[dim]):
                raise ValueError(
                    f"kansen.shape[i] {probability.shape[i]} != len(discretisatie[{dim}]) ({len(discretisation[dim])})."
                )
            setattr(self, f"n{dim}", len(discretisation[dim]))

    def calculate_probability(
        self, wind_direction: float, closing_situation: int = 1, given: list = []
    ) -> np.ndarray:
        """
        Return the probability of wlev, u, xtr1, xtr2 given the wind direction.

        Parameters
        ----------
        wind_direction : float
            The wind direction.
        closing_situation : int, optional
            The closing situation. Does not have any effect for
            StatisticsWaveOvertopping (default is 0)
        given : list, optional
            Given variables. Does not have any effect for
            StatisticsWaveOvertopping (default is 0)
        """
        # Obtain the wind direction id
        ir = list(self.stochastics_discrete["r"]).index(wind_direction)

        # Return the probability for this wind direction
        return np.take(self.probability, indices=ir, axis=self.dimensions.index("r"))
