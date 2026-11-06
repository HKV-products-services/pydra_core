import numpy as np

from ..statistics import Statistics
from ...probability_grid import DirectionProbabilityGrid
from ....location import Location


class StatisticsOvertopping(Statistics):
    def __init__(self, location: Location, water_levels: np.ndarray, probability: np.ndarray):
        """
        Init the Statistics class for the Eastern Scheldt

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit initialisation method from parent
        super().__init__(location.get_settings())

        #  Discrete, slow, fast stochatics
        self.stochastics_discrete = {"r": location.get_model().get_statistics().stochastics_discrete["r"]}
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
                raise ValueError(f"kansen.shape[i] {probability.shape[i]} != len(discretisatie[{dim}]) ({len(discretisation[dim])}).")
            setattr(self, f"n{dim}", len(discretisation[dim]))

        # Init [wlev, u, xtr1, xtr2, ..., r]
        self.set_probability(probability)

    def set_probability(self, probability: np.ndarray):
        """
        Set the probability of wlev, u, xtr1, xtr2, ..., r.
        A contiguous copy with the wind direction as first axis is saved once, so calculate_probability returns a view.
        The copy is read-only, so 'split_load_model' can keep the sparse form of its views. A grid per wind direction
        already holds the slices, and needs no copy.

        Parameters
        ----------
        probability : np.ndarray | DirectionProbabilityGrid
            The probability [wlev, u, xtr1, xtr2, ..., r]
        """
        self.probability = probability
        if isinstance(probability, DirectionProbabilityGrid):
            if self.dimensions.index("r") != probability.ndim - 1:
                raise ValueError("[ERROR] A probability grid per wind direction requires the wind direction as the last dimension.")
            self.probability_r = None
            return
        self.probability_r = np.ascontiguousarray(np.moveaxis(probability, self.dimensions.index("r"), 0))
        self.probability_r.flags.writeable = False

    def calculate_probability(self, wind_direction: float, closing_situation: int = 1, given: list = []) -> np.ndarray:
        """
        Return the probability of wlev, u, xtr1, xtr2 given the wind direction.

        Parameters
        ----------
        wind_direction : float
            The wind direction.
        closing_situation : int, optional
            The closing situation. Does not have any effect for StatisticsOvertopping (default is 0)
        given : list, optional
            Given variables. Does not have any effect for StatisticsOvertopping (default is 0)
        """
        # Obtain the wind direction id
        ir = list(self.stochastics_discrete["r"]).index(wind_direction)

        # Return the probability for this wind direction (a view or the kept slice, no copy)
        if isinstance(self.probability, DirectionProbabilityGrid):
            return self.probability.get_slice(ir)
        return self.probability_r[ir]
