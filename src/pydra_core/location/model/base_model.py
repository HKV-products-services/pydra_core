import numpy as np

from abc import ABC

from .loading.loading import Loading
from .statistics.statistics import Statistics
from ..settings.settings import Settings
from ...common.probability import ProbabilityFunctions


class BaseModel(ABC):
    def __init__(self, settings: Settings):
        """
        Base class
        """
        # Settings
        self.settings = settings
        self.statistics = None
        self.loading = None

    def calculate_probability_loading(
        self,
        result_variable: str,
        levels: np.ndarray,
        model_uncertainty: bool = True,
        split_input_variables: list = [],
        given: list = [],
    ):
        """
        Determine the probability that the load (water level, hbn, qov, etc.) falls within a range.

        The probabilities are divided among load levels and any 'split variables'.

        It is possible to specify a variable as 'given'. The probability is then determined conditionally on
        this variable. This is often the slow stochastic variable, so that the waveshape can be processed later.

        The model uncertainty in the water level can be processed. It is then integrated. Note that
        independence between different block durations is assumed.

        Parameters
        ----------
        result_variable: str
            Load variable to be split.
        levels: np.ndarray
            Array with values of the load for which the probability is determined.
        model_uncertainty: bool
            Whether to integrate model uncertainty. Only applicable for water level 'h'.
        split_input_variables: list
            Variables to be split, in addition to the load variable.
        given: list
            List of variables on which the probability is conditionally determined.
        """
        # Statistics
        loading = self.get_loading()
        statistics = self.get_statistics()

        # Init the array
        shp = [len(levels) + 1]
        for var in split_input_variables:
            if var in list(statistics.stochastics_discrete.keys()):
                shp.extend([len(statistics.stochastics_discrete[var])])
            elif var in list(statistics.stochastics_fast.keys()):
                shp.extend([len(statistics.stochastics_fast[var])])
            elif var in list(statistics.stochastics_slow.keys()):
                shp.extend([len(statistics.stochastics_slow[var])])
        probability = np.zeros(shp)

        # Seperate the discrete stochastics from the continuous stochastics (slow and fast)
        contvars = [
            var
            for var in split_input_variables
            if var not in statistics.stochastics_discrete
        ]

        # Loop through all loading models (per r, k)
        for (direction, closing_situation), _ in loading.iter_models():
            # Get the IDs of the wind direction and closing situation
            ir = loading.r.index(direction)
            ik = loading.k.index(closing_situation)

            # Calculate the probability of the loading
            probability_loading = self.split_load_model(
                direction=direction,
                closing_situation=closing_situation,
                result_variable=result_variable,
                levels=levels,
                split_input_variables=contvars,
                given=given,
            )

            # If the combination of r, k is non-existent
            if np.sum(probability_loading) == 0.0:
                continue

            # Process model uncertainty
            if model_uncertainty:
                # Check
                if result_variable not in ["h", "hs", "tspec", "tp"]:
                    raise NotImplementedError(
                        "[ERROR] Processing model uncertainties is only possible for database result variables."
                    )

                # Bepaal grenzen
                boundaries = ProbabilityFunctions.calculate_boundaries(levels)

                # Verwerk de onzekerheid gegeven de sluitsituatie
                probability_loading = (
                    statistics.get_model_uncertainties().process_model_uncertainty(
                        closing_situation=closing_situation,
                        result_variable=result_variable,
                        levels=boundaries,
                        exceedance_probability=probability_loading,
                        haxis=0,
                    )
                )

            # Create an index to accumulate the probabilities.
            # If the discrete variables are included in the splitting, assign a specific position here.
            # Otherwise, they are added to the rest of the probabilities.
            idx = [slice(None)] * len(shp)
            if "r" in split_input_variables:
                idx[split_input_variables.index("r") + 1] = ir
            if "k" in split_input_variables:
                idx[split_input_variables.index("k") + 1] = ik
            probability[tuple(idx)] += probability_loading

        return probability

    def split_load_model(
        self,
        direction: float,
        closing_situation: int,
        result_variable: str,
        levels: np.ndarray,
        split_input_variables: list = [],
        given: list = [],
    ) -> np.ndarray:
        """
        Integrates the loads over the given levels and variables to split over.

        The probabilities are derived from the statistics based on the input variables in the load model.

        For the load type, any of the result variables can be chosen.

        P(H in [h1, h2], split_vars | given)

        Parameters
        ----------
        direction : float
        closing_situation : int
        result_variable : str
            Result variable
        levels : np.ndarray
            Levels for splitting the load.
        split_input_variables : list
            Variables to split over in addition to the load variable (default : []).
        given : list
            Conditional variables P(X|given) (default : []).

        Returns
        -------
        np.ndarray
            Probability per load variable and splitting variable.
        """
        # Statistics
        loading = self.get_loading()
        statistics = self.get_statistics()

        # Make a copy of the loading model for (r, k) to prevent overwriting the one assigned to this object
        loading_model = loading.model[(direction, closing_situation)]

        # Refine the loading onto the (finer) grid used in the statistics
        comb_fast_slow = {**statistics.stochastics_fast, **statistics.stochastics_slow}
        comb_fast_slow = {
            key: comb_fast_slow[key] for key in loading_model.input_variables
        }
        refined_load = loading_model.refine(result_variable, comb_fast_slow)

        # Calculate the probability of these loading combinations
        loading_probability = statistics.calculate_probability(
            loading_model.direction, loading_model.closing_situation, given=given
        )

        # Check the dimension from the loading and probability arrays
        assert refined_load.shape == tuple(
            loading_probability.shape[: refined_load.ndim]
        )
        if refined_load.size != loading_probability.size:
            # When the dimensions of the interpolated load (refined_load) and the probabilities do not match,
            # this must be due to an additional dimension in the probabilities where the load is equal.
            # The assumption is that this variable is dependent on the given conditions.
            loading_probability = loading_probability.reshape(
                (refined_load.size,) + loading_probability.shape[refined_load.ndim :]
            )
            extra_kansvar = given[:]

        else:
            loading_probability = loading_probability.ravel()
            extra_kansvar = []

        # Determine the dimensions of the variables to split over
        var_sizes = [
            len({**statistics.stochastics_fast, **statistics.stochastics_slow}[var])
            for var in split_input_variables
        ]

        # Create an array to allocate probabilities, this is an array for the load variable,
        # plus the determined dimensions
        probability = np.zeros((len(levels) + 1, *var_sizes))

        # Split the load into level classes
        digitized = np.digitize(refined_load.ravel(), levels)

        # Create an array with ones, which will be used to place the probability
        # in the correct position in the array by multiplying [1, 2, 3, ...] along the relevant axis,
        # and then raveling (flattening into 1D)
        ones = np.ones(refined_load.shape, dtype=digitized.dtype)
        base_shape = [1] * ones.ndim

        # Create a list of index arrays, with the load class as the first element
        idxs = [digitized]

        # Loop through all other variables
        for var_size, var in zip(var_sizes, split_input_variables):
            # Check if the variable exists
            if (var not in list(comb_fast_slow)) and (var not in extra_kansvar):
                raise ValueError(
                    f"'{var}' is not in list ({', '.join(list(comb_fast_slow))})."
                )
            if var in extra_kansvar:
                continue

            # Create the dimensions for assigning the arange
            shape = base_shape[:]
            shape[list(comb_fast_slow).index(var)] = -1

            # Determine the splitting positions, reshape and add to the list
            idxs.append((ones * np.arange(var_size).reshape(tuple(shape))).ravel())

        # Sum all probabilities into the relevant bin (based on the levels)
        if len(extra_kansvar) > 0:
            for i, idx in enumerate(zip(*idxs)):
                probability[idx] += loading_probability[i]
        else:
            np.add.at(probability, tuple(idxs), loading_probability)

        # Return the probabilities
        return probability

    def get_statistics(self) -> Statistics:
        """
        Return the Statistics object

        Returns
        -------
        Statistics
            The Statistics object for this Location
        """
        return self.statistics

    def get_loading(self) -> Loading:
        """
        Return the Loading object

        Returns
        -------
        Loading
            The Loading object for this Location
        """
        return self.loading
