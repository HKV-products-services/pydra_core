import numpy as np

from copy import deepcopy
from scipy.stats import norm

from .calculation import Calculation
from .datamodels.frequency_line import FrequencyLine
from ..location.location import Location


class ExceedanceFrequencyLine(Calculation):
    """
    Calculate a frequency line for a result variable (e.g. h (waterlevel), hs (significant wave height)) for a location
    """

    def __init__(self, result_variable: str, model_uncertainty: bool = True):
        """
        The __init__ method initializes an instance of the ExceedanceFrequencyLine class. It takes in several parameters to configure the calculation of the frequency line.

        Parameters
        ----------
        result_variable: str
            The result variable for which the frequency line will be calculated.
        model_uncertainty: bool
            Enable or disable the use of model uncertainties when calculating the frequency line. Default is True.
        """
        # Inherit
        super().__init__()

        # Save settings
        self.set_result_variable(result_variable.lower())
        self.use_model_uncertainty(model_uncertainty)

        # Optional settings, can be adjusted by user outside of __init__
        # Without a step size, the step size is 0.1 or smaller (at least 100 levels)
        self.set_range(None, None)
        self.step_size = None

    def calculate_location(self, location: Location) -> FrequencyLine:
        """
        Calculate the exceedance probability of the variable at a given set of levels.
        If the levels are not specified, they will be chosen at the 1st and 99th percentile of all values in the database.

        Parameter
        ---------
        location : Location
            The Location object

        Returns
        -------
        FrequencyLine
            Frequency line of the result variable
        """
        # Objecten uit locatie
        model = location.get_model()
        loading = model.get_loading()
        stats = model.get_statistics()
        monz = stats.get_model_uncertainties()

        # Levels bepalen
        lower, upper = loading.get_quantile_range(self.result_variable, 0.01, 0.99, 3)
        lower = np.floor(lower * 10) / 10 if self.lower_bound is None else self.lower_bound
        upper = upper if self.upper_bound is None else self.upper_bound

        # Model uncertainty setup. Without a model uncertainty for the result variable (e.g. the water level at sea),
        # the frequency line is calculated without model uncertainty
        uncs = [unc for (_, rv), unc in monz.model_uncertainties.items() if rv == self.result_variable]
        use_monz = self.model_uncertainty and len(uncs) > 0
        if use_monz:
            # The edges are standardised, so equal for each closing situation
            steps = monz.step_size[self.result_variable]
            _, edges = uncs[0].discretise(steps)
            weights = np.diff(norm.cdf(edges))

            # The upper bound includes the largest model uncertainty (added to h, multiplied with hs, tspec and tp)
            if self.upper_bound is None:
                largest = max(np.max(unc.discretise(steps)[0]) for unc in uncs)
                upper = upper + largest if self.result_variable == "h" else upper * largest
        else:
            steps = 1
            weights = np.array([1.0])

        # Calculate the frequency line. If the bounds are not given, they are extended (at most 5 times each) and the
        # frequency line is calculated again:
        # - Lower bound: if the exceedance frequency at the lowest level is smaller than 1/10 per year (e.g. coast, where
        #   the database starts at a higher sea level than the statistics).
        # - Upper bound: if the exceedance frequency at the highest level is larger than 1e-6 per year (e.g. the wave
        #   height with model uncertainty, where the factors exceed the 99% quantile of the database)
        n_lower, n_upper = 0, 0
        while True:
            levels = self.__get_levels(lower, upper)
            exp = self.__calculate_exceedance_frequency(location, levels, weights, steps, use_monz)

            # Lower the lower bound (not below 0 for other result variables than the water level)
            extend_lower = (self.lower_bound is None) and (exp[0] < 0.1) and (n_lower < 5)
            new_lower = np.floor((lower - 0.5 * (upper - lower)) * 10) / 10
            if self.result_variable != "h":
                new_lower = max(new_lower, 0.0)
            extend_lower = extend_lower and (new_lower < lower)

            # Raise the upper bound
            extend_upper = (self.upper_bound is None) and (exp[-1] > 1e-6) and (n_upper < 5)
            new_upper = np.ceil((upper + 0.5 * (upper - lower)) * 10) / 10

            if not (extend_lower or extend_upper):
                break
            if extend_lower:
                lower, n_lower = new_lower, n_lower + 1
            if extend_upper:
                upper, n_upper = new_upper, n_upper + 1

        return FrequencyLine(levels, exp, self.result_variable)

    def __get_levels(self, lower: float, upper: float) -> np.ndarray:
        """
        Levels between the lower and upper bound. Without a step size, the step size is 0.1 or smaller (at least 100 levels).
        """
        if self.step_size is None:
            step_size = min(0.1, (upper - lower) / 100) if upper > lower else 0.1
        else:
            step_size = self.step_size
        return np.arange(lower, upper + 0.5 * step_size, step_size)

    def __calculate_exceedance_frequency(self, location: Location, levels: np.ndarray, weights: np.ndarray, steps, use_monz: bool) -> np.ndarray:
        """
        Calculate the exceedance frequency at the levels, integrating the (discretised) model uncertainty.
        """
        model = location.get_model()
        stats = model.get_statistics()
        monz = stats.get_model_uncertainties()

        slow_keys = list(stats.stochastics_slow.keys())
        exp = np.zeros(len(levels))
        for ip, w in enumerate(weights):
            _model = deepcopy(model)
            _loading = _model.get_loading()
            _loading.repair_loadingmodels(self.result_variable)

            # Model uncertainty toepassen
            if use_monz:
                for deelmodel, result in _loading.model.items():
                    # Skip closing situations without model uncertainty
                    unc = monz.get_model_uncertainty(self.result_variable, deelmodel[1])
                    if unc is None:
                        continue
                    disc, _ = unc.discretise(steps)
                    data = getattr(result, self.result_variable)

                    factor = disc[ip]
                    data = data + factor if self.result_variable == "h" else data * factor

                    setattr(result, self.result_variable, data)

            # Kansmassa per bin
            p_h = _model.calculate_probability_loading(
                result_variable=self.result_variable,
                levels=levels,
                model_uncertainty=False,
                split_input_variables=slow_keys,
                given=slow_keys,
            )

            # Survival (P(H >= h))
            ep = np.cumsum(p_h[::-1], axis=0)[-2::-1]

            # Trage stochasten verwerken
            if slow_keys:
                ep = _model.process_slow_stochastics(ep)
                ep *= location.get_settings().periods_base_duration
            else:
                ep *= location.settings.periods_block_duration

            # Accumuleren
            exp += w * ep

        return exp

    def set_result_variable(self, result_variable: str):
        """
        Change the result variable for which the frequency line will be calculated.

        Parameters
        ----------
        result_variable : str
            The result variable for which the frequency line will be calculated
        """
        # Raise an error when assigning the wave direction (dir)
        if result_variable == "dir":
            raise ValueError("[ERROR] Cannot calculate a frequency line for the wave direction (dir).")

        # Save result variable
        self.result_variable = result_variable

    def use_model_uncertainty(self, model_uncertainty: bool):
        """
        Use model uncertainty when calculating a frequency line.

        Parameters
        ----------
        model_uncertainty : bool
            Enable or disable the use of model uncertainties
        """
        self.model_uncertainty = model_uncertainty

    def set_range(self, lower_bound: float = None, upper_bound: float = None):
        """
        Set the lower and upper bound of the fragility curve.

        Parameters
        ----------
        levels : list, optional
            The levels at which the exceedance probability has to be calculated
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def set_step_size(self, step_size: float):
        """
        Change the step size of the frequency line.

        Parameters
        ----------
        step_size : float
            The step size of the frequency line
        """
        # Cannot be smaller or equal to 0
        if step_size <= 0:
            raise ValueError("[ERROR] Step size should be larger than 0.")

        # Save step size
        self.step_size = step_size
