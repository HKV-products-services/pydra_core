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
        self.set_range(None, None)
        self.set_step_size(0.1)

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
        lower, upper = loading.get_quantile_range(self.result_variable, 0.0, 1.0, 3)
        lower = np.floor(lower * 10) / 10 if self.lower_bound is None else self.lower_bound
        upper = upper if self.upper_bound is None else self.upper_bound
        levels = np.arange(lower, upper + 0.5 * self.step_size, self.step_size)

        # Model uncertainty setup
        if self.model_uncertainty:
            steps = monz.step_size[self.result_variable]
            _, edges = monz.model_uncertainties[1, self.result_variable].discretise(steps)
            weights = np.diff(norm.cdf(edges))
        else:
            steps = 1
            weights = np.array([1.0])

        slow_keys = list(stats.stochastics_slow.keys())
        exp = np.zeros(len(levels))
        for ip, w in enumerate(weights):
            _model = deepcopy(model)
            _loading = _model.get_loading()
            _loading.repair_loadingmodels(self.result_variable)

            # Model uncertainty toepassen
            if self.model_uncertainty:
                for deelmodel, result in _loading.model.items():
                    unc = monz.model_uncertainties[deelmodel[1], self.result_variable]
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

        return FrequencyLine(levels, exp)

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
