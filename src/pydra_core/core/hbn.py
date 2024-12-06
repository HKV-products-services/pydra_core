import numpy as np

from datetime import datetime

from .calculation import Calculation
from .datamodels.frequency_line import FrequencyLine
from ..common.probability import ProbabilityFunctions
from ..location.location import Location
from ..location.model.wave_overtopping import WaveOvertopping


class HBN(Calculation):
    """
    Calculate the HBN for a location
    """

    def __init__(
        self,
        q_overtopping: float = 0.010,
        levels: list = None,
        step_size: float = 0.1,
        model_uncertainty: bool = True,
    ):
        """
        The __init__ method initializes an instance of the ExceedanceFrequencyLine class. It takes in several parameters to configure the calculation of the frequency line.

        Parameters
        ----------
        q_overtopping : float (optional)
            The critical overtopping discharge. Default is 0.01 m3/m/s (10 l/m/s)
        levels : list (optional):
            The levels at which the exceedance probability has to be calculated. If not specified, the levels will be chosen between the 1st and 99th percentile of the values in the HRDatabase.
        step_size : float (optional)
            The step size of the frequency line. Default is 0.1.
        model_uncertainty : bool (optional)
            Enable or disable the use of model uncertainties when calculating the frequency line. Default is True.
        """
        # Inherit
        super().__init__()

        # Save settings
        self.set_critical_overtopping(q_overtopping)
        self.set_levels(levels)
        self.set_step_size(step_size)
        self.use_model_uncertainty(model_uncertainty)

    def calculate_location(self, location: Location) -> float:
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
            Frequency line of the crest level
        """
        # Check if a profile is defined
        if not location.has_profile():
            print(
                f"[WARNING] Profile is not assiged to location '{location.get_settings().location}'. Skipping this calculation."
            )
            return None

        # Obtain profile and validate
        profile = location.get_profile()
        if not profile.validate_profile():
            print(
                f"[WARNING] Something is wrong with the assigned profile of location '{location.get_settings().location}'. Skipping this calculation."
            )
            return None

        # Copy the levels
        levels = self.levels

        # Obtain location object
        settings = location.get_settings()
        model = location.get_model()
        statistics = model.get_statistics()
        loading = model.get_loading()
        slow_stochastics = list(statistics.stochastics_slow.keys())

        # TODO: Beter levels bepalen
        # If no levels are defined, derive them based on the water level in the database
        if levels is None:
            _, upper_ws = loading.get_quantile_range("h", 0.01, 0.99, 3)
            _, upper_hs = loading.get_quantile_range("hs", 0.01, 0.99, 3)
            levels = np.arange(
                0, upper_ws + 4 * upper_hs + 0.5 * self.step_size, self.step_size
            )

        # Calculate the boundaries
        h_boundaries = ProbabilityFunctions.calculate_boundaries(levels)

        # P(H=h, U=u, R=r | Q=q)
        p_hur_tr = model.calculate_probability_loading(
            result_variable="h",
            levels=h_boundaries,
            split_input_variables=["u"] + slow_stochastics + ["r"],
            model_uncertainty=self.model_uncertainty,
            given=slow_stochastics,
        )

        # Init wave overtopping statistics and loading
        wave_overtopping = WaveOvertopping(location, levels, p_hur_tr[1:-1])
        wave_overtopping_statistics = wave_overtopping.get_statistics()
        wave_overtopping_loading = wave_overtopping.get_loading()

        # Create an empty array for the levels and slow stochastics
        p_hbn_slow = np.zeros(
            (len(levels),)
            + tuple(
                [
                    getattr(wave_overtopping_statistics, f"n{x}")
                    for x in slow_stochastics
                ]
            )
        )

        # Iterate over the wave conditions model uncertainty
        # ASSUME: model uncertainties for wave height/period do not differ given the state of the barrier
        iterator = np.array(
            list(
                statistics.model_uncertainties.iterate_model_uncertainty_wave_conditions(
                    wave_period="tspec"
                )
            )
        )
        iterator = iterator if self.model_uncertainty else np.array([[1.0, 1.0, 1.0]])
        for n_id, (factor_hs, factor_tspec, p_occ) in enumerate(iterator):
            # Info
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}]: Model uncertainties {n_id + 1}/{len(list(iterator))} (fhs = {round(factor_hs, 3)}; ftspec = {round(factor_tspec, 3)}; p = {round(p_occ, 3)})"
            )

            # Calculate the height of the HBNs
            wave_overtopping_loading.calculate_hbn(
                profile, self.q_overtopping, factor_hs, factor_tspec
            )

            # Repair HBN
            wave_overtopping_loading.repair_loadingmodels("hbn")

            # Calculate the probability of a HBN
            p_hbn_monz = wave_overtopping.calculate_probability_loading(
                result_variable="hbn",
                levels=levels[1:],
                model_uncertainty=False,
                split_input_variables=slow_stochastics,
                given=slow_stochastics,
            )
            p_hbn_slow += p_hbn_monz * p_occ

        # Translate to exceedance probabilities P(HBN>hbn | Q=q)
        ep_hbn_slow = np.cumsum(p_hbn_slow[::-1], axis=0)[-1::-1]

        # Process slow stochastics (they are always at the last axes of the matrix)
        if len(statistics.stochastics_slow) > 0:
            p_trapezoidal = model.process_slow_stochastics(ep_hbn_slow)
            exceedance_probability = p_trapezoidal * settings.periods_base_duration

        # If there are no slow stochastics, return the excedeence probability times the amount of periods
        else:
            exceedance_probability = ep_hbn_slow * settings.periods_block_duration

        # Return the exceedance frequency line of the HBN
        return FrequencyLine(levels, exceedance_probability)

    def set_critical_overtopping(self, q_overtopping: float):
        """
        Change the critical overtopping

        Parameters
        ----------
        q_overtopping : float
            The critical overtopping [m3/m/s]
        """
        # Cannot be smaller or equal to 0
        if q_overtopping <= 0:
            raise ValueError("[ERROR] Critical overtopping should be larger than 0.")

        # Save the critical overtopping
        self.q_overtopping = q_overtopping

    def set_levels(self, levels: list = None):
        """
        Change the levels.
        If levels is not defined, the frequency line is calculated based upon the 1st and 99th percentile.

        Parameters
        ----------
        levels : list, optional
            The levels at which the exceedance probability has to be calculated
        """
        self.levels = levels

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

    def use_model_uncertainty(self, model_uncertainty: bool):
        """
        Use model uncertainty when calculating a frequency line.

        Parameters
        ----------
        model_uncertainty : bool
            Enable or disable the use of model uncertainties
        """
        self.model_uncertainty = model_uncertainty
