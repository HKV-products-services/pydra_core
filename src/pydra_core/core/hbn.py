import numpy as np

from datetime import datetime
from scipy.stats import norm

from .calculation import Calculation
from .datamodels.frequency_line import FrequencyLine
from ..common.enum import Breakwater
from ..common.probability import ProbabilityFunctions
from ..location.location import Location
from ..location.model.loading.other_systems.loading_overtopping import LoadingOvertopping
from ..location.model.model_overtopping import ModelOvertopping
from ..location.model.statistics.water_systems.statistics_coast import StatisticsCoast
from ..location.model.statistics.water_systems.statistics_eastern_scheldt import StatisticsEasternScheldt


class HBN(Calculation):
    """
    Calculate the HBN for a location
    """

    def __init__(self, q_overtopping: float, model_uncertainty: bool = True):
        """
        The __init__ method initializes an instance of the ExceedanceFrequencyLine class. It takes in several parameters to configure the calculation of the frequency line.

        Parameters
        ----------
        q_overtopping : float
            The average overtopping discharge.
        model_uncertainty : bool (optional)
            Enable or disable the use of model uncertainties when calculating the frequency line. Default is True.
        """
        # Inherit
        super().__init__()

        # Save settings
        self.set_overtopping_discharge(q_overtopping)
        self.use_model_uncertainty(model_uncertainty)

        # Optional settings, can be adjusted by user outside of __init__
        self.set_range(None, None)
        self.set_step_size(0.1)
        self.set_verbose(True)

    def calculate_location(self, location: Location) -> FrequencyLine:
        """
        Calculate the exceedance probability of the variable at a given set of levels.

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
            print(f"[WARNING] Profile is not assiged to location '{location.get_settings().location}'. Skipping this calculation.")
            return None

        # Obtain profile and validate
        profile = location.get_profile()
        if not profile.validate_profile():
            print(f"[WARNING] Something is wrong with the assigned profile of location '{location.get_settings().location}'. Skipping this calculation.")
            return None

        # Obtain location object
        settings = location.get_settings()
        model = location.get_model()
        prof = location.get_profile()
        statistics = model.get_statistics()
        loading = model.get_loading()
        slow_stochastics = list(statistics.stochastics_slow.keys())

        # Coast: the crest level on the statistics grid (u, m), as in Hydra-NL Zout
        if isinstance(statistics, StatisticsCoast):
            return self.__calculate_coast(location)

        # If no levels are defined, derive them based on the water level in the database
        lower_h, upper_h = loading.get_quantile_range("h", 0.0, 1.0, 3)
        _, upper_hs = loading.get_quantile_range("hs", 0.0, 1.0, 3)
        lower = np.floor((lower_h - upper_hs) * 10) / 10 if self.lower_bound is None else self.lower_bound
        upper_hbn = prof.calculate_crest_level(self.q_overtopping, upper_h, upper_hs, np.sqrt(2 * np.pi * upper_hs / (9.81 * 0.05)), prof.dike_orientation)
        upper = upper_hbn if self.upper_bound is None else self.upper_bound
        levels = np.arange(lower, upper + 0.5 * self.step_size, self.step_size)

        # Calculate the boundaries
        h_boundaries = ProbabilityFunctions.calculate_boundaries(levels)

        # Model uncertainty of the water level. As in Hydra-NL:
        # - Sea (Zout, coast and Eastern Scheldt): the water level is shifted and the waves are those at the shifted water level.
        #   This is done by integrating the model uncertainty in P(H=h, U=u, R=r), the waves are derived at h.
        # - Other water systems (Zoet): the water level of an event is shifted with dh, the waves remain those of the event.
        #   The uncertainty applies to all blocks of a slow wave (ONZ_PAFH = 100) and dh depends on the closing situation.
        monz = statistics.get_model_uncertainties()
        sea = isinstance(statistics, (StatisticsCoast, StatisticsEasternScheldt))
        groups, p_dh = self.__water_level_uncertainty(loading, monz, self.model_uncertainty and not sea)

        # P(H=h, U=u, R=r | slow) per group of closing situations with the same dh
        split = ["u"] + slow_stochastics + ["r"] + (["k"] if len(groups) > 1 else [])
        p_hur_tr = model.calculate_probability_loading(
            result_variable="h",
            levels=h_boundaries,
            split_input_variables=split,
            model_uncertainty=self.model_uncertainty and sea,
            given=slow_stochastics,
        )
        p_groups = []
        for dh, closing_situations in groups.items():
            p = p_hur_tr[..., [loading.k.index(k) for k in closing_situations]].sum(axis=-1) if len(groups) > 1 else p_hur_tr
            p_groups.append((np.array(dh), p[1:-1]))

        # Init wave overtopping statistics and loading
        model_hbn = ModelOvertopping(location, levels, p_groups[0][1])
        model_hbn_statistics = model_hbn.get_statistics()
        model_hbn_loading = model_hbn.get_loading()
        h_org = {key: m.h.copy() for key, m in model_hbn_loading.iter_models()}

        # With a straight slope without foreshore and breakwater, the crest level at h + dh is the crest level at h plus dh
        # (the waves are equal). The crest level is then calculated once and the levels are shifted with dh
        h_min = min(np.min(h) for h in h_org.values()) + min(np.min(dh) for dh, _ in p_groups)
        shift = self.__translation_invariant(profile, h_min)
        if len(p_groups) > 1:
            p_groups = [(dh, p, np.ascontiguousarray(np.moveaxis(p, model_hbn_statistics.dimensions.index("r"), 0))) for dh, p in p_groups]
        else:
            p_groups = [(p_groups[0][0], model_hbn_statistics.probability, model_hbn_statistics.probability_r)]

        # Iterate over the wave conditions model uncertainty
        # ASSUME: model uncertainties for wave height/period do not differ given the state of the barrier
        iterator = np.array(list(monz.iterate_model_uncertainty_wave_conditions(wave_period="tspec")))
        iterator = iterator if self.model_uncertainty else np.array([[1.0, 1.0, 1.0]])
        exceedance_probability = np.zeros(len(levels))
        for n_id, (factor_hs, factor_tspec, p_occ) in enumerate(iterator):
            # Info
            if self.verbose and self.model_uncertainty:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]: Model uncertainties {n_id + 1}/{len(iterator)} (fhs = {round(factor_hs, 3)}; ftspec = {round(factor_tspec, 3)}; p = {round(p_occ, 3)})")

            # Calculate and repair the height of the HBNs (once, if they shift with the water level)
            if shift:
                model_hbn_loading.calculate_hbn(profile, self.q_overtopping, factor_hs, factor_tspec)
                model_hbn_loading.repair_loadingmodels("hbn")

            # Every value of the water level uncertainty applies to all blocks of a slow wave
            for i, _p_dh in enumerate(p_dh):
                p_hbn_slow = 0.0
                for dh, p, p_r in p_groups:
                    # Shift the levels (P(HBN + dh in bin) = P(HBN in bin - dh)), or the water level and calculate the HBNs
                    if shift:
                        hbn_levels = levels[1:] - dh[i]
                    else:
                        hbn_levels = levels[1:]
                        for key, m in model_hbn_loading.iter_models():
                            m.h = h_org[key] + dh[i]
                        model_hbn_loading.calculate_hbn(profile, self.q_overtopping, factor_hs, factor_tspec)
                        model_hbn_loading.repair_loadingmodels("hbn")

                    # Calculate the probability of a HBN
                    model_hbn_statistics.probability, model_hbn_statistics.probability_r = p, p_r
                    p_hbn_slow = p_hbn_slow + model_hbn.calculate_probability_loading(
                        result_variable="hbn",
                        levels=hbn_levels,
                        model_uncertainty=False,
                        split_input_variables=slow_stochastics,
                        given=slow_stochastics,
                    )

                # Translate to exceedance probabilities P(HBN>hbn | Q=q)
                ep_hbn_slow = np.cumsum(p_hbn_slow[::-1], axis=0)[-1::-1]

                # Process slow stochastics (they are always at the last axes of the matrix)
                if len(statistics.stochastics_slow) > 0:
                    ep_hbn = model.process_slow_stochastics(ep_hbn_slow) * settings.periods_base_duration

                # If there are no slow stochastics, the excedeence probability times the amount of periods
                else:
                    ep_hbn = ep_hbn_slow * settings.periods_block_duration

                exceedance_probability += _p_dh * p_occ * ep_hbn

        # Restore the water levels
        for key, m in model_hbn_loading.iter_models():
            m.h = h_org[key]

        # Return the exceedance frequency line of the HBN
        return FrequencyLine(levels, exceedance_probability)

    def __calculate_coast(self, location: Location) -> FrequencyLine:
        """
        Calculate the frequency line of the crest level for the coast, as in Hydra-NL Zout (Berekenbelasting_Zout.f90):
        - The crest level on the wave table (u, h) per wind direction (DIJKBELASTING_ZOUT).
        - Bilinear inter- and extrapolation to the statistics grid (u, m) at the water level h(u, m) (PLAATS_BELASTINGEN).
        - Repair along u and m (REPAREREN_ZOUT), depending on the '{input_variable}_repair' flags.
        - The probability of each grid point (u, m) is assigned to its crest level, without classes for the water level.
        - The levels run up to the highest crest level.
        """
        # Obtain location objects
        settings = location.get_settings()
        profile = location.get_profile()
        model = location.get_model()
        statistics = model.get_statistics()
        loading = model.get_loading()
        monz = statistics.get_model_uncertainties()
        grid = {"u": statistics.stochastics_fast["u"], "m": statistics.stochastics_fast["m"]}

        # Wave table (u, h) per wind direction, with all water levels from the database
        table = LoadingOvertopping(location)

        # Water level and probability on the statistics grid (u, m) per loading model
        grids = []
        for (direction, closing_situation), loading_model in loading.iter_models():
            axes = [loading_model.input_variables.index("u"), loading_model.input_variables.index("m")]
            h = np.moveaxis(loading_model.refine("h", {key: grid[key] for key in loading_model.input_variables}), axes, [0, 1])
            probability = np.moveaxis(statistics.calculate_probability(direction, closing_situation), axes, [0, 1])
            repair_axes = [["u", "m"].index(var) for var in loading_model.repair_after_refine if var in ["u", "m"]]

            # Interpolation along h at h(u, m) (the same for every wave conditions model uncertainty)
            wlev = table.model[(direction, None)].wlev
            ih = np.clip(np.searchsorted(wlev, h) - 1, 0, len(wlev) - 2)
            wh = (h - wlev[ih]) / (wlev[ih + 1] - wlev[ih])
            grids.append((direction, ih, wh, probability.ravel(), repair_axes))

        # Iterate over the wave conditions model uncertainty
        iterator = np.array(list(monz.iterate_model_uncertainty_wave_conditions(wave_period="tspec")))
        iterator = iterator if self.model_uncertainty else np.array([[1.0, 1.0, 1.0]])
        results = []
        for factor_hs, factor_tspec, p_occ in iterator:
            # Crest levels on the wave table
            table.calculate_hbn(profile, self.q_overtopping, factor_hs, factor_tspec)

            hbn_all, p_all = [], []
            for direction, ih, wh, probability, repair_axes in grids:
                # Crest level on the wave table as (u, h)
                table_model = table.model[(direction, None)]
                hbn_table = np.moveaxis(table_model.hbn, [table_model.input_variables.index("u"), table_model.input_variables.index("wlev")], [0, 1])

                # Bilinear: first along u to the statistics grid, then along h at h(u, m)
                hbn_u = self.__inextrp(grid["u"], table_model.u, hbn_table)
                hbn = np.take_along_axis(hbn_u, ih, axis=1) * (1 - wh) + np.take_along_axis(hbn_u, ih + 1, axis=1) * wh

                # Repair
                for ax in repair_axes:
                    hbn = np.maximum.accumulate(hbn, axis=ax)
                hbn_all.append(hbn.ravel())
                p_all.append(probability)
            results.append((np.concatenate(hbn_all), np.concatenate(p_all), p_occ))

        # Levels up to the highest crest level (with a probability)
        lower = min(np.min(hbn[p > 0]) for hbn, p, _ in results)
        upper = max(np.max(hbn[p > 0]) for hbn, p, _ in results)
        lower = np.floor(lower * 10) / 10 if self.lower_bound is None else self.lower_bound
        upper = upper if self.upper_bound is None else self.upper_bound
        levels = np.arange(lower, upper + self.step_size, self.step_size)

        # P(HBN >= level): a crest level contributes to all levels lower than or equal to it
        exceedance_probability = np.zeros(len(levels))
        for hbn, p, p_occ in results:
            counts = np.bincount(np.searchsorted(levels, hbn, side="right"), weights=p, minlength=len(levels) + 1)
            exceedance_probability += p_occ * np.cumsum(counts[::-1])[::-1][1:]

        # Return the exceedance frequency line of the HBN
        return FrequencyLine(levels, exceedance_probability * settings.periods_block_duration)

    @staticmethod
    def __inextrp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        """
        Linear inter- and extrapolation of fp along the first axis (xp increasing) at x.
        """
        i = np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2)
        w = ((x - xp[i]) / (xp[i + 1] - xp[i]))[:, None]
        return fp[i] * (1 - w) + fp[i + 1] * w

    @staticmethod
    def __water_level_uncertainty(loading, monz, model_uncertainty: bool):
        """
        Discretised model uncertainty of the water level per closing situation, as in Hydra-NL (ONZ_WS_K): the values dh
        differ per closing situation, the probabilities and the index i are the same. Closing situations with the same dh are grouped.

        Returns
        -------
        dict, np.ndarray
            {tuple(dh): [closing situations]}, probability of each dh
        """
        groups = {}
        p_dh = np.array([1.0])
        for k in loading.k:
            dis = monz.get_model_uncertainty("h", k) if model_uncertainty else None
            if dis is not None:
                dh, edges = dis.discretise(monz.step_size["h"])
                p_dh = np.diff(norm.cdf(edges))
            else:
                dh = np.array([0.0])
            groups.setdefault(tuple(np.round(np.atleast_1d(dh), 12)), []).append(k)

        # Without uncertainty for one closing situation and with uncertainty for another, the first gets dh = 0 for all i
        n = len(p_dh)
        groups = {(dh * n if len(dh) == 1 else dh): ks for dh, ks in groups.items()}
        return groups, p_dh

    @staticmethod
    def __translation_invariant(profile, h_min: float) -> bool:
        """
        Whether the crest level shifts with the water level: a straight slope without foreshore and breakwater, and the
        water level above the toe.
        """
        no_breakwater = getattr(profile, "breakwater_type", None) in [None, Breakwater.NO_BREAKWATER]
        return (not profile.has_foreshore()) and no_breakwater and (len(profile.dike_x) == 2) and (h_min > profile.dike_y[0])

    def set_overtopping_discharge(self, q_overtopping: float):
        """
        Change the average overtopping discharge

        Parameters
        ----------
        q_overtopping : float
            The average overtopping discharge [m3/m/s]
        """
        # Cannot be smaller or equal to 0
        if q_overtopping <= 0:
            raise ValueError("[ERROR] Average overtopping discharge should be larger than 0.")

        # Save the critical overtopping
        self.q_overtopping = q_overtopping

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
        self.lower_bound = None
        self.upper_bound = None

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

    def set_verbose(self, verbose: bool):
        """
        Show info during calculation.

        Parameters
        ----------
        verbose : bool
            Show info during calculation.
        """
        self.verbose = verbose
