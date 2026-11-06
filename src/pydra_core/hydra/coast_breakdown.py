"""
The Hydra-NL breakdown and percentiles for the coast (BEREKEN_FAALKANSZEE, REGIO 5): the Wadden Sea, the Holland coast
and the Western Scheldt. The method, the repair and the model uncertainty are described in docs/hydra_extensions.qmd.
"""

import copy
import numpy as np

from .library import Contour, HydraInterpolate, Percentiles, Scaling
from ..core.memory import RememberedCalculation
from ..location.model.statistics.water_systems.statistics_coast import StatisticsCoast

# ISEQUAL ('isequal.f90'): |x - y| <= 1e-6 * |(x + y) / 2|, purely relative
_EPS = 1.0e-6

# The step with which pydra's 'LoadingModel.repair' raises the running maximum
_REPAIR_EPS = 1.0e-6

# Contour values at or below this are the separators between the parts of a contour
_SENTINEL = -9999.0

# Responses for which the outer loop over the model uncertainty is covered
_MODEL_UNCERTAINTY_RESPONSES = ("h", "hbn")

_RESPONSES = ("h", "hs", "tspec", "hbn")


class HydraCoastBreakdown(metaclass=RememberedCalculation):
    """
    The Hydra-NL breakdown of the exceedance frequency over sea level, wind speed and wind direction, for one location
    at the coast.
    """

    def __init__(
        self,
        hrd,
        settings,
        statistical_uncertainty: bool = True,
        model_uncertainty: bool = True,
        response: str = "h",
        profile=None,
        qcrit: float | None = None,
    ):
        """
        Prepare the breakdown for one location at the coast.

        Parameters
        ----------
        hrd : HRDatabase
            The database of the location
        settings : Settings
            The settings of the location, as the caller wants to calculate with them, including 'm_step'. They are copied,
            not changed.
        statistical_uncertainty : bool
            Whether the statistics include the statistical uncertainty
        model_uncertainty : bool
            Whether the model uncertainty of the waves is integrated out; only for "h" and "hbn"
        response : str
            "h" (water level), "hs" (significant wave height), "tspec" (spectral wave period) or "hbn" (required crest
            level; needs 'profile' and 'qcrit')
        profile : Profile
            The profile for the crest level
        qcrit : float
            The critical overtopping discharge for the crest level
        """
        HydraCoastBreakdown.check_response(response, model_uncertainty)
        if response == "hbn" and (profile is None or qcrit is None):
            raise ValueError("[ERROR] The crest level needs a profile and a critical overtopping discharge (qcrit).")
        self.statistical_uncertainty = statistical_uncertainty
        self.model_uncertainty = model_uncertainty
        self.response = response
        self._profile = profile
        self._qcrit = float(qcrit) if qcrit is not None else None

        settings = copy.deepcopy(settings)
        if not statistical_uncertainty:
            settings.include_statistical_uncertainty(False)
        self.location = settings.location
        location = hrd.create_location(settings)
        if profile is not None:
            location.set_profile(profile)
        model = location.get_model()
        stats = model.get_statistics()
        HydraCoastBreakdown.check_coast(stats, self.location)
        self._loading = model.get_loading()
        self.combinations = HydraCoastBreakdown.coast_uncertainty_combinations(stats, response, model_uncertainty)

        self.m_stat = np.asarray(stats.sea_level.get_discretisation(), float)
        self.u_stat = np.asarray(stats.wind_speed.get_discretisation(), float)
        self.directions = np.asarray(stats.wind_direction.get_discretisation(), float)
        self._kansr = np.asarray(stats.wind_direction.get_probability(), float)
        self._ovkansm = np.asarray(stats.sea_level.get_exceedance_probability(), float)
        self._p_mur = np.asarray(stats.p_mur, float)
        self._umax = float(settings.u_max)
        self._periods = float(settings.periods_block_duration)
        self._keys = {key[0]: key for key in self._loading.model}

        # The repaired loading field per uncertainty combination and direction. It does not depend on the level, so
        # it is calculated once, at the first breakdown.
        self._fields: list[dict[float, tuple]] | None = None

    def repaired_fields(self) -> list[dict[float, tuple]]:
        """
        Return the repaired loading field per uncertainty combination and direction, calculated at the first call.

        Returns
        -------
        list[dict[float, tuple]]
            Per combination and per direction (u, m, repaired field)
        """
        if self._fields is not None:
            return self._fields
        self._fields = []
        for f_hs, f_tspec, _probability in self.combinations:
            # The loading per combination: the crest level on the raw waves of that combination, then the repair. For
            # the other responses the loading is already in the model.
            if self.response == "hbn":
                for _, loading_model in self._loading.iter_models():
                    loading_model.calculate_hbn(self._profile, self._qcrit, f_hs, f_tspec)
            fields = {}
            for ir, r in enumerate(self.directions):
                if self._kansr[ir] <= 0.0:
                    continue
                u, m, field = HydraCoastBreakdown.loading_field(self._loading.model[self._keys[r]], self.response)
                fields[float(r)] = (u, m, HydraCoastBreakdown.repair_coast(field))
            self._fields.append(fields)
        return self._fields

    def breakdown(self, level: float) -> dict:
        """
        Break the exceedance frequency of a level down over sea level, wind speed and wind direction.

        The frequencies are the raw ones (times the number of periods per block, as UITSPLITSENBLOKKEN fills them),
        without SCHAALUITSPLITS: that scaling is uniform per level and leaves percentiles and percentages unchanged.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The keys 'sea_level', 'wind_speed', 'directions' (the grids), 'freq_mr' and 'freq_ur' (per grid value and
            direction), 'freq_r' (per direction), 'frequency', 'level', 'route' (per direction "pure", "contour",
            "mixed" if the uncertainty combinations take different routes, or "zero probability") and 'n_combinations'
        """
        level = float(level)
        n_m, n_u, n_r = len(self.m_stat), len(self.u_stat), len(self.directions)
        freq_ur = np.zeros((n_u, n_r))
        freq_mr = np.zeros((n_m, n_r))
        freq_r = np.zeros(n_r)
        routes: list[set[str]] = [set() for _ in range(n_r)]

        for (_f_hs, _f_tspec, probability), fields in zip(self.combinations, self.repaired_fields(), strict=True):
            for ir, r in enumerate(self.directions):
                if self._kansr[ir] <= 0.0:
                    routes[ir].add("zero probability")
                    continue
                u_loading, m_loading, field = fields[float(r)]
                wu, wm = Contour.isoline(u_loading, m_loading, field, level)
                if len(wm) > 0 and np.min(wm) <= _SENTINEL:
                    wu, wm = Contour.merge_contours(wu, wm, m_loading[0], m_loading[-1])

                pure, cm, ukarak = False, None, None
                if len(wm) >= 2:
                    if HydraCoastBreakdown.is_equal_after_repair(float(np.min(wm)), float(np.max(wm)), len(u_loading)):
                        pure, cm = True, float(np.max(wm))
                    else:
                        ukarak = HydraCoastBreakdown.refine_contour(wu, wm, self._umax, self.m_stat)
                if not pure and ukarak is None:
                    # The corner point rule of Hydra-NL for NP <= 1 or a contour that is constant in both directions
                    ukarak = np.full(n_m, self._umax if field[0, 0] < level else 0.0)

                if pure:
                    ovk_m, ovk_u, contour_probability = HydraCoastBreakdown.probabilities_pure_sea(
                        cm, ir, self.m_stat, self._ovkansm, self._p_mur, self._kansr
                    )
                    routes[ir].add("pure")
                else:
                    ovk_m, ovk_u, contour_probability = HydraCoastBreakdown.probabilities_contour_sea(
                        ukarak, ir, self.u_stat, self._p_mur, self._kansr
                    )
                    routes[ir].add("contour")

                freq_mr[:, ir] += probability * ovk_m
                freq_ur[:, ir] += probability * ovk_u
                freq_r[ir] += probability * contour_probability

        n = self._periods
        return {
            "sea_level": self.m_stat,
            "wind_speed": self.u_stat,
            "directions": self.directions,
            "freq_mr": n * freq_mr,
            "freq_ur": n * freq_ur,
            "freq_r": n * freq_r,
            "frequency": n * float(freq_r.sum()),
            "level": level,
            "route": [s.pop() if len(s) == 1 else ("mixed" if s else "zero probability") for s in routes],
            "n_combinations": len(self.combinations),
        }

    def percentiles(self, level: float) -> dict:
        """
        Calculate the table that Hydra-NL prints under "Bijdragen en percentielen": the percentiles of sea level and
        wind speed (PERCENTIELENS) and the contribution per direction in percent.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The keys 'pct', 'level', 'frequency', 'directions' (per direction 'r' and 'percentage'), 'sea_level' and
            'wind_speed' (None if there is no mass)
        """
        return HydraCoastBreakdown.breakdown_percentiles(self.breakdown(level))

    @staticmethod
    def is_equal(a: float, b: float) -> bool:
        """
        Compare two values as ISEQUAL: |a - b| <= 1e-6 * |(a + b) / 2|.

        Parameters
        ----------
        a, b : float
            The values

        Returns
        -------
        bool
            True if the values are equal
        """
        return abs(a - b) <= _EPS * abs((a + b) / 2.0)

    @staticmethod
    def is_equal_after_repair(a: float, b: float, n_u: int) -> bool:
        """
        Compare two values as ISEQUAL with the artificial slope of pydra's repair added to the margin: n_u wind speeds
        of 1e-6 each. Without this margin the test of ZUIVER_ZEEWS fails on a field that Hydra-NL has exactly vertical.

        Parameters
        ----------
        a, b : float
            The values
        n_u : int
            The number of wind speeds of the loading table

        Returns
        -------
        bool
            True if the values are equal
        """
        return abs(a - b) <= _EPS * abs((a + b) / 2.0) + n_u * _REPAIR_EPS

    @staticmethod
    def check_coast(stats, location: str) -> None:
        """
        Raise a 'NotImplementedError' if the location does not have the statistics of the coast (REGIO 5).

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        location : str
            The name of the location
        """
        if not isinstance(stats, StatisticsCoast):
            raise NotImplementedError(
                f"[ERROR] {location} does not have the statistics of the coast ('StatisticsCoast') but "
                f"{type(stats).__name__}; only the Wadden Sea, the Holland coast and the Western Scheldt are covered."
            )

    @staticmethod
    def check_response(response: str, model_uncertainty: bool) -> None:
        """
        Raise for a response that is not covered, or not covered with model uncertainty.

        Parameters
        ----------
        response : str
            The response
        model_uncertainty : bool
            Whether the model uncertainty is integrated out
        """
        if response not in _RESPONSES:
            raise ValueError(f"[ERROR] Unknown response {response!r} (expected one of {_RESPONSES}).")
        if model_uncertainty and response not in _MODEL_UNCERTAINTY_RESPONSES:
            raise NotImplementedError(
                f"[ERROR] The coast does not cover {response!r} with model uncertainty: Hydra-NL applies the wave "
                f"uncertainty for this response to the level and not to the loading field."
            )

    @staticmethod
    def repair_coast(field: np.ndarray, repair_only_u: bool = False) -> np.ndarray:
        """
        Repair the loading field B(u, m) as REPAREER_ZOUT ('repareren_Zout.f90'): a falling cell is set equal to its
        neighbour.

        Parameters
        ----------
        field : np.ndarray
            The loading field B(u, m)
        repair_only_u : bool, optional
            False (REP_ALLEEN_U = NEE, the setting of the coast) gives one coupled sweep over wind and sea level, with
            the maximum of the cell below in u and the cell below in m; True only repairs along u. Default is False.

        Returns
        -------
        np.ndarray
            The repaired loading field
        """
        b = np.array(field, float, copy=True)
        nu, nm = b.shape
        for im in range(nm):
            for iu in range(nu):
                if repair_only_u:
                    if iu == 0:
                        continue
                    mx = b[iu - 1, im]
                else:
                    if iu == 0 and im == 0:
                        continue
                    if im == 0:
                        mx = b[iu - 1, 0]
                    elif iu == 0:
                        mx = b[0, im - 1]
                    else:
                        mx = max(b[iu - 1, im], b[iu, im - 1])
                b[iu, im] = max(b[iu, im], mx)
        return b

    @staticmethod
    def coast_uncertainty_combinations(
        stats, response: str, model_uncertainty: bool
    ) -> list[tuple[float, float, float]]:
        """
        Return the combinations of model uncertainties for the coast, with the probabilities summing to 1.

        The water level uncertainty drops out: the sea level comes directly from the statistics and is not shifted
        through the loading chain. A database that does bring a water level uncertainty therefore raises.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        response : str
            The response
        model_uncertainty : bool
            False gives one combination with weight 1

        Returns
        -------
        list[tuple[float, float, float]]
            The combinations (f_Hm0, f_Tm-1,0, KANSONZ)
        """
        combinations = Scaling.uncertainty_combinations(stats, response, model_uncertainty)
        shifted = [d for d, _f, _t, _k in combinations if d != 0.0]
        if shifted:
            raise NotImplementedError(
                f"[ERROR] This coastal database has a water level uncertainty (delta_h up to {max(shifted):+.4g} m), "
                f"which the coast does not cover: the sea level comes from the statistics."
            )
        return [(f_hs, f_tspec, probability) for _d, f_hs, f_tspec, probability in combinations]

    @staticmethod
    def refine_contour(wu: np.ndarray, wm: np.ndarray, umax: float, m_stat: np.ndarray) -> np.ndarray | None:
        """
        Determine the characteristic wind speed per sea level of the statistics from the contour (VERFIJNEN_CONTOUR,
        'contourbewerking.f90'). Unlike 'Contour.flatten_contour' this has the branch for WM <= 0, which occurs at the
        coast: sea levels below NAP are part of the grid.

        Parameters
        ----------
        wu : np.ndarray
            The wind speeds of the contour
        wm : np.ndarray
            The sea levels of the contour
        umax : float
            The largest wind speed of the statistics (UMAX)
        m_stat : np.ndarray
            The sea levels of the statistics

        Returns
        -------
        np.ndarray | None
            The characteristic wind speed per sea level; None for the exit NP = 0 of Hydra-NL, a contour that is
            constant in both directions, where the caller decides with the corner point rule
        """
        wu = np.asarray(wu, float).copy()
        wm = np.asarray(wm, float).copy()
        n = len(wm)
        if n == 0:
            return None

        if HydraCoastBreakdown.is_equal(wm[0], wm[-1]):
            if HydraCoastBreakdown.is_equal(wu[0], wu[-1]):
                return None
            return np.where(np.asarray(m_stat, float) < wm[0], umax, 0.0)

        if wm[0] > wm[-1]:
            wm = wm[::-1].copy()
            wu = wu[::-1].copy()

        # Strictly increasing: a relative step above zero and an absolute step below, as in the Fortran
        for ip in range(n - 1):
            if wm[ip + 1] <= wm[ip]:
                wm[ip + 1] = wm[ip] * (1.0 + 2.0 * _EPS) if wm[ip] > 0.0 else wm[ip] + _EPS

        ukarak = HydraInterpolate.inextrp1d_interp(np.asarray(m_stat, float), wm, wu)
        return np.clip(ukarak, 0.0, umax)

    @staticmethod
    def probabilities_pure_sea(
        cm: float, ir: int, m_stat: np.ndarray, ovkansm: np.ndarray, p_mur: np.ndarray, kansr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate the probabilities of a vertical contour (KANSEN_ZUIVER_ZEE, 'watersysteem_zee.f90'): the probability
        of the contour is read from the exceedance curve, not summed.

        The subtraction of SCHALINGSWAARDE * OVKANSM(NMSTAT) is part of the Fortran and therefore of the answer,
        although the Fortran comment itself calls it unnecessary.

        Parameters
        ----------
        cm : float
            The sea level of the contour
        ir : int
            The index of the wind direction
        m_stat : np.ndarray
            The sea levels of the statistics
        ovkansm : np.ndarray
            The exceedance probability of the sea level per direction
        p_mur : np.ndarray
            The probability per (u, m, r)
        kansr : np.ndarray
            The probability per wind direction

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float]
            The contribution per sea level, per wind speed, and the probability of the contour
        """
        nm = len(m_stat)
        scale = (ovkansm[nm - 1, ir] / ovkansm[nm - 2, ir]) if ovkansm[nm - 2, ir] > 0.0 else 1.0
        contour_probability = float(
            (HydraInterpolate.interp1d_clamped(m_stat, ovkansm[:, ir], cm) - scale * ovkansm[nm - 1, ir]) * kansr[ir]
        )

        above = m_stat >= cm
        ovk_m = np.where(above, p_mur[:, :, ir].sum(axis=0), 0.0) * kansr[ir]
        ovk_u = p_mur[:, above, ir].sum(axis=1) * kansr[ir]

        total = float(ovk_m.sum())
        if total > 0.0:
            ovk_m = ovk_m * (contour_probability / total)
            ovk_u = ovk_u * (contour_probability / total)
        return ovk_m, ovk_u, contour_probability

    @staticmethod
    def probabilities_contour_sea(
        ukarak: np.ndarray, ir: int, u_stat: np.ndarray, p_mur: np.ndarray, kansr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate the probabilities of a slanted contour (KANSEN_CONTOURZEE, 'watersysteem_zee.f90'): a cell counts as
        soon as USTAT(IU) >= UKARAK(IM). Sea levels whose characteristic wind speed lies at or above the highest wind
        speed do not count at all.

        Parameters
        ----------
        ukarak : np.ndarray
            The characteristic wind speed per sea level
        ir : int
            The index of the wind direction
        u_stat : np.ndarray
            The wind speeds of the statistics
        p_mur : np.ndarray
            The probability per (u, m, r)
        kansr : np.ndarray
            The probability per wind direction

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float]
            The contribution per sea level, per wind speed, and the probability of the contour
        """
        ukarak = np.asarray(ukarak, float)
        indicator = u_stat[:, None] >= ukarak[None, :]
        indicator[:, ukarak >= u_stat[-1]] = False
        contribution = p_mur[:, :, ir] * indicator
        ovk_m = contribution.sum(axis=0) * kansr[ir]
        ovk_u = contribution.sum(axis=1) * kansr[ir]
        return ovk_m, ovk_u, float(ovk_m.sum())

    @staticmethod
    def loading_field(loading_model, response: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the loading field B(u, m) of one wind direction with its own axes: the grid of the loading table, not the
        finer grid of the statistics. Hydra-NL runs ISOLIJN on that grid too; only VERFIJNEN_CONTOUR goes to the
        statistics grid.

        Parameters
        ----------
        loading_model : LoadingModel
            The loading model of the wind direction
        response : str
            The response

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The wind speeds, the sea levels and the loading field B(u, m)
        """
        m_axis = next(v for v in loading_model.input_variables if v != "u")
        field = np.asarray(getattr(loading_model, response), float)
        if loading_model.input_variables.index("u") > loading_model.input_variables.index(m_axis):
            field = field.T
        return np.asarray(loading_model.u, float), np.asarray(getattr(loading_model, m_axis), float), np.squeeze(field)

    @staticmethod
    def breakdown_percentiles(breakdown: dict) -> dict:
        """
        Calculate the percentile table of a breakdown.

        Parameters
        ----------
        breakdown : dict
            The breakdown of 'HydraCoastBreakdown.breakdown'

        Returns
        -------
        dict
            The table, as 'percentiles'
        """
        freq_r = np.asarray(breakdown["freq_r"], float)
        total = float(freq_r.sum())
        pm = Percentiles.percentiles(
            np.asarray(breakdown["sea_level"], float), np.asarray(breakdown["freq_mr"]).sum(axis=1)
        )
        pu = Percentiles.percentiles(
            np.asarray(breakdown["wind_speed"], float), np.asarray(breakdown["freq_ur"]).sum(axis=1)
        )
        return {
            "pct": Percentiles.percentile_labels(),
            "level": breakdown["level"],
            "frequency": breakdown["frequency"],
            "directions": [
                {"r": float(r), "percentage": (100.0 * f / total) if total > 0 else 0.0}
                for r, f in zip(breakdown["directions"], freq_r, strict=True)
            ],
            "sea_level": None if pm is None else [float(v) for v in pm],
            "wind_speed": None if pu is None else [float(v) for v in pu],
        }
