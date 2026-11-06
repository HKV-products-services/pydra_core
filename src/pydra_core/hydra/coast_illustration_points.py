"""
The Hydra-NL illustration points for the coast (ILLU_PUNT_BEPALEN and ONDKANSEN_ILLU, REGIO 5). The method, the main
illustration point and the influence factors are described in docs/hydra_extensions.qmd.
"""

import copy
import numpy as np

from scipy.special import ndtri

from .coast_breakdown import HydraCoastBreakdown
from .illustration_points import HydraIllustrationPoint
from .library import Contour
from ..core import memory
from ..core.crest_level_tables import CrestLevelTables
from ..core.memory import RememberedCalculation

# EPSILON of ONDKANSEN_ILLU: the margin within which a cumulative sum counts as full
_EPSILON = 5.0e-7

# Probabilities below this (or above 1 minus this) have no finite normal score; ILLU_PUNT_BEPALEN uses +-HUGE(1)
_PROBABILITY_FLOOR = 5.0e-16

# FACTOR of ILLU_PUNT_BEPALEN: every contour segment in ten pieces
_FACTOR = 10

# REAL(MAXINT) of Hydra-NL; a point marked with it can never win the minimum
_INFINITE = float(np.iinfo(np.int32).max)

# Contour values at or below this are the separators between the parts of a contour
_SENTINEL = -9999.0

# The responses with an illustration point: the field of the wave height and the wave period is not covered
_RESPONSES = ("h", "hbn")


class HydraCoastIllustrationPoints(metaclass=RememberedCalculation):
    """
    The Hydra-NL illustration point per wind direction for one location at the coast.
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
        Prepare the illustration points for one location at the coast.

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
            Whether the model uncertainty of the waves adds to the distance; only for "h" and "hbn"
        response : str
            "h" (water level) or "hbn" (required crest level; needs 'profile' and 'qcrit')
        profile : Profile
            The profile for the crest level
        qcrit : float
            The critical overtopping discharge for the crest level
        """
        if response not in _RESPONSES:
            raise ValueError(f"[ERROR] Unknown response {response!r} (expected one of {_RESPONSES}).")
        HydraCoastBreakdown.check_response(response, model_uncertainty)
        if response == "hbn" and (profile is None or qcrit is None):
            raise ValueError("[ERROR] The crest level needs a profile and a critical overtopping discharge (qcrit).")
        self.statistical_uncertainty = statistical_uncertainty
        self.model_uncertainty = model_uncertainty
        self.response = response
        self._profile = profile
        self._qcrit = float(qcrit) if qcrit is not None else None

        # The contribution per direction decides the main illustration point; the breakdown has its own location, so
        # that its loading chain does not touch the loading read here
        self.breakdown = HydraCoastBreakdown(
            hrd, settings, statistical_uncertainty, model_uncertainty, response, profile, qcrit
        )

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

        self.m_stat = np.asarray(stats.sea_level.get_discretisation(), float)
        self.u_stat = np.asarray(stats.wind_speed.get_discretisation(), float)
        self.directions = np.asarray(stats.wind_direction.get_discretisation(), float)
        self._kansr = np.asarray(stats.wind_direction.get_probability(), float)
        p_mur = np.asarray(stats.p_mur, float)
        self._non_exceedance = [
            HydraCoastIllustrationPoints.non_exceedance_probabilities(p_mur, ir) for ir in range(len(self.directions))
        ]
        self._keys = {key[0]: key for key in self._loading.model}

        # (f_Hm0, f_Tm-1,0, AFSTONZ, (eps_h^2, eps_Hm0^2, eps_Tspec^2)) per combination, sorted on distance
        by_distance = HydraIllustrationPoint.uncertainty_combinations_by_distance(stats, response, model_uncertainty)
        self.combinations = [
            (float(f_hs), float(f_tspec), float(distance2), comps)
            for _dh, f_hs, f_tspec, _f_tp, distance2, comps in by_distance
        ]

        # The repaired loading field on the statistics grid per (combination, direction). It does not depend on the
        # level; for the crest level it is the expensive part (one overtopping calculation per cell).
        self._fields: dict[tuple[int, float], np.ndarray] = {}
        self._ahead_done = False

    def field(self, ic: int, r: float) -> np.ndarray:
        """
        Return the repaired loading field of an uncertainty combination and direction on the grid of the statistics.

        Parameters
        ----------
        ic : int
            The index of the uncertainty combination
        r : float
            The wind direction

        Returns
        -------
        np.ndarray
            The repaired loading field B(u, m)
        """
        key = (ic, float(r))
        if key in self._fields:
            return self._fields[key]
        f_hs, f_tspec, _distance2, _comps = self.combinations[ic]
        if self.response == "h":
            field = self._loading.model[self._keys[r]].refine("h", {"u": self.u_stat, "m": self.m_stat})
        else:
            # The model uncertainty factors are applied to the waves before the crest level, as Hydra-NL reads the
            # loading of each combination
            field = CrestLevelTables.crest_level_table(
                self._profile, self._qcrit, *self.refined_loading(r), f_hs, f_tspec
            )
        self._fields[key] = HydraCoastBreakdown.repair_coast(field)
        return self._fields[key]

    def refined_loading(self, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Refine the loading of a direction to the grid of the statistics, as Hydra-NL does for the sea.

        Parameters
        ----------
        r : float
            The wind direction

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The water level, significant wave height, spectral wave period and wave direction on (u, m)
        """
        loading_model = self._loading.model[self._keys[r]]
        grid = {"u": self.u_stat, "m": self.m_stat}
        return tuple(loading_model.refine(variable, grid) for variable in ("h", "hs", "tspec", "dir"))

    def fields_ahead(self) -> None:
        """
        Calculate the crest level fields of all uncertainty combinations and directions at once in worker processes,
        so that 'field' finds them. Only from 'MIN_POINTS_PARALLEL' points together, and not with 'memory.processes'
        1; the fields are kept in this object, not in 'core.memory'.
        """
        if self.response != "hbn" or self._ahead_done:
            return
        self._ahead_done = True
        directions = [float(r) for ir, r in enumerate(self.directions) if self._kansr[ir] > 0.0]
        combinations = [(0.0, f_hs, f_tspec) for f_hs, f_tspec, _distance2, _comps in self.combinations]
        work = [(self.refined_loading(r), combinations) for r in directions]
        n_points = sum(loading[0].size * len(combinations) for loading, _ in work)
        n_processes = memory.processes if memory.processes is not None else CrestLevelTables.default_processes()
        if not CrestLevelTables.worth_starting_workers(n_points, n_processes, "set 'memory.processes = 1'"):
            return
        tables = CrestLevelTables.calculate_in_workers(self._profile, self._qcrit, work, n_processes)
        for r, per_direction in zip(directions, tables, strict=True):
            for ic, table in enumerate(per_direction):
                self._fields[(ic, r)] = HydraCoastBreakdown.repair_coast(table)

    def illustration_points(self, level: float) -> dict:
        """
        Find the illustration point per wind direction at a level, plus the main illustration point.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The keys 'level', 'points' and 'main' (the point of the direction with the largest contribution, None if no
            contribution is positive). Every point has 'r', the design point 'm' (sea level) and 'u' (wind speed), the
            loading at it ('h', 'hs', 'tspec', 'wave_direction'), 'distance' (beta^2), 'beta', 'contribution' (the
            frequency of the direction), 'percentage', 'reason' ("zero probability" or "no contour" for a direction
            without a point), 'alpha2' and the winning combination 'f_hs' and 'f_tspec'.
        """
        level = float(level)
        freq_r = np.asarray(self.breakdown.breakdown(level)["freq_r"], float)
        total = float(freq_r.sum())
        self.fields_ahead()

        # Per direction the best point so far: (total distance^2, u, m, conditions, terms of the distance)
        best: list[tuple] = [(_INFINITE, -1.0, -1.0, None, None) for _ in self.directions]
        for ic, (f_hs, f_tspec, distance2, comps) in enumerate(self.combinations):
            for ir, r in enumerate(self.directions):
                if self._kansr[ir] <= 0.0:
                    continue
                wu, wm = Contour.isoline(self.u_stat, self.m_stat, self.field(ic, r), level)
                if len(wm) > 0 and np.min(wm) <= _SENTINEL:
                    wu, wm = Contour.merge_contours(wu, wm, self.m_stat[0], self.m_stat[-1])

                ondkans_m, ondkans_udm = self._non_exceedance[ir]
                u, m, distance, x2, y2 = HydraCoastIllustrationPoints.design_point_on_contour(
                    wu, wm, self.u_stat, self.m_stat, ondkans_m, ondkans_udm
                )
                if distance >= _INFINITE:
                    continue
                total_distance = distance + distance2
                if total_distance < best[ir][0]:
                    conditions = HydraCoastIllustrationPoints.conditions_at_point(
                        self._loading.model[self._keys[r]], u, m, f_hs, f_tspec
                    )
                    conditions.update(f_hs=float(f_hs), f_tspec=float(f_tspec))
                    best[ir] = (total_distance, u, m, conditions, (x2, y2, comps))

        points = []
        for ir, r in enumerate(self.directions):
            distance, u, m, conditions, terms = best[ir]
            found = distance < _INFINITE
            point = {
                "r": float(r),
                "u": float(u) if found else None,
                "m": float(m) if found else None,
                "distance": float(distance) if found else None,
                "beta": float(np.sqrt(distance)) if found else None,
                "contribution": float(freq_r[ir]),
                "percentage": (100.0 * float(freq_r[ir]) / total) if total > 0 else 0.0,
                "reason": None if found else ("zero probability" if self._kansr[ir] <= 0.0 else "no contour"),
                "alpha2": None if terms is None else self.alpha2(distance, terms),
            }
            point.update(
                conditions
                or {"h": None, "hs": None, "tspec": None, "wave_direction": None, "f_hs": None, "f_tspec": None}
            )
            points.append(point)

        # HOOFDIP4OVERZICHT: the largest contribution with a strict test, so the lowest direction index wins a tie
        largest, main = 0.0, None
        for ir in range(len(freq_r)):
            if float(freq_r[ir]) > largest:
                largest, main = float(freq_r[ir]), ir
        return {"level": level, "points": points, "main": points[main] if main is not None else None}

    def alpha2(self, distance2: float, terms: tuple) -> list[dict]:
        """
        Calculate the influence factors of one design point: sea level S and wind speed U, plus the model uncertainty
        terms without the water level term.

        Parameters
        ----------
        distance2 : float
            The total squared distance of the design point
        terms : tuple
            (X^2, Y^2, (eps_h^2, eps_Hm0^2, eps_Tspec^2))

        Returns
        -------
        list[dict]
            The influence factors, as 'HydraIllustrationPoint.alpha2_decomposition'
        """
        x2, y2, comps = terms
        phys = [("S", "S", float(x2)), ("U", "U", float(y2))]
        factors = HydraIllustrationPoint.alpha2_decomposition(
            phys, comps, float(distance2), self.response, self.model_uncertainty
        )
        return [f for f in factors if not (f["key"] == "h" and f["alpha2"] == 0.0)]

    @staticmethod
    def extremes(values: np.ndarray) -> np.ndarray:
        """
        Determine EXTREMEN on the interval [0, 1] per point: whether the corner values span at least 99 % of the
        probability range. Linear interpolation between those corners is then meaningless and Hydra-NL drops the point.

        Parameters
        ----------
        values : np.ndarray
            The corner values, with the corners on the last axis

        Returns
        -------
        np.ndarray
            True where the corner values span at least 0.99; False for a NaN corner
        """
        values = np.asarray(values, float)
        return (values.max(axis=-1) - values.min(axis=-1)) >= 0.99

    @staticmethod
    def normal_score(probability: np.ndarray) -> np.ndarray:
        """
        Calculate the standard normal score, with the cut-off of Hydra-NL at both tails.

        'ndtri' and not 'norm.ppf': 'norm.ppf' calls the same 'ndtri', but with the generic machinery of the
        distributions around it, which costs far more than the calculation.

        Parameters
        ----------
        probability : np.ndarray
            The non-exceedance probability

        Returns
        -------
        np.ndarray
            The standard normal score
        """
        probability = np.asarray(probability, float)
        score = np.where(probability <= _PROBABILITY_FLOOR, -_INFINITE, 0.0)
        score = np.where(1.0 - probability <= _PROBABILITY_FLOOR, _INFINITE, score)
        inside = (probability > _PROBABILITY_FLOOR) & (1.0 - probability > _PROBABILITY_FLOOR)
        score[inside] = ndtri(probability[inside])
        # A NaN probability passes none of the tests, as in the loop it came from, and gets the score of 'norm.ppf'
        score[np.isnan(probability)] = np.nan
        return score

    @staticmethod
    def non_exceedance_probabilities(p_mur: np.ndarray, ir: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the non-exceedance probabilities for the coast (ONDKANSEN_ILLU without a barrier, so KANSKERING = 1).
        Both are cumulative sums of half cell masses, which puts the probability on the middle of the cell instead of on
        its edge.

        Parameters
        ----------
        p_mur : np.ndarray
            The probability per (u, m, r)
        ir : int
            The index of the wind direction

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ONDKANSM(m), the non-exceedance probability of the sea level, and ONDKANSUDM(u, m), that of the wind given
            the sea level
        """
        p_um = np.asarray(p_mur, float)[:, :, ir]
        nu, nm = p_um.shape

        # Mass per sea level, and the conditional wind distribution within it
        p_m = p_um.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_u_given_m = np.where(p_m > 0.0, p_um / p_m, 0.0)

        # ONDKANSUDMX: cumulative half masses over the wind per sea level, scaled to 1 only if there is something to
        # scale
        ondkans_udm = np.zeros((nu, nm))
        if nu > 1:
            half = 0.5 * (p_u_given_m[:-1, :] + p_u_given_m[1:, :])
            ondkans_udm[1:, :] = np.cumsum(half, axis=0)
        last = ondkans_udm[-1, :]
        scale = (last > 0.0) & ((1.0 - last) > _EPSILON)
        ondkans_udm[:, scale] /= last[scale]

        # ONDKANSMDX: the masses normalised on the state probability first
        state_probability = float(p_m.sum())
        p_m = p_m / state_probability if state_probability > 0.0 else np.zeros_like(p_m)

        ondkans_m = np.zeros(nm)
        if nm > 2:
            ondkans_m[1] = p_m[0] + p_m[1] / 2.0
            for im in range(2, nm - 1):
                ondkans_m[im] = ondkans_m[im - 1] + (p_m[im - 1] + p_m[im]) / 2.0
            ondkans_m[nm - 1] = ondkans_m[nm - 2] + p_m[nm - 2] / 2.0 + p_m[nm - 1]
        else:
            ondkans_m[-1] = 1.0
        if ondkans_m[-1] > 1.0 - _EPSILON:
            ondkans_m /= ondkans_m[-1]

        return ondkans_m, ondkans_udm

    @staticmethod
    def upper_index(value: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Find per value the upper grid point of the interval in which it lies, as ILLU_PUNT_BEPALEN searches it: the
        first point from the second to the second-last with value <= grid[i], falling back to the last. Beyond the grid
        the first or last interval is extrapolated.

        Parameters
        ----------
        value : np.ndarray
            The values
        grid : np.ndarray
            The grid

        Returns
        -------
        np.ndarray
            The index of the upper grid point per value
        """
        grid = np.asarray(grid, float)
        return np.searchsorted(grid[1:-1], np.asarray(value, float), side="left") + 1

    @staticmethod
    def linear(x0: np.ndarray, x1: np.ndarray, y0: np.ndarray, y1: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Interpolate linearly, extrapolating beyond the interval (DINTERPOL).

        Parameters
        ----------
        x0, x1 : np.ndarray
            The coordinates of the two points
        y0, y1 : np.ndarray
            The values at the two points
        x : np.ndarray
            The coordinates to interpolate at

        Returns
        -------
        np.ndarray
            The interpolated values; y0 where x0 equals x1
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x1 == x0, y0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))

    @staticmethod
    def design_point_on_contour(
        wu: np.ndarray,
        wm: np.ndarray,
        u_stat: np.ndarray,
        m_stat: np.ndarray,
        ondkans_m: np.ndarray,
        ondkans_udm: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """
        Find the point on the contour with the smallest distance (ILLU_PUNT_BEPALEN).

        Every segment of the contour is divided into '_FACTOR' pieces, the last one with its end point; all points are
        evaluated at once, with per point the operations of the Fortran loop in the same order. The first point with the
        smallest distance wins, as in the loop.

        Parameters
        ----------
        wu : np.ndarray
            The wind speeds of the contour
        wm : np.ndarray
            The sea levels of the contour
        u_stat : np.ndarray
            The wind speeds of the statistics
        m_stat : np.ndarray
            The sea levels of the statistics
        ondkans_m : np.ndarray
            The non-exceedance probability of the sea level
        ondkans_udm : np.ndarray
            The non-exceedance probability of the wind given the sea level

        Returns
        -------
        tuple[float, float, float, float, float]
            (wind speed, sea level, squared distance, X^2, Y^2); for a contour of fewer than two points (-1, -1,
            infinite, 0, 0), as the initialisation of the Fortran. X^2 and Y^2 are the terms of the distance, for the
            influence factors.
        """
        wu = np.asarray(wu, float)
        wm = np.asarray(wm, float)
        n_points = len(wm)
        best = (-1.0, -1.0, _INFINITE, 0.0, 0.0)
        if n_points <= 1:
            return best

        # The points in the order of the loop: per segment the fractions 0 .. (FACTOR - 1) / FACTOR, the last segment
        # also fraction 1
        segment = np.repeat(np.arange(n_points - 1), _FACTOR)
        fraction = np.tile(np.arange(_FACTOR) / _FACTOR, n_points - 1)
        segment = np.append(segment, n_points - 2)
        fraction = np.append(fraction, _FACTOR / _FACTOR)
        sea_level = wm[segment] + (wm[segment + 1] - wm[segment]) * fraction
        wind_speed = wu[segment] + (wu[segment + 1] - wu[segment]) * fraction

        lin = HydraCoastIllustrationPoints.linear
        dm = HydraCoastIllustrationPoints.upper_index(sea_level, m_stat)
        x = np.where(
            HydraCoastIllustrationPoints.extremes(np.stack([ondkans_m[dm - 1], ondkans_m[dm]], axis=-1)),
            _INFINITE,
            HydraCoastIllustrationPoints.normal_score(
                lin(m_stat[dm - 1], m_stat[dm], ondkans_m[dm - 1], ondkans_m[dm], sea_level)
            ),
        )

        du = HydraCoastIllustrationPoints.upper_index(wind_speed, u_stat)
        corners = np.stack(
            [ondkans_udm[du - 1, dm - 1], ondkans_udm[du - 1, dm], ondkans_udm[du, dm - 1], ondkans_udm[du, dm]], axis=-1
        )
        lower = lin(m_stat[dm - 1], m_stat[dm], ondkans_udm[du - 1, dm - 1], ondkans_udm[du - 1, dm], sea_level)
        upper = lin(m_stat[dm - 1], m_stat[dm], ondkans_udm[du, dm - 1], ondkans_udm[du, dm], sea_level)
        y = np.where(
            HydraCoastIllustrationPoints.extremes(corners),
            _INFINITE,
            HydraCoastIllustrationPoints.normal_score(lin(u_stat[du - 1], u_stat[du], lower, upper, wind_speed)),
        )

        distance = x * x + y * y
        # Only a distance below the initial value can win, as 'distance < best' in the loop (NaN never wins)
        candidate = distance < _INFINITE
        if not candidate.any():
            return best
        i = int(np.argmin(np.where(candidate, distance, np.inf)))
        return (float(wind_speed[i]), float(sea_level[i]), float(distance[i]), float(x[i] * x[i]), float(y[i] * y[i]))

    @staticmethod
    def conditions_at_point(loading_model, u: float, m: float, f_hs: float, f_tspec: float) -> dict:
        """
        Interpolate the loading at the design point, with the waves of the winning uncertainty combination.

        Parameters
        ----------
        loading_model : LoadingModel
            The loading model of the wind direction
        u : float
            The wind speed of the design point
        m : float
            The sea level of the design point
        f_hs : float
            The factor for the significant wave height
        f_tspec : float
            The factor for the spectral wave period

        Returns
        -------
        dict
            The four columns that Hydra-NL prints next to the design point: 'h' (at the coast equal to the sea level),
            'hs', 'tspec' and 'wave_direction'
        """
        grid = {"u": np.array([float(u)]), "m": np.array([float(m)])}
        factor = {"h": 1.0, "hs": float(f_hs), "tspec": float(f_tspec), "wave_direction": 1.0}
        return {
            name: factor[name] * float(np.ravel(loading_model.refine(source, grid))[0])
            for name, source in (("h", "h"), ("hs", "hs"), ("tspec", "tspec"), ("wave_direction", "dir"))
        }
