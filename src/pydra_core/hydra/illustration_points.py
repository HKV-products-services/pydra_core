"""
The Hydra-NL illustration point (ILLU_PUNTQ) for the lakes and the upper rivers. The method and the treatment of the
model uncertainty are described in docs/hydra_extensions.qmd.
"""

import copy
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtri
from scipy.stats import norm

from .library import Scaling
from ..core.memory import Memory, RememberedCalculation

# 'isoline' takes scipy's internal fast route of 'RegularGridInterpolator' ('_rgi_cython.evaluate_linear_2d',
# '_find_indices', '_values', '_grid'). If that route is not there, the public call gives the same interpolation.
try:
    from scipy.interpolate._rgi_cython import evaluate_linear_2d
except ImportError:
    evaluate_linear_2d = None

_EPS = 1e-12

# Cap for Phi^-1, as the MAXINT clip in the Fortran
_BIG = 40.0

# Hydra-NL ONZWAARDEN_VECTOR, the same as in 'DistributionUncertainty.discretise'
_KEUZERESTKANS = 0.05

# Labels of the model uncertainty terms (Hydra-NL notation) per grid key
_MODEL_LABEL = {"h": "δh", "hs": "f_Hm0", "tspec": "f_Tspec"}


@dataclass(frozen=True)
class _SlowStochastic:
    """
    The slow stochastic over which the illustration point lays the isoline.

    'key' is the name of the axis in 'stochastics_slow' and in the loading table ('LoadingModel.<key>'), "a" for the
    lakes and "q" for the upper rivers. 'stat_attr' is the 'Statistics' attribute with the marginal probabilities,
    'label' the name in the output. Both statistics return the instantaneous exceedance probability, so the
    non-exceedance probability that the Rosenblatt transformation needs is 1 minus it.
    """

    key: str
    stat_attr: str
    label: str


_SLOW: dict[str, _SlowStochastic] = {
    "a": _SlowStochastic("a", "lake_level", "M"),
    "q": _SlowStochastic("q", "discharge", "Q"),
}


class HydraIllustrationPoint(metaclass=RememberedCalculation):
    """
    The Hydra-NL illustration point for one location of a lake or an upper river.
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
        crest_level_table: Callable | None = None,
    ):
        """
        Prepare the illustration point for one location of a lake or an upper river.

        Parameters
        ----------
        hrd : HRDatabase
            The database of the location
        settings : Settings
            The settings of the location, as the caller wants to calculate with them. They are copied, not changed.
        statistical_uncertainty : bool
            Whether the statistics include the statistical uncertainty
        model_uncertainty : bool
            Whether the model uncertainty adds dimensions to the distance: the water level for the water level, the wave
            height for the wave height, and water level, wave height and wave period for the crest level
        response : str
            "h" (water level), "hs" (significant wave height) or "hbn" (required crest level; needs 'profile' and 'qcrit')
        profile : Profile
            The profile for the crest level
        qcrit : float
            The critical overtopping discharge for the crest level
        crest_level_table : Callable
            The function that calculates the crest level on a loading grid, with the signature of
            'core.crest_level_tables.crest_level_table'. By default the tables are kept in 'core.memory'; a caller that
            keeps its tables passes its own function here, and the object is then not kept either.
        """
        self.statistical_uncertainty = statistical_uncertainty
        self.model_uncertainty = model_uncertainty
        self.response = response
        self._profile = profile
        self._qcrit = float(qcrit) if qcrit is not None else None
        self._crest_level_table = (
            crest_level_table if crest_level_table is not None else Memory.remembered_crest_level_table(settings)
        )
        if response not in ("h", "hs", "hbn"):
            raise ValueError(f"[ERROR] Unknown response {response!r} (expected 'h', 'hs' or 'hbn').")
        if response == "hbn" and (profile is None or qcrit is None):
            raise ValueError("[ERROR] The crest level needs a profile and a critical overtopping discharge (qcrit).")

        settings = copy.deepcopy(settings)
        if not statistical_uncertainty:
            settings.include_statistical_uncertainty(False)
        self.location = settings.location
        model = hrd.create_location(settings).get_model()
        loading = model.get_loading()
        stats = model.get_statistics()

        slow_keys = list(stats.stochastics_slow.keys())
        if len(slow_keys) != 1:
            raise NotImplementedError(
                f"[ERROR] The illustration point covers one slow stochastic; {self.location} has {slow_keys}."
            )
        self._slow = _SLOW.get(slow_keys[0])
        if self._slow is None:
            raise NotImplementedError(f"[ERROR] The slow stochastic {slow_keys[0]!r} is not covered.")
        n_closing_situations = len({k for (_r, k) in loading.model})
        if n_closing_situations != 1:
            raise NotImplementedError(
                f"[ERROR] The illustration point covers one closing situation; {self.location} has {n_closing_situations}."
            )

        # Non-exceedance probability F_S of the slow stochastic and F_U|r of the wind speed
        slow_stat = getattr(stats, self._slow.stat_attr)
        self._s = np.asarray(slow_stat.get_discretisation(), dtype=float)
        self._FS = 1.0 - np.asarray(slow_stat.get_exceedance_probability(), dtype=float)
        self._u_stat = np.asarray(stats.wind_speed.get_discretisation(), dtype=float)
        self._FU = 1.0 - np.asarray(stats.wind_speed.get_exceedance_probability(), dtype=float)

        # 'zU' looks up the column of the wind statistics with the index of the direction in the loading table, so
        # both need the same directions. A BOI2023 database has 16 sectors; without the matching statistics the
        # default WBI2017 wind statistics have 12.
        self._directions = np.asarray(list(loading.r), dtype=float)
        if self._FU.shape[1] != len(self._directions):
            raise ValueError(
                f"[ERROR] The wind statistics have {self._FU.shape[1]} wind directions and the loading table "
                f"{len(self._directions)}. Use the statistics that belong to this database: a BOI2023 database has "
                f"16 sectors, the default WBI2017 statistics have 12."
            )

        # Per direction: the response interpolator for the water level and the wave height, the raw tables for the
        # crest level (which depends on the uncertainty factors and is therefore calculated per combination), and
        # interpolators of every loading field for the display
        response_field = {"h": "h", "hs": "hs"}.get(response)
        self._response: dict[float, RegularGridInterpolator] = {}
        self._raw: dict[float, tuple] = {}
        self._wave: dict[float, dict[str, RegularGridInterpolator]] = {}
        self._u_bounds: dict[float, tuple[float, float]] = {}
        self._s_bounds: dict[float, tuple[float, float]] = {}
        for (r, _k), lm in loading.model.items():
            u = np.asarray(lm.u, dtype=float)
            s = np.asarray(getattr(lm, self._slow.key), dtype=float)
            h = np.asarray(lm.h, dtype=float)
            self._u_bounds[float(r)] = (float(u.min()), float(u.max()))
            self._s_bounds[float(r)] = (float(s.min()), float(s.max()))
            # The wave fields are read for every response, so that the limit state figure can show the contours of
            # every loading field. Only fields on the same (u, s) grid are taken: an interpolator on another shape
            # would only fail when called, or give plausible nonsense.
            waves: dict[str, np.ndarray] = {}
            for name in ("hs", "tspec", "tp", "dir"):
                arr = getattr(lm, name, None)
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=float)
                if arr.shape == h.shape:
                    waves[name] = arr
            missing = {"hs", "tspec", "dir"} - waves.keys()
            if response == "hbn" and missing:
                raise NotImplementedError(
                    f"[ERROR] The crest level needs the wave fields {sorted(missing)}, which this loading table does not "
                    f"have on the (u, {self._slow.key}) grid."
                )
            if not missing:
                self._raw[float(r)] = (u, s, h, waves["hs"], waves["tspec"], waves["dir"])
            display = dict(waves)
            if "dir" in display:
                display["dir"] = HydraIllustrationPoint.unwrap_direction(display["dir"])
            self._wave[float(r)] = {
                v: RegularGridInterpolator((u, s), arr, method="linear", bounds_error=False, fill_value=None)
                for v, arr in (("h", h), *display.items())
            }
            if response != "hbn":
                self._response[float(r)] = RegularGridInterpolator(
                    (u, s), np.asarray(getattr(lm, response_field), dtype=float),
                    method="linear", bounds_error=False, fill_value=None,
                )

        # The combinations with their distance contribution, sorted on distance, for the search of the design point
        self._combinations = HydraIllustrationPoint.uncertainty_combinations_by_distance(
            stats, self.response, self.model_uncertainty
        )
        # The same combinations with their probability KANSONZ, for integrating out the model uncertainty
        self.combinations_with_probability = Scaling.uncertainty_combinations(
            stats, self.response, self.model_uncertainty
        )
        # Interpolation indices and distances of 'isoline' per (direction, fine grid): they only depend on the loading
        # grid of the direction, not on the uncertainty combination
        self._grid_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def slow_label(self) -> str:
        """
        Return the name of the slow stochastic in the output.

        Returns
        -------
        str
            "M" (lake level) or "Q" (discharge)
        """
        return self._slow.label

    def directions(self) -> list[float]:
        """
        Return the wind directions of the loading table.

        Returns
        -------
        list[float]
            The wind directions
        """
        return [float(r) for r in self._directions]

    def nearest(self, R: float) -> float:
        """
        Return the wind direction of the loading table nearest to a direction.

        Parameters
        ----------
        R : float
            The wind direction

        Returns
        -------
        float
            The nearest wind direction of the loading table
        """
        # Circular distance: north (0) belongs to 360, not to 22.5
        d = np.abs(((self._directions - R + 180.0) % 360.0) - 180.0)
        return float(self._directions[int(np.argmin(d))])

    def zS(self, s: np.ndarray) -> np.ndarray:
        """
        Calculate Y = Phi^-1(F_S(s)), with linear interpolation of the non-exceedance probability, as in the Fortran.

        Parameters
        ----------
        s : np.ndarray
            The values of the slow stochastic

        Returns
        -------
        np.ndarray
            The standard normal coordinates
        """
        return HydraIllustrationPoint.phi_inv(np.interp(s, self._s, self._FS))

    def zU(self, u: np.ndarray, ir: int) -> np.ndarray:
        """
        Calculate Z = Phi^-1(F_U|r(u)), with linear interpolation of the non-exceedance probability, as in the Fortran.

        Parameters
        ----------
        u : np.ndarray
            The wind speeds
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The standard normal coordinates
        """
        return HydraIllustrationPoint.phi_inv(np.interp(u, self._u_stat, self._FU[:, ir]))

    def crest_interp(self, r: float, delta: float, f_hs: float, f_ts: float) -> RegularGridInterpolator:
        """
        Build the crest level interpolator for one uncertainty combination: the required crest level on the loading
        with the water level shifted (h + delta) and the waves multiplied (Hm0 * f_hs, Tspec * f_tspec).

        Parameters
        ----------
        r : float
            The wind direction
        delta : float
            The shift of the water level
        f_hs : float
            The factor for the significant wave height
        f_ts : float
            The factor for the spectral wave period

        Returns
        -------
        RegularGridInterpolator
            The interpolator of the crest level on the (u, s) grid
        """
        u, s, h, hs, ts, dr = self._raw[r]
        # A writable copy: scipy only takes the fast route with writable values, and a kept table may be read-only
        crest = np.array(self._crest_level_table(self._profile, self._qcrit, h + delta, hs, ts, dr, f_hs, f_ts))
        return RegularGridInterpolator((u, s), crest, method="linear", bounds_error=False, fill_value=None)

    def isoline(
        self,
        interp: RegularGridInterpolator,
        r: float,
        level: float,
        n_u: int = 200,
        n_s: int = 250,
        u_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the points (u, s) on the contour response(u, s) = level: per fine u the value of s where the response
        crosses.

        The loading grid is coarse (about 10 x 6), so this refinement is ample: the design point agrees within a
        centimetre and 0.1 m/s with a 300 x 400 refinement, at half the time.

        'u_max' extends the wind speed scan beyond the loading table (the interpolator extrapolates linearly). That is
        only for the displayed isoline, so that the limit state line runs as far as the extrapolated field behind it;
        the search for the illustration point keeps the edge of the table.

        Parameters
        ----------
        interp : RegularGridInterpolator
            The interpolator of the response on the (u, s) grid
        r : float
            The wind direction
        level : float
            The level
        n_u : int, optional
            The number of wind speeds of the refinement. Default is 200.
        n_s : int, optional
            The number of values of the slow stochastic of the refinement. Default is 250.
        u_max : float | None, optional
            The largest wind speed of the scan, None for the edge of the table. Default is None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The wind speeds and the values of the slow stochastic on the contour
        """
        ulo, uhi = self._u_bounds[r]
        slo, shi = self._s_bounds[r]
        if u_max is not None:
            uhi = max(uhi, float(u_max))
        ug = np.linspace(ulo, uhi, n_u)
        sg = np.linspace(slo, shi, n_s)
        UU, SS = np.meshgrid(ug, sg, indexing="ij")
        points = np.column_stack([UU.ravel(), SS.ravel()])

        # The same steps as 'interp(points)', but with the indices and distances only once per direction: they are
        # equal for all uncertainty combinations
        values = interp._values
        fast = (
            evaluate_linear_2d is not None
            and hasattr(interp, "_find_indices")
            and values.ndim == 2
            and values.flags.writeable
            and values.dtype == np.float64
            and values.dtype.byteorder == "="
        )
        if fast:
            key = (r, n_u, n_s, uhi)
            if key not in self._grid_cache:
                self._grid_cache[key] = interp._find_indices(points.T)
            indices, distances = self._grid_cache[key]
            H = evaluate_linear_2d(values, indices, distances, interp._grid, np.empty(len(points)))
            H = H.reshape(UU.shape)
        else:
            H = interp(points).reshape(UU.shape)
        us, ss = [], []
        for j in range(n_u):
            col = H[j]
            if col[0] >= level or col[-1] <= level:
                # The response does not cross the level within the range of s at this wind speed
                continue
            ss.append(float(np.interp(level, col, sg)))
            us.append(float(ug[j]))
        return np.asarray(us), np.asarray(ss)

    def design_point(self, R: float, level: float) -> dict | None:
        """
        Find the illustration point for a wind direction at a level: ILLU_PUNTQ with the loop over the uncertainty
        combinations, stopped as soon as no combination can give a smaller distance.

        Parameters
        ----------
        R : float
            The wind direction
        level : float
            The level

        Returns
        -------
        dict | None
            The keys 'r', 'design' ({slow label: s*, "U": u*}), 'dh', 'beta', 'p', 'u' (the standard normal
            coordinates), 'alpha2', 'uncertainty' (the factors of the winning combination: 'h', 'f_hs', 'f_tspec',
            'f_tp'), for the wave height also 'f_hs', and for the crest level 'waves' (the wave conditions at the toe).
            None if the isoline is not within the loading table for any combination.
        """
        r = self.nearest(R)
        ir = int(np.argmin(np.abs(self._directions - r)))
        best = None
        for delta, f_hs, f_ts, f_tp, distance2, comps in self._combinations:
            if best is not None and distance2 >= best["distance2"]:
                # Sorted on distance: the remaining combinations cannot give a smaller distance
                break
            if self.response == "hbn":
                us, ss = self.isoline(self.crest_interp(r, delta, f_hs, f_ts), r, level)
            elif self.response == "hs":
                # The wave height times f_hs equals the isoline at level / f_hs of the unchanged wave height
                us, ss = self.isoline(self._response[r], r, level / float(f_hs))
            else:
                # The water level plus delta equals the isoline at level - delta of the unchanged water level
                us, ss = self.isoline(self._response[r], r, level - float(delta))
            if us.size == 0:
                continue
            Y = self.zS(ss)
            Z = self.zU(us, ir)
            d2 = Y * Y + Z * Z + distance2
            i0 = int(np.argmin(d2))
            if best is None or d2[i0] < best["distance2"]:
                best = {
                    "distance2": float(d2[i0]), "s": float(ss[i0]), "u": float(us[i0]), "delta": float(delta),
                    "f_hs": float(f_hs), "f_ts": float(f_ts), "f_tp": float(f_tp), "y": float(Y[i0]),
                    "z": float(Z[i0]), "comps": comps,
                }
        if best is None:
            return None
        beta = float(np.sqrt(best["distance2"]))
        label = self._slow.label
        phys = [(label, label, best["y"] * best["y"]), ("U", "U", best["z"] * best["z"])]
        out = {
            "r": r,
            "design": {label: best["s"], "U": best["u"]},
            "dh": best["delta"],
            "beta": beta,
            "p": float(norm.cdf(-beta)),
            "u": {label: best["y"], "U": best["z"]},
            "alpha2": HydraIllustrationPoint.alpha2_decomposition(
                phys, best["comps"], best["distance2"], self.response, self.model_uncertainty
            ),
        }
        if self.response == "hs":
            out["f_hs"] = best["f_hs"]
        # 'design_grid' applies these factors to every loading field it returns, so they always come with the point
        out["uncertainty"] = {"h": best["delta"], "f_hs": best["f_hs"], "f_tspec": best["f_ts"], "f_tp": best["f_tp"]}
        if self.response == "hbn":
            # The wave conditions at the design point with the uncertainty factors applied, at the toe: the loading
            # gives the waves before the foreshore, and 'calculate_crest_level' transforms them to the toe with the
            # same 'transform_wave_conditions'. Without a foreshore the waves are unchanged.
            pt = np.array([[best["u"], best["s"]]])
            w = self._wave[r]
            h0 = float(w["h"](pt)[0]) + best["delta"]
            hs0 = float(w["hs"](pt)[0]) * best["f_hs"]
            ts0 = float(w["tspec"](pt)[0]) * best["f_ts"]
            d0 = float(w["dir"](pt)[0])
            th, ths, tts, tdir = self._profile.transform_wave_conditions(h0, hs0, ts0, d0)
            out["waves"] = {"h": float(th), "Hm0": float(ths), "Tspec": float(tts), "dir": float(tdir)}
        return out

    def fields(self, r: float, dp: dict, pts: np.ndarray, shape: tuple) -> dict:
        """
        Calculate every loading field on the same (U, S) grid, with the uncertainty of the winning combination, for the
        contours of the limit state figure.

        All factors come from one combination ('dp["uncertainty"]'): h carries '+ delta' and Hs carries '* f_hs', and
        those must belong together. The wave direction gets no factor, Hydra-NL has no model uncertainty on it.

        The field of the calculated response is left out: it is the field 'response' of 'design_grid', the only source
        of the limit state line.

        The values are rounded to 0.1 mm, which is orders of magnitude finer than a pixel of the figure.

        Parameters
        ----------
        r : float
            The wind direction of the loading table
        dp : dict
            The illustration point of 'design_point'
        pts : np.ndarray
            The (u, s) points of the grid
        shape : tuple
            The shape of the grid

        Returns
        -------
        dict
            Per loading field the values as [i_s][j_u]
        """
        o = dp["uncertainty"]
        factor = {"h": None, "hs": o["f_hs"], "tspec": o["f_tspec"], "tp": o["f_tp"], "dir": 1.0}
        out: dict[str, list] = {}
        for name, w in self._wave.get(r, {}).items():
            field = w(pts).reshape(shape)
            if name == "h":
                field = field + float(o["h"])
            else:
                field = field * float(factor.get(name, 1.0))
            out[name] = np.round(field, 4).T.tolist()
        # The crest level only exists with a profile and a critical discharge
        if r in self._raw and self._profile is not None and self._qcrit is not None:
            hbn = self.crest_interp(r, o["h"], o["f_hs"], o["f_tspec"])(pts).reshape(shape)
            out["hbn"] = np.round(hbn, 4).T.tolist()
        out.pop(self.response, None)
        return out

    def design_grid(self, R: float, level: float, n_u: int = 100, n_s: int = 100) -> dict | None:
        """
        Calculate the limit state field with the design point for a wind direction at a level. The field is the
        response of the winning uncertainty combination: h(u, s) + delta* for the water level, the required crest level
        with the winning (delta*, f_Hm0*, f_Tspec*) for the crest level. The contour 'level' of this field therefore
        coincides with the isoline of the design point, and the design point lies on it.

        Parameters
        ----------
        R : float
            The wind direction
        level : float
            The level
        n_u : int, optional
            The number of wind speeds of the field. Default is 100.
        n_s : int, optional
            The number of values of the slow stochastic of the field. Default is 100.

        Returns
        -------
        dict | None
            The keys 'slow_label', 's_values', 'u_values', 'response' ([i_s][j_u]), 'fields' (the other loading
            fields), 'uncertainty', 'design' ({"M": s*, "U": u*}), 'beta', 'u' and 'alpha2'. None if there is no limit
            state within range.
        """
        dp = self.design_point(R, float(level))
        if dp is None:
            return None
        r = self.nearest(R)
        slo, shi = self._s_bounds[r]
        ulo, _uhi = self._u_bounds[r]
        u_display = float(self._u_stat.max())
        S_vals = np.linspace(slo, shi, n_s)
        U_vals = np.linspace(ulo, u_display, n_u)
        UU, SS = np.meshgrid(U_vals, S_vals, indexing="ij")
        pts = np.column_stack([UU.ravel(), SS.ravel()])
        if self.response == "hbn":
            o = dp["uncertainty"]
            H = self.crest_interp(r, o["h"], o["f_hs"], o["f_tspec"])(pts).reshape(UU.shape)
        else:
            H = self._response[r](pts).reshape(UU.shape) + float(dp.get("dh", 0.0))
        label = self._slow.label
        return {
            "slow_label": label,
            "s_values": S_vals.tolist(),
            "u_values": U_vals.tolist(),
            "response": H.T.tolist(),
            "fields": self.fields(r, dp, pts, UU.shape),
            "uncertainty": dp["uncertainty"],
            "design": {"M": dp["design"][label], "U": dp["design"]["U"]},
            "beta": dp["beta"],
            "u": {"M": dp["u"][label], "U": dp["u"]["U"]},
            "alpha2": dp.get("alpha2"),
        }

    def all_directions(self, level: float) -> list[dict]:
        """
        Find the illustration point per wind direction at a level, for the directions where there is one.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        list[dict]
            The illustration points of 'design_point'
        """
        out = []
        for r in self.directions():
            dp = self.design_point(r, float(level))
            if dp is not None:
                out.append(dp)
        return out

    @staticmethod
    def phi_inv(F: np.ndarray | float) -> np.ndarray:
        """
        Calculate Phi^-1 with clipping (Fortran: KANS <= 5e-16 gives -MAXINT, 1 - KANS <= 5e-16 gives +MAXINT).

        'ndtri' and not 'norm.ppf': 'norm.ppf' calls the same 'ndtri', but with the generic machinery of the
        distributions around it, which costs far more than the calculation for the small arrays here. Its domain checks
        do nothing, because the input is clipped to [_EPS, 1 - _EPS] first.

        Parameters
        ----------
        F : np.ndarray | float
            The non-exceedance probability

        Returns
        -------
        np.ndarray
            The standard normal coordinate
        """
        return np.clip(ndtri(np.clip(F, _EPS, 1.0 - _EPS)), -_BIG, _BIG)

    @staticmethod
    def onzwaarden_vector(mu: float, sigma: float, n: int) -> np.ndarray:
        """
        Discretise an uncertainty as Hydra-NL ONZWAARDEN_VECTOR: n grid values around mu, with the step from the
        residual probability formula. The same discretisation as 'DistributionUncertainty.discretise', but around any
        centre; it is needed for the conditionally shifted period grid given the bin of Hs.

        Parameters
        ----------
        mu : float
            The centre
        sigma : float
            The standard deviation
        n : int
            The number of grid values

        Returns
        -------
        np.ndarray
            The grid values
        """
        if n <= 1:
            return np.array([mu], dtype=float)
        residual_probability = _KEUZERESTKANS / (n**1.5)
        distance = -norm.ppf(0.5 * residual_probability) * sigma
        lower, upper = mu - distance, mu + distance
        return lower + (np.arange(1, n + 1) - 0.5) * (upper - lower) / n

    @staticmethod
    def rho(model_uncertainties, a: str, b: str) -> float:
        """
        Return the correlation coefficient between two wave uncertainties.

        Parameters
        ----------
        model_uncertainties : ModelUncertainty
            The model uncertainties of the location
        a, b : str
            The names of the two wave uncertainties

        Returns
        -------
        float
            The correlation coefficient, 0.0 if there is none
        """
        c = model_uncertainties.get_correlation(a, b)
        return float(c.rho) if c is not None else 0.0

    @staticmethod
    def unwrap_direction(field: np.ndarray) -> np.ndarray:
        """
        Express the wave direction on one continuous branch, so that 359 and 1 degrees stay neighbours instead of
        opposites.

        A loading table at wind direction 360 can have half of its nodes at 360.0 and the other half at 4.0 to 5.1. That
        is a spread of five degrees, but numerically 356: linear interpolation between two such nodes gives values
        around 182 degrees, a wave in the opposite direction of both nodes, and a contour plot draws a steep slope
        through the field.

        All values are therefore expressed relative to the circular mean, every difference in (-180, 180]. The result
        may lie outside [0, 360). If the whole field lies below zero it is shifted by 360, which does not break the
        branch; a field that really spans zero is left as it is.

        This is only used for the display interpolators ('_wave'). The raw table in '_raw', from which 'crest_interp'
        reads, is not changed.

        Parameters
        ----------
        field : np.ndarray
            The wave direction field in degrees

        Returns
        -------
        np.ndarray
            The wave direction field on one continuous branch
        """
        arr = np.asarray(field, dtype=float)
        rad = np.deg2rad(arr)
        mean = np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
        out = mean + (arr - mean + 180.0) % 360.0 - 180.0
        return out + 360.0 if out.max() < 0 else out

    @staticmethod
    def alpha2_decomposition(
        phys: list[tuple[str, str, float]],
        comps: tuple[float, float, float],
        beta2: float,
        response: str,
        model_uncertainty: bool,
    ) -> list[dict]:
        """
        Calculate the influence factors alpha^2 per stochastic from the coordinates of the design point.
        beta^2 = sum u_i^2 in standard normal space, so alpha^2_i = u_i^2 / beta^2 and the alpha^2 sum to 1.

        Parameters
        ----------
        phys : list[tuple[str, str, float]]
            The physical stochastics as (key, label, squared coordinate): [slow stochastic, wind U] in 2D, [Q, M, U] in
            3D
        comps : tuple[float, float, float]
            (eps_h^2, eps_Hm0^2, eps_Tspec^2) from 'uncertainty_combinations_by_distance'
        beta2 : float
            The squared distance of the design point
        response : str
            The response; it decides which model uncertainty terms count (h: delta_h; hs: f_Hm0; hbn: all three)
        model_uncertainty : bool
            Without model uncertainty there are no model uncertainty terms

        Returns
        -------
        list[dict]
            Per stochastic the key, label, kind ("stochastic" or "model") and alpha^2
        """
        b2 = max(float(beta2), _EPS)
        out = [{"key": k, "label": label, "kind": "stochastic", "alpha2": float(c2) / b2} for (k, label, c2) in phys]
        if model_uncertainty:
            terms = []
            if response in ("h", "hbn"):
                terms.append(("h", comps[0]))
            if response in ("hs", "hbn"):
                terms.append(("hs", comps[1]))
            if response == "hbn":
                terms.append(("tspec", comps[2]))
            out += [
                {"key": k, "label": _MODEL_LABEL[k], "kind": "model", "alpha2": float(e2) / b2} for (k, e2) in terms
            ]
        return out

    @staticmethod
    def uncertainty_combinations_by_distance(
        stats, response: str, model_uncertainty: bool, closing_situation: int = 1
    ) -> list[tuple[float, float, float, float, float, tuple[float, float, float]]]:
        """
        Return the uncertainty combinations with their distance contribution (Hydra-NL ONZWAARDEN), sorted on
        distance^2, so that the search for the design point can stop early. The weights KANSONZ of the same combinations
        are in 'Scaling.uncertainty_combinations'.

        In a water system with a storm surge barrier the model uncertainty may differ per
        barrier state. Only the values depend on it, not the distances: ONZWAARDEN_VECTOR gives delta = mu + sigma * z
        with z depending only on the number of classes, so eps = (delta - mu) / sigma does not depend on sigma. Hydra-NL
        has AFSTONZ on the single column ONZ_WS, not on the column per state ONZ_WS_K. The distances and 'comps' are
        therefore taken from state 1 for every state, and only delta_h shifts. Calculated per state, eps_h differs at
        1e-16, and for the crest level (three terms, many equal distances) that changes the sort order.

        This only holds for the water level: WS_PAR_PER_KERING and ONZ_WS_K only concern the water level, ONZ_GH and
        ONZ_GP come from one column without barrier state ('modelonzekerheden.f90'). The wave uncertainties are
        therefore taken from state 1.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        response : str
            "h", "hs" or "hbn"
        model_uncertainty : bool
            False gives one combination without distance
        closing_situation : int, optional
            The barrier state (1-based) of the water level uncertainty. Default is 1.

        Returns
        -------
        list[tuple[float, float, float, float, float, tuple[float, float, float]]]
            Per combination (delta_h, f_Hm0, f_Tm-1,0, f_Tp, distance^2, (eps_h^2, eps_Hm0^2, eps_Tspec^2)). The last
            element splits distance^2 over the three model uncertainty terms, for the influence factor alpha^2 per term;
            which terms are not zero depends on the response (h: only eps_h; hs: only eps_Hm0; hbn: all three).
        """
        if not model_uncertainty:
            return [(0.0, 1.0, 1.0, 1.0, 0.0, (0.0, 0.0, 0.0))]

        model_uncertainties = stats.get_model_uncertainties()

        # Wave height: only the uncertainty of Hs counts, with the coordinate eps = (value / mu - 1) / sigma
        if response == "hs":
            mu_hs = model_uncertainties.get_model_uncertainty("hs")
            if mu_hs is None:
                return [(0.0, 1.0, 1.0, 1.0, 0.0, (0.0, 0.0, 0.0))]
            hs_values = np.atleast_1d(mu_hs.discretise(model_uncertainties.step_size["hs"])[0]).astype(float)
            eps_hs = (hs_values / mu_hs.mu - 1.0) / mu_hs.sigma
            combinations = [
                (0.0, float(max(_EPS, f)), 1.0, 1.0, float(e * e), (0.0, float(e * e), 0.0))
                for f, e in zip(hs_values, eps_hs, strict=True)
            ]
            combinations.sort(key=lambda c: c[4])
            return combinations

        # Water level, with the coordinate eps = (delta - mu) / sigma. Not every HRD has a water level uncertainty; the
        # coastal databases have none, and then delta_h = 0, as in 'Scaling.uncertainty_combinations'.
        mu_h = model_uncertainties.get_model_uncertainty("h", closing_situation)
        if mu_h is None:
            dh, eps_h = np.array([0.0]), np.array([0.0])
        else:
            dh = np.atleast_1d(mu_h.discretise(model_uncertainties.step_size["h"])[0]).astype(float)
            mu_h1 = model_uncertainties.get_model_uncertainty("h", 1)
            dh1 = np.atleast_1d(mu_h1.discretise(model_uncertainties.step_size["h"])[0]).astype(float)
            eps_h = (dh1 - mu_h1.mu) / mu_h1.sigma

        if response == "h":
            combinations = [
                (float(d), 1.0, 1.0, 1.0, float(e * e), (float(e * e), 0.0, 0.0))
                for d, e in zip(dh, eps_h, strict=True)
            ]
            combinations.sort(key=lambda c: c[4])
            return combinations

        # Crest level: the wave uncertainties. Tspec and Tp are conditional on the bin of Hs through the correlation, so
        # their grid is built with 'onzwaarden_vector'
        mu_hs = model_uncertainties.get_model_uncertainty("hs")
        mu_ts = model_uncertainties.get_model_uncertainty("tspec")
        mu_tp = model_uncertainties.get_model_uncertainty("tp")

        if mu_hs is not None:
            hs_values = np.atleast_1d(mu_hs.discretise(model_uncertainties.step_size["hs"])[0]).astype(float)
            hs_raw = hs_values / mu_hs.mu
            eps_hs = (hs_raw - 1.0) / mu_hs.sigma
        else:
            hs_values, hs_raw, eps_hs = np.array([1.0]), np.array([1.0]), np.array([0.0])

        rho_ts = HydraIllustrationPoint.rho(model_uncertainties, "hs", "tspec")
        rho_tp = HydraIllustrationPoint.rho(model_uncertainties, "hs", "tp")
        n_period = model_uncertainties.step_size["tspec"] if mu_ts is not None else 1

        combinations = []
        for j in range(len(hs_values)):
            if mu_ts is not None:
                # Conditional uncertainty of Tspec given the bin of Hs (ONZWAARDEN)
                centre_ts = 1.0 + rho_ts * (mu_ts.sigma / mu_hs.sigma) * (hs_raw[j] - 1.0)
                sigma_ts = max(0.001, mu_ts.sigma * np.sqrt(1.0 - rho_ts * rho_ts))
                ts_raw = HydraIllustrationPoint.onzwaarden_vector(centre_ts, sigma_ts, n_period)
                eps_ts = (ts_raw - centre_ts) / sigma_ts
                ts_values = ts_raw * mu_ts.mu
            else:
                ts_raw, eps_ts, ts_values, centre_ts, sigma_ts = (
                    np.array([1.0]), np.array([0.0]), np.array([1.0]), 1.0, 1.0,
                )
            if mu_tp is not None and mu_ts is not None:
                # Tp is proportional to the realisation of Tspec: no dimension of its own in the distance
                centre_tp = 1.0 + rho_tp * (mu_tp.sigma / mu_hs.sigma) * (hs_raw[j] - 1.0)
                sigma_tp = max(0.001, mu_tp.sigma * np.sqrt(1.0 - rho_tp * rho_tp))
                tp_values = (centre_tp + (ts_raw - centre_ts) * sigma_tp / sigma_ts) * mu_tp.mu
            else:
                tp_values = np.full(len(ts_values), float(mu_tp.mu) if mu_tp is not None else 1.0)

            for i in range(len(dh)):
                for k in range(len(ts_values)):
                    e_h2, e_hs2, e_ts2 = eps_h[i] ** 2, eps_hs[j] ** 2, eps_ts[k] ** 2
                    distance2 = e_h2 + e_hs2 + e_ts2
                    combinations.append(
                        (
                            float(dh[i]), float(max(0.0, hs_values[j])), float(max(0.0, ts_values[k])),
                            float(max(0.0, tp_values[k])), float(distance2), (float(e_h2), float(e_hs2), float(e_ts2)),
                        )
                    )
        combinations.sort(key=lambda c: c[4])
        return combinations
