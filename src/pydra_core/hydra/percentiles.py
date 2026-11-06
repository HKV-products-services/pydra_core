"""
The Hydra-NL percentiles of the slow stochastic and the wind speed, for the lakes and the upper rivers: the
'T00x_UITSPL' and 'T00x_UR' tables. The Fortran chain behind it is described in docs/hydra_extensions.qmd.
"""

import copy
import math
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass

from .library import Contour, Grid, HydraInterpolate, Percentiles, Scaling, WaveProcessing
from ..core.memory import Memory, RememberedCalculation


@dataclass(frozen=True)
class _SlowPeak:
    """
    The slow stochastic (lake level or discharge) of the breakdown.

    'key' is the name of the axis in 'stochastics_slow' and in the loading table ('LoadingModel.<key>'). 'stat_attr'
    is the 'Statistics' attribute with the peak and block grids; 'peak_attr', 'peak_probability_attr' and
    'block_attr' are the peak values, their exceedance probability and the block grid. 'wave_type' is the name of the
    'WaveShapeType'. 'rise_attr' is the setting with the climate rise of the peak values (None for the discharge),
    'transition_attr' the setting with the shift of the wave shape (VERSCHUIVING; None for the discharge, whose wave
    is not shifted). 'clamp_duration' says whether the exceedance durations are clamped to at least one block
    duration. 'output_key' is the key of the percentiles in the output.
    """

    key: str
    stat_attr: str
    peak_attr: str
    peak_probability_attr: str
    block_attr: str
    wave_type: str
    rise_attr: str | None
    transition_attr: str | None
    clamp_duration: bool
    output_key: str


_SLOW: dict[str, _SlowPeak] = {
    "a": _SlowPeak(
        "a", "lake_level", "apeak", "epapeak", "ablok", "LAKE_LEVEL",
        "lake_level_rise", "transition_lake_wave_shape", True, "lake_level",
    ),
    "q": _SlowPeak("q", "discharge", "qpeak", "epqpeak", "qblok", "DISCHARGE", None, None, False, "discharge"),
}


class HydraPercentiles(metaclass=RememberedCalculation):
    """
    The Hydra-NL percentiles of the slow stochastic and the wind speed for one location of a lake or an upper river.

    One slow stochastic and one closing situation are covered; anything else raises a 'NotImplementedError'.
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
        Prepare the percentiles for one location of a lake or an upper river.

        Parameters
        ----------
        hrd : HRDatabase
            The database of the location
        settings : Settings
            The settings of the location, as the caller wants to calculate with them. They are copied, not changed.
        statistical_uncertainty : bool
            Whether the statistics include the statistical uncertainty
        model_uncertainty : bool
            Whether the model uncertainty is integrated out
        response : str
            "h" (water level) or "hbn" (required crest level; needs 'profile' and 'qcrit')
        profile : Profile
            The profile for the crest level
        qcrit : float
            The critical overtopping discharge for the crest level
        crest_level_table : Callable
            The function that calculates the crest level on a loading grid, with the signature of
            'core.crest_level_tables.crest_level_table'. By default the tables are kept in 'core.memory'; a caller that
            keeps its tables passes its own function here, and the object is then not kept either.
        """
        self.response = response
        self.model_uncertainty = model_uncertainty
        self._profile = profile
        self._qcrit = float(qcrit) if qcrit is not None else None
        self._crest_level_table = (
            crest_level_table if crest_level_table is not None else Memory.remembered_crest_level_table(settings)
        )
        if response == "hbn" and (profile is None or qcrit is None):
            raise ValueError("[ERROR] The crest level needs a profile and a critical overtopping discharge (qcrit).")

        # 'WaveShape' changes 'base_duration' when the location is built, so the Hydra-NL parameters and the wave
        # shapes come from a copy that is not used for anything else
        fresh = copy.deepcopy(settings)
        settings = copy.deepcopy(settings)
        if not statistical_uncertainty:
            settings.include_statistical_uncertainty(False)
        model = hrd.create_location(settings).get_model()
        loading = model.get_loading()
        stats = model.get_statistics()

        slow_keys = list(stats.stochastics_slow.keys())
        if len(slow_keys) != 1:
            raise NotImplementedError(
                f"[ERROR] The percentiles cover one slow stochastic; {settings.location} has {slow_keys}."
            )
        self._slow = _SLOW.get(slow_keys[0])
        if self._slow is None:
            raise NotImplementedError(f"[ERROR] The slow stochastic {slow_keys[0]!r} is not covered.")
        n_closing_situations = len({k for (_r, k) in loading.model})
        if n_closing_situations != 1:
            raise NotImplementedError(
                f"[ERROR] The percentiles cover one closing situation; {settings.location} has {n_closing_situations}."
            )

        self._umax = float(fresh.u_max)
        self._block_duration_wind = float(fresh.block_duration)

        # Wind statistics: exceedance probability of the potential wind per direction, and the direction probabilities
        self._u_stat = np.asarray(stats.wind_speed.get_discretisation(), dtype=float)
        self._ovkansu = np.asarray(stats.wind_speed.get_exceedance_probability(), dtype=float)
        self._r_stat = np.asarray(stats.wind_direction.get_discretisation(), dtype=float)
        self._p_direction = np.asarray(stats.wind_direction.get_probability(), dtype=float)

        # Slow stochastic: peak grid with exceedance probabilities, and the block grid. The block grid equals the peak
        # grid; ROOSTERBLOKKEN only extends it when the lower bound lies above the minimum of the statistics.
        slow_stat = getattr(stats, self._slow.stat_attr)
        self._peaks = np.asarray(getattr(slow_stat, self._slow.peak_attr), dtype=float)
        self._peak_probability = np.asarray(getattr(slow_stat, self._slow.peak_probability_attr), dtype=float)
        self._block = np.asarray(getattr(slow_stat, self._slow.block_attr), dtype=float)

        rise = getattr(fresh, self._slow.rise_attr, 0.0) if self._slow.rise_attr else 0.0
        transition = getattr(fresh, self._slow.transition_attr, None) if self._slow.transition_attr else None
        self._times, self._wave_shapes, wave_shape = WaveProcessing.wave_shapes_on_hydra_axis(
            fresh, self._slow.wave_type, self._peaks, rise=rise or 0.0, transition=transition
        )
        self._durations = WaveProcessing.exceedance_durations(
            wave_shape, self._block, self._wave_shapes, self._block_duration_wind, clamp=self._slow.clamp_duration
        )

        # Breakdown grid of the wind speed: VECTORVULLEN(0, UMAX, 0.2), with the density per direction (STAT_WIND)
        nu = math.floor(self._umax / 0.2 + 0.5) + 1
        self._u_breakdown = np.array([i * 0.2 for i in range(nu - 1)] + [self._umax])
        densities = []
        for ir in range(self._ovkansu.shape[1]):
            ov = np.maximum(
                HydraInterpolate.inextrp1d_interp(self._u_breakdown, self._u_stat, self._ovkansu[:, ir]), 0.0
            )
            densities.append(Grid.probability_density_raw(self._u_breakdown, ov))
        self._density_u = np.column_stack(densities)

        # Loading tables per wind direction, with the slow axis 'a' (lakes) or 'q' (rivers)
        self._table: dict[float, tuple] = {}
        for (r, _k), lm in loading.model.items():
            u = np.asarray(lm.u, dtype=float)
            s = np.asarray(getattr(lm, self._slow.key), dtype=float)
            h = np.asarray(lm.h, dtype=float)
            if response == "hbn":
                self._table[float(r)] = (
                    u, s, h, np.asarray(lm.hs, dtype=float), np.asarray(lm.tspec, dtype=float),
                    np.asarray(lm.dir, dtype=float),
                )
            else:
                self._table[float(r)] = (u, s, h)

        # ONZWAARDEN with KANSONZ: (delta_h, f_hs, f_tspec, probability)
        self._combinations = Scaling.uncertainty_combinations(stats, self.response, self.model_uncertainty)

        # The response tables do not depend on the level, and for the crest level every table is a call of the
        # overtopping library, so they are kept per (combination, direction)
        self._response_tables: dict[tuple[int, float], np.ndarray] = {}

    def response_table(self, ic: int, r: float) -> np.ndarray:
        """
        Return the response on the (u, s) grid: h + delta_h for the water level, the required crest level at
        (h + delta_h, Hm0 * f_hs, Tspec * f_tspec) for the crest level.

        Parameters
        ----------
        ic : int
            The index of the uncertainty combination
        r : float
            The wind direction

        Returns
        -------
        np.ndarray
            The response on the (u, s) grid
        """
        key = (ic, r)
        if key not in self._response_tables:
            dh, f_hs, f_ts, _ = self._combinations[ic]
            if self.response == "hbn":
                _u, _s, h, hs, ts, dr = self._table[r]
                self._response_tables[key] = self._crest_level_table(
                    self._profile, self._qcrit, h + dh, hs, ts, dr, f_hs, f_ts
                )
            else:
                _u, _s, h = self._table[r]
                self._response_tables[key] = h + dh
        return self._response_tables[key]

    def characteristic_wind_speed(self, ic: int, r: float, level: float) -> np.ndarray:
        """
        Determine u_characteristic on the block grid of the slow stochastic: the wind speed on the contour
        response(u, s) = level, per value of the slow stochastic. ISOLIJN ('Hydra_contour.f90'), CONTOUREN_SAMENVOEGEN
        and VLAKKEN_CONTOUR ('blokkans.f90') for the rivers, CONTOURVERWERKING for the lakes.

        Where even the maximum response over the wind does not reach the level, s cannot be reached and
        u_characteristic is UMAX (block probability 0). That is the Fortran pattern 'HMAXQ < HULPDIJK' in
        BLOKKENREKENEN_R.

        Parameters
        ----------
        ic : int
            The index of the uncertainty combination
        r : float
            The wind direction
        level : float
            The level

        Returns
        -------
        np.ndarray
            u_characteristic on the block grid
        """
        table = self._table[r]
        u, s = table[0], table[1]
        resp = self.response_table(ic, r)

        if self._slow.key == "a":
            return self.characteristic_wind_speed_lake(u, s, resp, level)
        wu_c, ws_c = Contour.merge_along_axis(*Contour.isoline(u, s, resp, level))
        u_characteristic = Contour.flatten_contour(wu_c, ws_c, self._umax, self._block)
        if u_characteristic is None:
            u_characteristic = np.full(len(self._block), 0.0 if resp[0, 0] >= level else self._umax)
        hmax = HydraInterpolate.inextrp1d_interp(self._block, s, resp.max(axis=0))
        return np.where(hmax < level, self._umax, u_characteristic)

    def characteristic_wind_speed_lake(self, u, s, resp, level) -> np.ndarray:
        """
        Determine u_characteristic from the contour of the lakes, as BLOKKENREKENEN_M with CONTOURVERWERKING
        ('contour.f90').

        Hydra-NL only calls CONTOUREN_SAMENVOEGEN for a contour of more than one part ('MINVAL(WM) <= -9999'); a
        contour of one part goes to CONTOURVERWERKING in the order of the path. Forcing strictly increasing values
        (+1e-6) then flattens a local inversion in s (the ripple of the overtopping library at high wind speeds);
        sorting on s would make the edge section increasing in u and the extrapolation would flip to u = 0
        (block probability 1). Without a contour (fewer than two points) the corner point rule of BLOKKENREKENEN_M
        holds: UMAX if the response in the first corner lies below the level, 0 otherwise.

        Parameters
        ----------
        u : np.ndarray
            The wind speeds of the loading table
        s : np.ndarray
            The lake levels of the loading table
        resp : np.ndarray
            The response on the (u, s) grid
        level : float
            The level

        Returns
        -------
        np.ndarray
            u_characteristic on the block grid
        """
        wu_raw, ws_raw = Contour.isoline(u, s, resp, level)
        wu_raw = np.asarray(wu_raw, float)
        ws_raw = np.asarray(ws_raw, float)
        if ws_raw.size and (ws_raw.min() <= -9999.0 or wu_raw.min() <= -9999.0):
            wu_c, ws_c = Contour.merge_contours(wu_raw, ws_raw, float(s[0]), float(s[-1]))
        else:
            wu_c, ws_c = wu_raw, ws_raw
        if wu_c is None or len(wu_c) < 2:
            return np.full(len(self._block), self._umax if resp[0, 0] < level else 0.0)
        ws_c = np.array(ws_c, dtype=float, copy=True)
        for ip in range(len(ws_c) - 1):
            if ws_c[ip + 1] <= ws_c[ip]:
                ws_c[ip + 1] = ws_c[ip] + 1.0e-6
        return np.clip(HydraInterpolate.inextrp1d_interp(self._block, ws_c, np.asarray(wu_c, float)), 0.0, self._umax)

    def breakdown(self, level: float) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        """
        Break the exceedance frequency of a level down over the slow stochastic, the wind speed and the wind direction,
        summed over the uncertainty combinations with their KANSONZ.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict, dict]
            FREQM on the block grid, FREQU on the grid 0:0.2:UMAX, FREQUR per wind direction and FREQR per wind
            direction
        """
        freq_slow = np.zeros(len(self._block))
        freq_u = np.zeros(len(self._u_breakdown))
        freq_ur = {r: np.zeros(len(self._u_breakdown)) for r in self._table}
        freq_r = dict.fromkeys(self._table, 0.0)
        factors = WaveProcessing.block_factors(self._times)
        density_peak = Grid.probability_density_raw(self._peaks, self._peak_probability)
        delta_peak = Grid.step_sizes(self._peaks)
        delta_s = Grid.step_sizes(self._block)
        delta_u = Grid.step_sizes(self._u_breakdown)

        for ic, (_dh, _f_hs, _f_ts, p_combination) in enumerate(self._combinations):
            if p_combination <= 0.0:
                continue
            u_characteristic = {}
            ovk_characteristic = {}
            for r in self._table:
                u_characteristic[r] = self.characteristic_wind_speed(ic, r, level)
                ir = int(np.argmin(np.abs(self._r_stat - r)))
                ovk_characteristic[r] = np.maximum(
                    HydraInterpolate.inextrp1d_interp(u_characteristic[r], self._u_stat, self._ovkansu[:, ir]), 0.0
                )

            # Block probability p(s) = sum_r P(r) * P_pot(U >= u_characteristic(s, r) | r)
            p_block = np.zeros(len(self._block))
            for r in self._table:
                ir = int(np.argmin(np.abs(self._r_stat - r)))
                p_block += self._p_direction[ir] * ovk_characteristic[r]

            # SCHALING per peak (VERWERKGOLF), with the block probability read linearly along one wave shape
            ovk_t = np.empty((len(self._times), len(self._peaks)))
            for ip in range(len(self._peaks)):
                ovk_t[:, ip] = np.interp(self._wave_shapes[:, ip], self._block, p_block)
            scaling = WaveProcessing.process_wave_scaling(
                ovk_t, factors, self._times[1] - self._times[0], self._block_duration_wind
            )

            # Momentary frequency density of the slow stochastic (FREQDICHTHM, formula 6.3)
            ovfreq = (self._durations * (density_peak * scaling * delta_peak)[:, None]).sum(axis=0)
            density_slow = Grid.probability_density_raw(self._block, ovfreq)

            # SPLITSENMQ: breakdown over the slow stochastic and the wind speed
            freq_slow += p_combination * p_block * density_slow * delta_s
            weight_s = density_slow * delta_s
            for r in self._table:
                ir = int(np.argmin(np.abs(self._r_stat - r)))
                mask = self._u_breakdown[:, None] >= u_characteristic[r][None, :]
                contribution_u = (
                    (p_combination * self._p_direction[ir]) * (self._density_u[:, ir] * delta_u) * (mask @ weight_s)
                )
                freq_u += contribution_u
                freq_ur[r] += contribution_u
                # FREQR: the contribution of the direction to the breakdown over the slow stochastic (FREQRKD)
                freq_r[r] += float(p_combination * self._p_direction[ir] * (ovk_characteristic[r] * weight_s).sum())

        return freq_slow, freq_u, freq_ur, freq_r

    def wind_percentiles_per_direction(self, level: float) -> dict:
        """
        Calculate the wind speed percentiles per wind direction, the T00x_UR table.

        Per direction the percentage 100 * FREQR(r) / sum FREQR (PERCENTAGE_UR) and the percentiles of the wind
        breakdown of that direction. Hydra-NL prints '-----' where FREQR(r) / total < 5e-4; that is presentation and
        is left to the caller, which gets the percentage.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The percentile labels, per wind direction the percentage and the wind speed percentiles, and 'total': sum
            FREQR, the absolute exceedance frequency of the breakdown at this level
        """
        _freq_slow, _freq_u, freq_ur, freq_r = self.breakdown(float(level))
        total = sum(freq_r.values())
        directions = []
        for r in sorted(freq_ur):
            pu = Percentiles.percentiles(self._u_breakdown, freq_ur[r]) if freq_r[r] > 0.0 else None
            directions.append(
                {
                    "r": float(r),
                    "percentage": (100.0 * freq_r[r] / total) if total > 0.0 else 0.0,
                    "wind_speed": None if pu is None else [float(v) for v in pu],
                }
            )
        return {"pct": Percentiles.percentile_labels(), "directions": directions, "total": float(total)}

    def percentiles(self, level: float) -> dict:
        """
        Calculate the percentiles of the slow stochastic and the wind speed at a level, the T00x_UITSPL tables.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The percentile labels and the percentiles of the slow stochastic ("lake_level" or "discharge") and the
            wind speed
        """
        freq_slow, freq_u, _freq_ur, _freq_r = self.breakdown(float(level))
        ps = Percentiles.percentiles(self._block, freq_slow)
        pu = Percentiles.percentiles(self._u_breakdown, freq_u)
        return {
            "pct": Percentiles.percentile_labels(),
            self._slow.output_key: None if ps is None else [float(v) for v in ps],
            "wind_speed": None if pu is None else [float(v) for v in pu],
        }
