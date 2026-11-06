"""
The Hydra-NL breakdown, percentiles and frequency line for the IJssel-Vecht delta (REGIO 1), with the split over the
open and closed Ramspol barrier. The Fortran chain, the two loading tables and the scaling are described in
docs/hydra_extensions.qmd.
"""

import copy
import numpy as np

from collections.abc import Callable
from scipy import sparse
from scipy.stats import norm
from typing import NamedTuple

from . import compiled_wind_segments
from .ijssel_vecht_delta import BarrierClosingData, HydraIJsselVechtDeltaIllustrationPoint
from .illustration_points_3d import HydraIllustrationPoint3D
from .library import Grid, HydraInterpolate, Percentiles, Scaling, WaveProcessing
from ..core.memory import Memory, RememberedCalculation


class _BarrierAndSegments(NamedTuple):
    """
    The part of 'breakdown_of_table' per wind direction that does not depend on the uncertainty combination.
    """

    b_open: np.ndarray  # BIJDRAGEKERING open, per (m, q) cell
    b_closed: np.ndarray  # BIJDRAGEKERING closed
    closes: np.ndarray  # whether the barrier closes in this cell
    closing_wind: np.ndarray  # the closing wind speed
    uk1: np.ndarray  # begin of the first segment of S1
    segments_s2: tuple  # wind segments for BIJDRAGENWIND: S2
    segments_s2_below: tuple  # S2 below the closing wind speed
    segments_s1_above: tuple  # S1 above the closing wind speed


class HydraIJsselVechtDeltaPercentiles(metaclass=RememberedCalculation):
    """
    The Hydra-NL breakdown of the IJssel-Vecht delta for one location: percentiles, the split over the open and closed
    barrier, and the frequency line along Hydra-NL's route.

    A location without the slow stochastics lake level and discharge and the closing situations 1 and 2 is refused
    with a 'NotImplementedError'.
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
        hydra_ini_directory=None,
        with_repaired_loading: bool = False,
    ):
        """
        Prepare the breakdown for one location in the IJssel-Vecht delta.

        Parameters
        ----------
        hrd : HRDatabase
            The database of the location
        settings : Settings
            The settings of the location, as the caller wants to calculate with them. They are copied, not changed.
        statistical_uncertainty : bool
            Whether the statistics include the statistical uncertainty
        model_uncertainty : bool
            Whether the model uncertainty is integrated over
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
        hydra_ini_directory : str or Path
            The folder 'data/ini' of the Hydra-NL installation, with the KeringUD CSV of the region. Only read if the
            database has no KeringUD table.
        with_repaired_loading : bool
            Also hold Hydra-NL's repaired loading table, for comparing the method with Hydra-NL (see
            docs/hydra_extensions.qmd)
        """
        self.statistical_uncertainty = statistical_uncertainty
        self.model_uncertainty = model_uncertainty
        self.response = response
        self.profile = profile
        self.qcrit = float(qcrit) if qcrit is not None else None
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
        if response == "hs":
            # Hydra-NL repairs a wave height line, so the waves are repaired here as well
            loading.repair_loadingmodels("hs")
        stats = model.get_statistics()
        if set(stats.stochastics_slow) != {"a", "q"} or sorted(stats.barrier.k) != [1, 2]:
            raise NotImplementedError(
                f"[ERROR] {self.location} is not a location of the IJssel-Vecht delta: the breakdown expects the slow "
                f"stochastics lake level and discharge ('a', 'q') and the closing situations 1 and 2 of the Ramspol "
                f"barrier."
            )

        self._u_max = float(getattr(settings, "u_max", 50.0) or 50.0)
        self._alpha = float(getattr(settings, "failure_probability_ramspol", 0.01) or 0.01)
        self._block_duration_wind = float(settings.block_duration)
        # Base durations per winter half year
        self._n_data_blocks = float(settings.periods_base_duration)
        # For the levels of the line: KANSMASSA = OVMINM ('stat_meerpeil.f90') and VLAG_AFTOPPEN ('invoer_Zoet.f90')
        self._probability_mass = min(1.0, float(stats.lake_level.epapeak[0]))
        self._capped = (settings.q_limit < settings.q_max) or (settings.a_limit < settings.a_max)

        # Wind statistics per direction; pydra's grid already runs to UMAX, including the extrapolated tail
        self._u_stat = np.asarray(stats.wind_speed.get_discretisation(), float)
        self._epu = np.asarray(stats.wind_speed.get_exceedance_probability(), float)  # (u, r)
        self._r_stat = np.asarray(stats.wind_direction.get_discretisation(), float)
        self._p_r = np.asarray(stats.wind_direction.get_probability(), float)

        # Block grids, equal to the breakdown grids MBLOK and QBLOK
        self._mblock = np.asarray(stats.lake_level.ablok, float)
        self._qblock = np.asarray(stats.discharge.qblok, float)
        self._dm = Grid.step_sizes(self._mblock)
        self._dq = Grid.step_sizes(self._qblock)

        # Wind speed grid of the breakdown, VECTORVULLEN(0, UMAX, 0.2) for REGIO != 0, and its density (STAT_WIND)
        self._u_breakdown = Grid.fill_vector(0.0, self._u_max, 0.2)
        self._du_breakdown = Grid.step_sizes(self._u_breakdown)
        densities = []
        for ir in range(self._epu.shape[1]):
            exceedance = np.maximum(
                HydraInterpolate.inextrp1d_fractions(self._u_stat, self._epu[:, ir], self._u_breakdown), 0.0
            )
            densities.append(Grid.probability_density_raw(self._u_breakdown, exceedance))
        self._density_u = np.column_stack(densities)  # (u_breakdown, r)

        # Joint peak density, equal to Hydra-NL's DICHTH_MQPIEK, and the peak step sizes
        self._peak = np.asarray(stats.density_aq_peak, float)  # (s = lake level peak, k = discharge peak)
        self._dmp = Grid.step_sizes(np.asarray(stats.lake_level.apeak, float))
        self._dqp = Grid.step_sizes(np.asarray(stats.discharge.qpeak, float))

        # Wave shapes on the shared time axis
        ws_m, ws_q = stats.lake_level.wave_shape, stats.discharge.wave_shape
        times_m = np.asarray(ws_m.time, float)
        times_q = np.asarray(ws_q.time, float)
        if len(times_m) != len(times_q) or not np.allclose(times_m, times_q):
            raise NotImplementedError(
                "[ERROR] The lake level and discharge waves do not share a time axis, which VERWERKGOLF assumes."
            )
        self._times = times_m
        self._waves_m = np.asarray(ws_m.wave_shapes, float)  # (t, s)
        self._waves_q = np.asarray(ws_q.wave_shapes, float)  # (t, k)
        self._block_factors = WaveProcessing.block_factors(self._times)

        # Crossing times per peak wave and block level, for OVERSCHRIJDINGSDUREN; they do not depend on the level
        self._crossing_m = [
            HydraIJsselVechtDeltaIllustrationPoint.crossing_times(self._times, self._waves_m[:, s], self._mblock)
            for s in range(self._waves_m.shape[1])
        ]
        crossing_q = [
            HydraIJsselVechtDeltaIllustrationPoint.crossing_times(self._times, self._waves_q[:, k], self._qblock)
            for k in range(self._waves_q.shape[1])
        ]
        self._s1q = np.stack([t[0] for t in crossing_q])  # (k, q)
        self._s2q = np.stack([t[1] for t in crossing_q])
        self._middle_q = np.stack([t[3] for t in crossing_q]) == 1

        # Interpolation of the wave shapes on the block grids (V_INDICES_ONDER and CALC_FRACS): the lower index is
        # clamped, the fraction is not, so the edge extrapolates
        self._m_index, self._m_fraction = Grid.lower_index_and_fraction(self._mblock, self._waves_m)
        self._q_index, self._q_fraction = Grid.lower_index_and_fraction(self._qblock, self._waves_q)

        # Loading per (r, input IK), axes (u, m, q), in two forms:
        #   '_loading_line'     the table of the line and the breakdown: as pydra delivers it, extended linearly to UMAX
        #                       and not repaired again;
        #   '_loading_repaired' only with 'with_repaired_loading': repaired, extended to UMAX and repaired again, as
        #                       Hydra-NL does.
        # Input IK = 1 (functioning) is pydra's k = 2 (regular closing); input IK = 2 (failing, always open) is k = 1.
        wanted = {"h": ["h"], "hs": ["hs"], "hbn": ["h", "hs", "tspec", "dir"]}[response]
        # The fields the response depends on
        self._response_fields = wanted
        self._loading_repaired: dict[tuple[float, int], dict] = {}
        self._loading_line: dict[tuple[float, int], dict] = {}
        for (r, k), lm in loading.model.items():
            iv = list(lm.input_variables)
            perm = [iv.index("u"), iv.index("a"), iv.index("q")]
            arrays = {name: np.transpose(np.asarray(getattr(lm, name), float), perm) for name in wanted}
            ik_in = 3 - int(k)
            axes = {"m": np.asarray(lm.a, float), "q": np.asarray(lm.q, float)}
            u_line, line = HydraIllustrationPoint3D.extend_u(np.asarray(lm.u, float), arrays, self._u_max)
            self._loading_line[(float(r), ik_in)] = {"u": u_line, **axes, **line}
            if not with_repaired_loading:
                continue
            # The water level (water level and crest level) or the wave height
            main = wanted[0]
            arrays[main] = HydraIllustrationPoint3D.repair_3d(arrays[main])
            u_ext, arrays = HydraIllustrationPoint3D.extend_u(np.asarray(lm.u, float), arrays, self._u_max)
            arrays[main] = HydraIllustrationPoint3D.repair_3d(arrays[main])
            self._loading_repaired[(float(r), ik_in)] = {"u": u_ext, **axes, **arrays}
        self._directions = sorted({r for (r, _ik) in self._loading_line})

        # The closing wind speed per direction on the (m, q) grid (KeringUD, BLOKSNELHEID)
        barrier = BarrierClosingData(settings.database_path, hydra_ini_directory)
        self._closing_wind = {
            r: BarrierClosingData.closing_wind_speed_on_grid(barrier, r, self._mblock, self._qblock)
            for r in self._directions
        }

        # Uncertainty combinations (delta_h, f_hs, f_tspec, probability): ONZWAARDEN with KANSONZ
        self._combinations = self.build_combinations(stats)
        self._breakdown_cache: dict[float, dict] = {}
        self._duration_matrix_cache: sparse.csr_matrix | None = None

        # Response tables; the key is coarser than (ic, r, ik), see 'response_table'
        self._response_cache: dict[tuple, np.ndarray] = {}
        self._equal_over_barrier_repaired = (
            self.loading_equal_over_barrier(self._loading_repaired) if with_repaired_loading else {}
        )
        self._equal_over_barrier_line = self.loading_equal_over_barrier(self._loading_line)
        # Per (line, direction): whether the axes u, m and q are equal for both closing situations; see
        # 'block_probability'
        self._axes_equal = {
            (line, r): all(np.array_equal(loading[(r, 1)][a], loading[(r, 2)][a]) for a in "umq")
            for line, loading in ((True, self._loading_line), (False, self._loading_repaired))
            if loading
            for r in self._directions
        }
        self._waveless: dict[tuple, bool] = {}
        self._ahead_done = False
        self._first_table: dict[tuple, tuple[tuple[float, float], np.ndarray]] = {}

    def loading_equal_over_barrier(self, loading: dict) -> dict[float, bool]:
        """
        Determine per direction whether the loading the response depends on is bit identical for both closing
        situations. Then one response table serves both.

        Only '_response_fields' is compared. The axes u, m and q stay separate per closing situation: they bypass
        'response_table' ('block_probability' reads them itself), and the response is an elementwise function of the
        fields, so equal fields give an equal table whatever the axes.

        In the IJsseldelta this holds for 9 of the 16 directions, 22.5 to 202.5 degrees: the sector where the Ramspol
        barrier does nothing.

        Parameters
        ----------
        loading : dict
            The loading per (direction, input IK)

        Returns
        -------
        dict[float, bool]
            Per direction True if the loading is bit identical for both closing situations
        """
        equal: dict[float, bool] = {}
        for r in self._directions:
            a, b = loading.get((r, 1)), loading.get((r, 2))
            equal[r] = bool(
                a is not None
                and b is not None
                and all(a[v].shape == b[v].shape and np.array_equal(a[v], b[v]) for v in self._response_fields)
            )
        return equal

    def learn_waveless(self, rk: tuple, dh: float, factors: tuple[float, float], table: np.ndarray) -> None:
        """
        Learn with one comparison per (direction, closing situation) whether the wave does anything here.

        The combination loop runs delta outside and the wave inside, so the second table with the same delta has other
        wave factors. If it is bit identical to the first, the overtopping calculation does not take the wave into
        account for this direction, and 'f_hs' and 'f_tspec' may leave the cache key. After that nothing is compared.

        The bound lies in the overtopping calculation and is not read from a direction threshold: this comparison
        proves it on the own loading table, without an extra overtopping call, because both tables were needed anyway.

        Parameters
        ----------
        rk : tuple
            The key (line, direction, closing situation) of the response table
        dh : float
            The shift of the water level
        factors : tuple[float, float]
            The factors for the significant wave height and the spectral wave period
        table : np.ndarray
            The crest level table of this combination
        """
        if rk in self._waveless:
            return
        earlier = self._first_table.get((rk, dh))
        if earlier is None:
            self._first_table[(rk, dh)] = (factors, table)
            return
        old_factors, old_table = earlier
        if old_factors == factors:
            return
        self._waveless[rk] = bool(np.array_equal(old_table, table))
        if self._waveless[rk]:
            self._response_cache[(rk, dh)] = table

    # Uncertainty combinations

    def build_combinations(self, stats) -> list[tuple[float, float, float, float]]:
        """
        Build the uncertainty combinations with their probability KANSONZ (ONZWAARDEN).

        Parameters
        ----------
        stats : Statistics
            The statistics of the location

        Returns
        -------
        list[tuple[float, float, float, float]]
            The combinations (delta_h, f_hs, f_tspec, probability)
        """
        if not self.model_uncertainty:
            return [(0.0, 1.0, 1.0, 1.0)]
        mu = stats.get_model_uncertainties()
        if self.response == "hs":
            waves = list(mu.iterate_model_uncertainty_wave_conditions(wave_period="tspec"))
            return [(0.0, float(fhs), float(fts), float(pw)) for fhs, fts, pw in waves]
        mu_h = mu.get_model_uncertainty("h")
        dh, edges = mu_h.discretise(mu.step_size["h"])
        dh = np.atleast_1d(dh).astype(float)
        cdf = norm.cdf(np.asarray(edges, float))
        p_dh = cdf[1:] - cdf[:-1]
        if self.response == "hbn":
            waves = list(mu.iterate_model_uncertainty_wave_conditions(wave_period="tspec"))
        else:
            waves = [(1.0, 1.0, 1.0)]
        return [
            (float(d), float(fhs), float(fts), float(pd * pw)) for d, pd in zip(dh, p_dh, strict=True) for fhs, fts, pw in waves
        ]

    def tables_ahead(self) -> None:
        """
        Calculate the crest level tables of '_loading_line' for every combination with a probability, once, in worker
        processes ('Memory.calculate_ahead'); 'response_table' then finds them in the store.
        """
        if self.response != "hbn" or self._ahead_done:
            return
        self._ahead_done = True
        combinations = [(dh, f_hs, f_ts) for dh, f_hs, f_ts, p_unc in self._combinations if p_unc > 0.0]
        work = []
        for r in self._directions:
            for ik in (1,) if self._equal_over_barrier_line.get(r) else (1, 2):
                loading = self._loading_line[(r, ik)]
                work.append(((loading["h"], loading["hs"], loading["tspec"], loading["dir"]), combinations))
        Memory.calculate_ahead(self._crest_level_table, self.profile, self.qcrit, work)

    def response_table(self, ic: int, r: float, ik_in: int, line: bool = True) -> np.ndarray:
        """
        Return the response table (u, m, q) for an uncertainty combination. Water level and wave height: the table
        itself, the level shifts instead (level - delta, level / f_hs). Crest level: the crest level table per
        combination (h + delta, hs * f_hs, tspec * f_tspec).

        The key is coarser than (ic, r, ik):

        * the water level and wave height do not depend on the combination; that table is there once per (direction,
          closing situation);
        * where 'loading_equal_over_barrier' says equal, one table serves both closing situations;
        * where 'learn_waveless' found that the wave does nothing, the wave factors leave the key and only delta stays.

        The table is therefore shared and read-only: a caller that writes into it fails loudly instead of silently
        changing another combination.

        Parameters
        ----------
        ic : int
            The index of the uncertainty combination
        r : float
            The wind direction
        ik_in : int
            The closing situation on the input side (1 functioning, 2 failing)
        line : bool, optional
            True for '_loading_line', False for '_loading_repaired'. Default is True.

        Returns
        -------
        np.ndarray
            The response table on the (u, m, q) axes of the loading
        """
        if line:
            loading_set, equal = self._loading_line, self._equal_over_barrier_line
        else:
            loading_set, equal = self._loading_repaired, self._equal_over_barrier_repaired
        rk = (line, r, 1 if equal.get(r) else ik_in)
        dh = f_hs = f_ts = None
        if self.response == "hbn":
            dh, f_hs, f_ts, _ = self._combinations[ic]
            key = (rk, dh) if self._waveless.get(rk) else (rk, dh, f_hs, f_ts)
        else:
            key = rk

        table = self._response_cache.get(key)
        if table is not None:
            return table

        loading = loading_set[(r, ik_in)]
        if self.response == "hbn":
            table = np.array(
                self._crest_level_table(
                    self.profile, self.qcrit, loading["h"] + dh, loading["hs"], loading["tspec"], loading["dir"],
                    f_hs, f_ts,
                )
            )
            table.setflags(write=False)
            self.learn_waveless(rk, dh, (f_hs, f_ts), table)
        else:
            table = loading["h" if self.response == "h" else "hs"]
        self._response_cache[key] = table
        return table

    # Building blocks

    def block_probability(self, ic: int, level: float, line: bool, kept: dict | None = None) -> np.ndarray:
        """
        Calculate the block probability P(H >= h | q, m) as a sum over wind segments, per (direction, input IK), with
        the alpha mixing over the closing situations and summed over the directions.

        Parameters
        ----------
        ic : int
            The index of the uncertainty combination
        level : float
            The level, already shifted for the combination
        line : bool
            True for '_loading_line', False for '_loading_repaired'
        kept : dict | None, optional
            The block probability per (table, direction, axes, level) over the combinations: 'response_table' shares a
            table between closing situations and wave combinations, and then the block probability is the same call.
            Default is None.

        Returns
        -------
        np.ndarray
            The block probability on the (m, q) block grid
        """
        loading_set = self._loading_line if line else self._loading_repaired
        ovkans_qm = np.zeros((len(self._mblock), len(self._qblock)))
        for r in self._directions:
            ir = int(np.argmin(np.abs(self._r_stat - r)))
            ovk = {}
            for ik_in in (1, 2):
                loading = loading_set[(r, ik_in)]
                table = self.response_table(ic, r, ik_in, line)
                key = (id(table), line, r, 1 if self._axes_equal[(line, r)] else ik_in, level)
                if kept is not None and key in kept:
                    ovk[ik_in] = kept[key]
                    continue
                ovk[ik_in] = compiled_wind_segments.block_probability(
                    loading["u"], loading["m"], loading["q"], table, self._mblock, self._qblock, level, self._u_stat,
                    self._epu[:, ir], self._u_max,
                )
                if kept is not None:
                    kept[key] = ovk[ik_in]
            ovkans_qm += self._p_r[ir] * (self._alpha * ovk[2] + (1.0 - self._alpha) * ovk[1])
        return ovkans_qm

    def process_wave(self, ovkans_qm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process the block probability along the waves (VERWERKGOLF, REGIO 1), with the block probability bilinear on the
        simultaneous lake level and discharge waves.

        Parameters
        ----------
        ovkans_qm : np.ndarray
            The block probability on the (m, q) block grid

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            OVKANS_KS and SCHALING per (lake level peak s, discharge peak k)
        """
        ovk_t = compiled_wind_segments.on_waves(
            ovkans_qm, self._m_index, self._m_fraction, self._q_index, self._q_fraction
        )  # (t, s, k)
        return WaveProcessing.process_wave(
            ovk_t, self._block_factors, self._times[1] - self._times[0], self._block_duration_wind
        )

    def duration_matrix(self) -> sparse.csr_matrix:
        """
        Return all DUREN as one sparse matrix. Built once; it depends neither on the level nor on the combination.

        Sparse because about three quarters are zero. The order of summation differs from a dense sum, so the last
        digits shift (relative <= 1.3e-14).

        Returns
        -------
        sparse.csr_matrix
            The durations, with rows (s, k) and columns (m, q)
        """
        if self._duration_matrix_cache is None:
            blocks = [
                sparse.csr_matrix(dur.transpose(1, 0, 2).reshape(dur.shape[1], -1))
                for dur in (
                    HydraIJsselVechtDeltaIllustrationPoint.joint_durations(cm, self._s1q, self._s2q, self._middle_q)
                    for cm in self._crossing_m
                )
            ]  # (k, m * q) per s
            self._duration_matrix_cache = sparse.vstack(blocks, format="csr")
        return self._duration_matrix_cache

    def frequency_density_mq(self, scaling: np.ndarray) -> np.ndarray:
        """
        Calculate the joint frequency density (FREQDICHTHQM): OVFREQMQ(m, q) = sum over (s, k) of dm_s * dq_k *
        DICHTH_MQPIEK(s, k) * SCHALING(s, k) * DUREN(m, s, q, k), then KANSDICHTHEID twice (first to the lake level,
        then to the discharge).

        Parameters
        ----------
        scaling : np.ndarray
            SCHALING per (lake level peak s, discharge peak k)

        Returns
        -------
        np.ndarray
            The frequency density on the (m, q) block grid
        """
        w = self._dmp[:, None] * self._dqp[None, :] * self._peak * scaling  # (s, k)
        ovfreq = (self.duration_matrix().T @ w.ravel()).reshape(len(self._mblock), len(self._qblock))
        # Durations in units of the block duration
        ovfreq /= self._block_duration_wind
        between = np.empty_like(ovfreq)
        for iq in range(ovfreq.shape[1]):
            between[:, iq] = Grid.probability_density_raw(self._mblock, ovfreq[:, iq])
        density = np.empty_like(between)
        for im in range(between.shape[0]):
            density[im, :] = Grid.probability_density_raw(self._qblock, between[im, :])
        return density

    # The breakdown

    def breakdown(self, level: float) -> dict:
        """
        Calculate all breakdown arrays at a level, summed over the uncertainty combinations, on the table of the line.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The keys 'freqQK' (nq, 2), 'freqMK' (nm, 2), 'freqRK' (nr, 2), 'freqUK' (nu, 2), 'freqURK' (nu, nr, 2) and
            'freqUR' (nu, nr), with the barrier axis 0 = open and 1 = closed (output side), plus the grids 'q', 'm',
            'u' and 'r'
        """
        level = float(level)
        if level not in self._breakdown_cache:
            self.tables_ahead()
            self._breakdown_cache[level] = self.breakdown_of_table(level, line=True)
        return self._breakdown_cache[level]

    def barrier_and_segments(
        self, r: float, ir: int, tables: list[np.ndarray], level: float, loading_set: dict
    ) -> _BarrierAndSegments:
        """
        Calculate the part of 'breakdown_of_table' per direction that does not depend on the combination:
        BIJDRAGEKERING per cell and the wind segments for BIJDRAGENWIND.

        Parameters
        ----------
        r : float
            The wind direction
        ir : int
            The index of the wind direction in the statistics
        tables : list[np.ndarray]
            The response tables of input IK 1 and 2
        level : float
            The level, already shifted for the combination
        loading_set : dict
            The loading per (direction, input IK)

        Returns
        -------
        _BarrierAndSegments
            BIJDRAGEKERING and the prepared wind segments
        """
        alpha = self._alpha
        s1, s2 = (
            compiled_wind_segments.segments(
                loading_set[(r, ik)]["u"], loading_set[(r, ik)]["m"], loading_set[(r, ik)]["q"], table, self._mblock,
                self._qblock, level,
            )
            for ik, table in zip((1, 2), tables, strict=True)
        )
        closing_wind = self._closing_wind[r]
        closes = closing_wind >= 0.0
        b = closing_wind[:, :, None]
        s2_below = (np.where(s2[0] < b, s2[0], np.nan), np.minimum(s2[1], b))
        s1_above = (np.where(s1[1] > b, np.maximum(s1[0], b), np.nan), s1[1])
        uk1 = np.where(np.isnan(s1[0][:, :, 0]), self._u_max, s1[0][:, :, 0])
        ovk_tmp = self._epu[:, ir] - self._epu[-1, ir]

        def probability_above(x):
            """
            Calculate P(U >= x) relative to the end of the statistics grid.

            Parameters
            ----------
            x : np.ndarray
                The wind speeds

            Returns
            -------
            np.ndarray
                The exceedance probabilities
            """
            return HydraInterpolate.inextrp1d_fractions(self._u_stat, ovk_tmp, x)

        # BIJDRAGEKERING, vectorised over (m, q)
        p2 = HydraIJsselVechtDeltaPercentiles.segment_probability(*s2, probability_above)
        b_open = np.where(
            closes,
            alpha * p2
            + np.where(
                uk1 < closing_wind,
                (1.0 - alpha) * HydraIJsselVechtDeltaPercentiles.segment_probability(*s2_below, probability_above),
                0.0,
            ),
            p2,
        )
        b_closed = np.where(
            closes,
            (1.0 - alpha) * HydraIJsselVechtDeltaPercentiles.segment_probability(*s1_above, probability_above),
            0.0,
        )
        b_open = np.maximum(b_open, 0.0)
        b_closed = np.maximum(b_closed, 0.0)

        u_top = self._u_breakdown[-1]
        prepared = [
            HydraIJsselVechtDeltaPercentiles.wind_segments_prepare(self._u_breakdown, *seg, u_top)
            for seg in (s2, s2_below, s1_above)
        ]
        return _BarrierAndSegments(b_open, b_closed, closes, closing_wind, uk1, *prepared)

    def breakdown_of_table(self, level: float, line: bool) -> dict:
        """
        Calculate 'breakdown' on '_loading_line' or on Hydra-NL's repaired '_loading_repaired'.

        The chain is Hydra-NL's (BLOKKENREKENEN_VIJ, VERWERKGOLF, FREQDICHTHQM, SPLITSENRD), but with wind segments
        instead of one characteristic wind speed per cell, as in 'frequency'. Per cell and direction S1 and S2 are the
        wind segments with h >= level of the functioning (input IK = 1) and the failing barrier (IK = 2), and B the
        closing wind speed. BIJDRAGEKERING and BIJDRAGENWIND are then:

            closed = (1 - alpha) * P(S1 and [B, inf))
            open   = alpha * P(S2) + (1 - alpha) * P(S2 and [0, B))   if u_karak1 < B (BIJDRAGENWIND: <=), else alpha * P(S2)
            open   = P(S2), closed = 0                               if the barrier does not close (B < 0)

        with u_karak1 the begin of the first segment of S1. For a rising table S = [u_karak, inf) and this is exactly
        Hydra-NL's formula, except where u_karak2 > B with u_karak1 < B: there Hydra-NL counts
        P(U >= u_karak2) - (1 - alpha) * P(U >= B) (which can be negative), here alpha * P(S2).

        Parameters
        ----------
        level : float
            The level
        line : bool
            True for '_loading_line', False for '_loading_repaired'

        Returns
        -------
        dict
            The breakdown arrays, as 'breakdown'
        """
        loading_set = self._loading_line if line else self._loading_repaired
        nm, nq, nu = len(self._mblock), len(self._qblock), len(self._u_breakdown)
        nr = len(self._directions)
        freqQK = np.zeros((nq, 2))
        freqMK = np.zeros((nm, 2))
        freqRK = np.zeros((nr, 2))
        freqUK = np.zeros((nu, 2))
        freqURK = np.zeros((nu, nr, 2))
        freqUR = np.zeros((nu, nr))
        alpha = self._alpha
        kept: dict = {}
        previous: dict = {}

        for ic, (dh, f_hs, _f_ts, p_unc) in enumerate(self._combinations):
            if p_unc <= 0.0:
                continue
            lvl = {"h": level - dh, "hs": level / f_hs, "hbn": level}[self.response]

            # Block probability P(H >= h | q, m), SCHALING(s, k) and the joint frequency density f(m, q)
            density_mq = self.frequency_density_mq(
                self.process_wave(self.block_probability(ic, lvl, line, kept))[1]
            )
            # SPLITSENRD per direction (ND = 1, KANSD = 1): BIJDRAGEKERING and BIJDRAGENWIND
            for jr, r in enumerate(self._directions):
                ir = int(np.argmin(np.abs(self._r_stat - r)))
                if self._p_r[ir] <= 0.0:
                    continue
                w = density_mq * self._dm[:, None] * self._dq[None, :] * (self._p_r[ir] * p_unc)
                # What does not depend on w follows from the two tables and the level. The combinations run with the
                # wave inside, so in the lee sector successive combinations share their tables: per direction only the
                # previous case is kept (keeping everything costs about 4 MB per pair of tables)
                tables = [self.response_table(ic, r, ik, line) for ik in (1, 2)]
                key = (id(tables[0]), id(tables[1]), lvl)
                if r not in previous or previous[r][0] != key:
                    previous[r] = (key, self.barrier_and_segments(r, ir, tables, lvl, loading_set))
                barrier = previous[r][1]

                freqQK[:, 0] += (w * barrier.b_open).sum(axis=0)
                freqQK[:, 1] += (w * barrier.b_closed).sum(axis=0)
                freqMK[:, 0] += (w * barrier.b_open).sum(axis=1)
                freqMK[:, 1] += (w * barrier.b_closed).sum(axis=1)
                freqRK[jr, 0] += (w * barrier.b_open).sum()
                freqRK[jr, 1] += (w * barrier.b_closed).sum()

                # BIJDRAGENWIND: per (m, q) cell the weight on the wind speeds within the segments
                density = self._density_u[:, ir] * self._du_breakdown
                closes = barrier.closes
                contribution_open = density * (
                    HydraIJsselVechtDeltaPercentiles.wind_segments(
                        barrier.segments_s2, w * np.where(closes, alpha, 1.0)
                    )
                    + HydraIJsselVechtDeltaPercentiles.wind_segments(
                        barrier.segments_s2_below,
                        w * np.where(closes & (barrier.uk1 <= barrier.closing_wind), 1.0 - alpha, 0.0),
                    )
                )
                contribution_closed = density * HydraIJsselVechtDeltaPercentiles.wind_segments(
                    barrier.segments_s1_above, w * np.where(closes, 1.0 - alpha, 0.0)
                )

                freqURK[:, jr, 0] += contribution_open
                freqURK[:, jr, 1] += contribution_closed
                freqUK[:, 0] += contribution_open
                freqUK[:, 1] += contribution_closed
                freqUR[:, jr] += contribution_open + contribution_closed

        return {
            "freqQK": freqQK,
            "freqMK": freqMK,
            "freqRK": freqRK,
            "freqUK": freqUK,
            "freqURK": freqURK,
            "freqUR": freqUR,
            "q": self._qblock,
            "m": self._mblock,
            "u": self._u_breakdown,
            "r": np.asarray(self._directions, float),
        }

    # The frequency line

    def frequency(self, level: float) -> float:
        """
        Calculate the frequency line at a level (per winter half year) along Hydra-NL's own route, without the wind
        repair.

        'rekenenvak_VIJ.f90': per uncertainty combination the block probability and VERWERKGOLF; the trapezium
        probabilities OVKANS_KS times KANSONZ summed (VAR_H_AFH, the error equal for the whole wave);
        COMBINEERGOLVEN_VIJ = sum of DICHTH_MQPIEK * OVKANS_KS * dm * dq over the peaks; times NGEGEVENSBLOK. The block
        probability is the sum over wind segments on '_loading_line', instead of Hydra-NL's characteristic wind speed
        on the table repaired in the wind.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        float
            The frequency per winter half year
        """
        self.tables_ahead()
        return self.frequency_of_table(level, line=True)

    def line(
        self, requested_frequencies: list[float], progress: Callable[[int], None] | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Calculate the frequency line on Hydra-NL's own levels ('Scaling.generate_levels'). Start value 4 m and search
        step 1 m (as in Hydra-NL for the IJssel-Vecht delta); grid step 5 cm with capping, otherwise 10 cm.

        Parameters
        ----------
        requested_frequencies : list[float]
            The frequencies the user asked for
        progress : Callable[[int], None] | None, optional
            Called before every new level with the number of levels calculated so far. A caller that wants to stop
            raises an exception from it. Default is None.

        Returns
        -------
        tuple[list[float], list[float]]
            The levels and the frequencies
        """
        calculated: dict[float, float] = {}

        def frequency(h: float) -> float:
            """
            Return the kept frequency at a level, or calculate and keep it.

            Parameters
            ----------
            h : float
                The level

            Returns
            -------
            float
                The frequency
            """
            if h not in calculated:
                if progress is not None:
                    progress(len(calculated))
                calculated[h] = self.frequency(h)
            return calculated[h]

        levels = Scaling.generate_levels(
            frequency,
            requested_frequencies,
            hstart=4.0,
            hstep=1.0,
            hd=0.05 if self._capped else 0.1,
            fmax=self._n_data_blocks * self._probability_mass,
        )
        return levels, [frequency(h) for h in levels]

    def frequency_of_table(self, level: float, line: bool) -> float:
        """
        Calculate 'frequency' on '_loading_line' or on Hydra-NL's repaired '_loading_repaired'.

        Parameters
        ----------
        level : float
            The level
        line : bool
            True for '_loading_line', False for '_loading_repaired'

        Returns
        -------
        float
            The frequency per winter half year
        """
        if not line and not self._loading_repaired:
            raise RuntimeError("[ERROR] Hydra-NL's repaired loading table only exists with 'with_repaired_loading=True'.")
        level = float(level)
        ovkans_ks = np.zeros_like(self._peak)
        kept: dict = {}
        for ic, (dh, f_hs, _f_ts, p_unc) in enumerate(self._combinations):
            if p_unc <= 0.0:
                continue
            lvl = {"h": level - dh, "hs": level / f_hs, "hbn": level}[self.response]
            ovkans_ks += (
                compiled_wind_segments.trapezium_probability_on_waves(
                    self.block_probability(ic, lvl, line, kept),
                    self._m_index,
                    self._m_fraction,
                    self._q_index,
                    self._q_fraction,
                    self._block_factors,
                    self._times[1] - self._times[0],
                    self._block_duration_wind,
                )
                * p_unc
            )
        trapezium_probability = float((self._peak * ovkans_ks * self._dmp[:, None] * self._dqp[None, :]).sum())
        return self._n_data_blocks * trapezium_probability

    def breakdown_scaled(self, level: float, target_frequency: float) -> dict:
        """
        Calculate 'breakdown' with SCHAALUITSPLITS: the families put on the frequency line ('Scaling.scale_breakdown').
        Use this as soon as an absolute number is taken from the breakdown; for percentiles and percentages it makes no
        difference.

        Parameters
        ----------
        level : float
            The level
        target_frequency : float
            The frequency line at this level

        Returns
        -------
        dict
            The scaled breakdown
        """
        return Scaling.scale_breakdown(self.breakdown(level), target_frequency)

    # Output tables

    def percentiles(self, level: float) -> dict:
        """
        Calculate the T00x_UITSPL output: percentiles of discharge, lake level and wind speed (open and closed, open,
        closed) plus the percentages of the open and closed barrier.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The output of 'Percentiles.breakdown_percentiles'
        """
        b = self.breakdown(level)
        return Percentiles.breakdown_percentiles(
            b["freqRK"],
            {
                "discharge": (b["q"], b["freqQK"]),
                "lake_level": (b["m"], b["freqMK"]),
                "wind_speed": (b["u"], b["freqUK"]),
            },
        )

    def wind_percentiles_per_direction(self, level: float, barrier_state: str | None = None) -> dict:
        """
        Calculate the UR table: per wind direction the percentage and the wind speed percentiles. A closed barrier only
        exists in the western sector.

        Parameters
        ----------
        level : float
            The level
        barrier_state : str | None, optional
            None for the UR table, "open" for the UR1 table and "closed" for the UR2 table. Default is None.

        Returns
        -------
        dict
            The output of 'Percentiles.wind_percentiles_per_direction'
        """
        return Percentiles.wind_percentiles_per_direction(self.breakdown(level), barrier_state)

    def direction_contributions(self, level: float) -> dict:
        """
        Split the illustration point tables over the open and closed barrier: per closing situation its percentage and
        the percentages of the exceedance frequency per wind direction.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The output of 'Percentiles.direction_contributions'
        """
        return Percentiles.direction_contributions(self.breakdown(level))

    @staticmethod
    def threshold_order(u: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sort the thresholds and count per u the number of thresholds <= u. With 'sum_in_order' this gives per u the sum
        of the weights of all cells with threshold <= u: the same as '(u[:, None] >= threshold[None, :]) @ weight', but
        by sorting once and summing cumulatively instead of a matrix of wind speeds times cells. The order of summation
        differs, so the last digits can differ (order 1e-15 relative). A threshold NaN counts nowhere.

        Parameters
        ----------
        u : np.ndarray
            The wind speeds
        threshold : np.ndarray
            The thresholds

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The sort order of the thresholds and per u the number of thresholds <= u
        """
        order = np.argsort(threshold, kind="stable")
        # NaN sorts last and is never counted by searchsorted
        return order, np.searchsorted(threshold[order], u, side="right")

    @staticmethod
    def sum_in_order(order: np.ndarray, count: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Sum per u the weights of the first 'count' thresholds in sort order.

        Parameters
        ----------
        order : np.ndarray
            The sort order of the thresholds, from 'threshold_order'
        count : np.ndarray
            Per u the number of thresholds <= u, from 'threshold_order'
        weight : np.ndarray
            The weight per threshold

        Returns
        -------
        np.ndarray
            The summed weight per u
        """
        return np.concatenate(([0.0], np.cumsum(weight[order])))[count]

    @staticmethod
    def segment_probability(begin: np.ndarray, end: np.ndarray, probability_above) -> np.ndarray:
        """
        Sum per cell over the wind segments P(U >= begin) - P(U >= end); a segment up to the edge (end = inf) counts
        P(U >= begin), a segment with begin NaN does not count.

        Parameters
        ----------
        begin, end : np.ndarray
            The begin and end of the segments, with the segments on the last axis
        probability_above : Callable
            P(U >= x)

        Returns
        -------
        np.ndarray
            The probability per cell
        """
        valid = ~np.isnan(begin)
        p_begin = probability_above(np.where(valid, begin, 0.0))
        p_end = np.where(np.isfinite(end), probability_above(np.where(np.isfinite(end), end, 0.0)), 0.0)
        return np.where(valid, p_begin - p_end, 0.0).sum(axis=-1)

    @staticmethod
    def wind_segments_prepare(u: np.ndarray, begin: np.ndarray, end: np.ndarray, u_top: float) -> tuple:
        """
        Prepare what 'wind_segments' needs and does not depend on the weight. A segment that only begins at U(NU) does
        not count (Hydra-NL's guard on U(NU)); a segment with begin NaN neither.

        Parameters
        ----------
        u : np.ndarray
            The wind speeds of the breakdown
        begin, end : np.ndarray
            The begin and end of the segments per cell, with the segments on the last axis
        u_top : float
            The highest wind speed of the breakdown, U(NU)

        Returns
        -------
        tuple
            Which segments count, and the sort orders and counts of the begins and ends
        """
        return (
            begin < u_top,
            *HydraIJsselVechtDeltaPercentiles.threshold_order(u, begin.ravel()),
            *HydraIJsselVechtDeltaPercentiles.threshold_order(u, end.ravel()),
        )

    @staticmethod
    def wind_segments(prepared: tuple, weight: np.ndarray) -> np.ndarray:
        """
        Sum per wind speed u_i the weights (per cell) of all segments with begin <= u_i < end.

        Parameters
        ----------
        prepared : tuple
            The result of 'wind_segments_prepare'
        weight : np.ndarray
            The weight per (m, q) cell

        Returns
        -------
        np.ndarray
            The summed weight per wind speed
        """
        counts, order_begin, count_begin, order_end, count_end = prepared
        w = np.where(counts, weight[:, :, None], 0.0).ravel()
        return HydraIJsselVechtDeltaPercentiles.sum_in_order(
            order_begin, count_begin, w
        ) - HydraIJsselVechtDeltaPercentiles.sum_in_order(order_end, count_end, w)
