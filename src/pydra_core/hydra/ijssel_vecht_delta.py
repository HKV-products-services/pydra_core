"""
The Hydra-NL illustration point for the IJssel-Vecht delta (REGIO 1): a 3D loading field with the Ramspol barrier. The
joint density, the barrier and the closing situations are described in docs/hydra_extensions.qmd.
"""

import csv
import numpy as np
import sqlite3

from collections.abc import Callable
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from .illustration_points_3d import HydraIllustrationPoint3D
from .library import Grid, HydraInterpolate


class BarrierClosingData:
    """
    The KeringUD data: per (wind direction, wind speed, discharge, lake level) whether the closing criterion of the
    Ramspol barrier is met (value >= 0) or not (-9).

    The source is the table 'KeringUD' in the database if it is there and not empty; otherwise the CSV of the region in
    'hydra_ini_directory', as Hydra-NL itself writes it into the database when it opens it the first time.
    """

    def __init__(self, database_path: str | Path, hydra_ini_directory: str | Path | None = None):
        """
        Read the KeringUD data from the database, or from the CSV of its region.

        Parameters
        ----------
        database_path : str or Path
            The path of the HRD database
        hydra_ini_directory : str or Path (optional)
            The folder 'data/ini' of the Hydra-NL installation, with the KeringUD CSV of the region. Only read if the
            database has no KeringUD table.
        """
        con = sqlite3.connect(str(database_path))
        try:
            has_table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='KeringUD'"
            ).fetchone()
            rows = (
                con.execute("SELECT Windrichting, Windsnelheid, Afvoer, Meerpeil, KeringUD FROM KeringUD").fetchall()
                if has_table
                else []
            )
        finally:
            con.close()
        if not rows:
            csv_path = BarrierClosingData.find_barrier_csv(database_path, hydra_ini_directory)
            if csv_path is None:
                raise NotImplementedError(
                    "[ERROR] The IJssel-Vecht delta needs the KeringUD closing data of the Ramspol barrier. This database has "
                    "no KeringUD table, and no KeringUD CSV was found for its region. Pass the folder 'data/ini' of "
                    "the Hydra-NL installation as 'hydra_ini_directory', or open the database once in Hydra-NL so "
                    "that it writes the table."
                )
            rows = BarrierClosingData.read_barrier_csv(csv_path)
        arr = np.asarray(rows, dtype=float)
        self.r = np.unique(arr[:, 0])
        self.u = np.unique(arr[:, 1])
        self.q = np.unique(arr[:, 2])
        self.m = np.unique(arr[:, 3])
        self.values = np.full((len(self.u), len(self.m), len(self.q), len(self.r)), np.nan)
        iu = np.searchsorted(self.u, arr[:, 1])
        iq = np.searchsorted(self.q, arr[:, 2])
        im = np.searchsorted(self.m, arr[:, 3])
        ir = np.searchsorted(self.r, arr[:, 0])
        self.values[iu, im, iq, ir] = arr[:, 4]

    def closing_wind_speed(self, r: float) -> np.ndarray:
        """
        Determine SLUITWIND per database node (m, q): 0 (always closes), -1 (never closes), or the mean of the smallest
        wind speed at which the criterion is met and the largest wind speed below it at which it is not (BLOKSNELHEID,
        'sluitfreq_VIJ.f90').

        Parameters
        ----------
        r : float
            The wind direction

        Returns
        -------
        np.ndarray
            The closing wind speed per database node, with the shape (m, q)
        """
        ir = int(np.argmin(np.abs(self.r - r)))
        k = self.values[:, :, :, ir]
        nm, nq = k.shape[1], k.shape[2]
        sw = np.empty((nm, nq))
        for iq in range(nq):
            for im in range(nm):
                col = k[:, im, iq]
                if col[0] >= 0.0:
                    sw[im, iq] = 0.0
                elif col[1:].max() < 0.0:
                    sw[im, iq] = -1.0
                else:
                    du = next(i for i in range(1, len(col)) if col[i] >= 0.0)
                    sw[im, iq] = -1.0
                    for iu in range(du - 1, -1, -1):
                        if -99.0 < col[iu] < 0.0:
                            sw[im, iq] = (self.u[du] + self.u[iu]) / 2.0
                            break
        return sw

    @staticmethod
    def find_barrier_csv(database_path: str | Path, hydra_ini_directory: str | Path | None) -> Path | None:
        """
        Find the KeringUD CSV of Hydra-NL that belongs to this database, in the folder 'data/ini' of the Hydra-NL
        installation. The region (IJsseldelta or Vechtdelta, both the Ramspol barrier) comes from the file name of the
        database.

        The file name differs per Hydra-NL version, 'KeringUD IJsseldelta 2017.csv' with spaces and
        'KeringUD_IJsseldelta_BOI2023.csv' with underscores, while the contents are the same. Every file that starts
        with 'KeringUD' is therefore considered, and the region is matched with the separators removed.

        Parameters
        ----------
        database_path : str | Path
            The path of the HRD database
        hydra_ini_directory : str | Path | None
            The folder 'data/ini' of the Hydra-NL installation

        Returns
        -------
        Path | None
            The path of the CSV; None if there is no folder or no CSV for the region
        """
        name = str(database_path).lower()
        region = "IJsseldelta" if "ijsseldelta" in name else ("Vechtdelta" if "vechtdelta" in name else None)
        if region is None or hydra_ini_directory is None:
            return None
        directory = Path(hydra_ini_directory)
        if not directory.exists():
            return None

        def flat(text: str) -> str:
            """
            Remove the case and the separators from a name.

            Parameters
            ----------
            text : str
                The name

            Returns
            -------
            str
                The name in lower case without '_', ' ' and '-'
            """
            return text.lower().replace("_", "").replace(" ", "").replace("-", "")

        candidates = sorted(p for p in directory.glob("KeringUD*.csv") if flat(region) in flat(p.name))
        return candidates[0] if candidates else None

    @staticmethod
    def read_barrier_csv(path: str | Path) -> list[tuple[float, float, float, float, float]]:
        """
        Read the KeringUD CSV (columns Windrichting;Afvoer;Meerpeil;Windsnelheid;KeringUD, separated by ';', decimal
        comma).

        Parameters
        ----------
        path : str | Path
            The path of the CSV

        Returns
        -------
        list[tuple[float, float, float, float, float]]
            The rows (r, u, q, m, value), the same order as the query on the database table
        """
        out: list[tuple[float, float, float, float, float]] = []
        with open(path, newline="", encoding="latin-1") as f:
            reader = csv.reader(f, delimiter=";")
            # Header
            next(reader, None)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                r, q, m, u, value = (float(x.replace(",", ".")) for x in row[:5])
                out.append((r, u, q, m, value))
        return out

    @staticmethod
    def closing_wind_speed_on_grid(
        data: "BarrierClosingData", r: float, mblock: np.ndarray, qblock: np.ndarray
    ) -> np.ndarray:
        """
        Determine the closing wind speed on the (lake level, discharge) block grid (BLOKSNELHEID), with the explicit
        logic of the 2x2 cases around 'infinite' (-1) of 'sluitfreq_VIJ.f90'. Outside the range of the database the edge
        is taken.

        Parameters
        ----------
        data : BarrierClosingData
            The KeringUD data
        r : float
            The wind direction
        mblock, qblock : np.ndarray
            The block grids of the lake level and the discharge

        Returns
        -------
        np.ndarray
            The closing wind speed, with the shape (len(mblock), len(qblock)); -1 where the barrier never closes
        """
        sw = data.closing_wind_speed(r)
        mdb, qdb = data.m, data.q

        def interpol(x1, x2, y1, y2, x):
            """
            Interpolate linearly between two points (INTERPOL).

            Parameters
            ----------
            x1, x2 : float
                The coordinates of the two points
            y1, y2 : float
                The values at the two points
            x : float
                The coordinate to interpolate at

            Returns
            -------
            float
                The interpolated value
            """
            return y1 + (x - x1) / (x2 - x1) * (y2 - y1)

        def interpol2(m1, m2, q1, q2, v11, v21, v12, v22, m, q):
            """
            Interpolate bilinearly between four points (INTERPOL2).

            Parameters
            ----------
            m1, m2 : float
                The lake levels of the points
            q1, q2 : float
                The discharges of the points
            v11, v21, v12, v22 : float
                The values at (m1, q1), (m2, q1), (m1, q2) and (m2, q2)
            m, q : float
                The point to interpolate at

            Returns
            -------
            float
                The interpolated value
            """
            # INTERPOL2(M1, M2, Q1, Q2, U(M1,Q1), U(M2,Q1), U(M1,Q2), U(M2,Q2), LM, LQ): bilinear
            a = interpol(m1, m2, v11, v21, m)
            b = interpol(m1, m2, v12, v22, m)
            return interpol(q1, q2, a, b, q)

        out = np.empty((len(mblock), len(qblock)))
        for jq, qv in enumerate(qblock):
            lq = min(max(qv, qdb[0]), qdb[-1])
            dq = len(qdb) - 1
            for i in range(1, len(qdb) - 1):
                if qdb[i] >= lq:
                    dq = i
                    break
            for jm, mv in enumerate(mblock):
                lm = min(max(mv, mdb[0]), mdb[-1])
                dm = len(mdb) - 1
                for i in range(1, len(mdb) - 1):
                    if mdb[i] >= lm:
                        dm = i
                        break
                middle_m = (mdb[dm - 1] + mdb[dm]) / 2.0
                middle_q = (qdb[dq - 1] + qdb[dq]) / 2.0
                # The four database nodes (m-1, q-1), (m-1, q), (m, q-1) and (m, q)
                s11, s12 = sw[dm - 1, dq - 1], sw[dm - 1, dq]
                s21, s22 = sw[dm, dq - 1], sw[dm, dq]
                if max(s11, s12, s21, s22) < 0.0:
                    u = -1.0
                elif min(s11, s12, s21, s22) >= 0.0:
                    u = interpol2(mdb[dm - 1], mdb[dm], qdb[dq - 1], qdb[dq], s11, s21, s12, s22, lm, lq)
                elif max(s11, s12, s22) < 0.0:
                    # Only (m, q-1) finite
                    u = s21 if (lm >= middle_m and lq <= middle_q) else -1.0
                elif s11 < 0.0 and s12 < 0.0:
                    # Upper row (m-1) infinite, lower row (m) finite
                    if mv >= middle_m:
                        u = interpol(qdb[dq - 1], qdb[dq], s21, s22, lq)
                    else:
                        u = -1.0
                elif s12 < 0.0 and s22 < 0.0:
                    # Right column (q) infinite, left column (q-1) finite
                    if qv <= middle_q:
                        u = interpol(mdb[dm - 1], mdb[dm], s11, s21, lm)
                    else:
                        u = -1.0
                elif s12 < 0.0:
                    # Only (m-1, q) infinite
                    if lm < middle_m and lq > middle_q:
                        u = -1.0
                    elif lq > middle_q:
                        ucent = (s11 + s22) / 2.0
                        umidm = s22
                        umidq = (s21 + s22) / 2.0
                        u = interpol2(middle_m, mdb[dm], middle_q, qdb[dq], ucent, umidm, umidq, s22, lm, lq)
                    elif lm < middle_m:
                        ucent = (s11 + s22) / 2.0
                        umidm = (s11 + s21) / 2.0
                        umidq = s11
                        u = interpol2(mdb[dm - 1], middle_m, qdb[dq - 1], middle_q, s11, umidq, umidm, ucent, lm, lq)
                    else:
                        ucent = (s11 + s22) / 2.0
                        umidm = (s11 + s21) / 2.0
                        umidq = (s21 + s22) / 2.0
                        u = interpol2(middle_m, mdb[dm], qdb[dq - 1], middle_q, umidm, ucent, s21, umidq, lm, lq)
                else:
                    # The remaining cases: infinite nodes replaced by the highest wind speed of the database
                    def finite(v):
                        """
                        Replace an infinite closing wind speed by the highest wind speed of the database.

                        Parameters
                        ----------
                        v : float
                            The closing wind speed

                        Returns
                        -------
                        float
                            The finite closing wind speed
                        """
                        return data.u[-1] if (-99.0 < v < 0.0) else v

                    u = interpol2(
                        mdb[dm - 1], mdb[dm], qdb[dq - 1], qdb[dq], finite(s11), finite(s21), finite(s12), finite(s22),
                        lm, lq,
                    )
                out[jm, jq] = u
        return out


class HydraIJsselVechtDeltaIllustrationPoint(HydraIllustrationPoint3D):
    """
    The Hydra-NL illustration point for one location in the IJssel-Vecht delta, with the Ramspol barrier.

    A location without the slow stochastics lake level and discharge and the closing situations 1 and 2 is refused
    with a 'NotImplementedError'.
    """

    # The lake level is called 'a' in the loading table
    _level_axis: str = "a"

    # The same table as the frequency line and the breakdown: without Hydra-NL's repair after the extension to UMAX
    _hydra_repairs: bool = False

    # Only candidate points on the limit state: no connecting segments between the parts of an isoline
    _isoline_per_part: bool = True

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
        hydra_ini_directory: str | Path | None = None,
    ):
        """
        Prepare the illustration point for one location in the IJssel-Vecht delta.

        Parameters
        ----------
        hrd : HRDatabase
            The database of the location
        settings : Settings
            The settings of the location, as the caller wants to calculate with them. They are copied, not changed.
        statistical_uncertainty : bool
            Whether the statistics include the statistical uncertainty
        model_uncertainty : bool
            Whether the model uncertainty adds dimensions to the distance
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
        """
        self._hydra_ini_directory = hydra_ini_directory
        super().__init__(
            hrd, settings, statistical_uncertainty, model_uncertainty, response, profile, qcrit, crest_level_table
        )

    def check(self, stats) -> None:
        """
        Refuse a location without the slow stochastics lake level and discharge and the closing situations 1 and 2.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        if set(stats.stochastics_slow) != {"a", "q"} or sorted(stats.barrier.k) != [1, 2]:
            raise NotImplementedError(
                f"[ERROR] {self.location} is not a location of the IJssel-Vecht delta: the illustration point expects the slow "
                f"stochastics lake level and discharge ('a', 'q') and the closing situations 1 and 2 of the Ramspol "
                f"barrier."
            )

    def statistics_grids(self, stats) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the block grids of the lake level and the discharge.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The block grids of the lake level and the discharge
        """
        # Lake level and discharge are both slow, and pydra keeps block axes for them
        return np.asarray(stats.lake_level.ablok, float), np.asarray(stats.discharge.qblok, float)

    def prepare(self, stats) -> None:
        """
        Prepare the failure probability of the barrier, the joint density of lake level and discharge and the KeringUD
        data.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        # The failure probability of the barrier, Hydra-NL's density of lake level and discharge, and the KeringUD data
        self._failure_probability = float(getattr(self._settings, "failure_probability_ramspol", 0.01) or 0.01)
        self._density_mq = HydraIJsselVechtDeltaIllustrationPoint.joint_density_lake_level_discharge(stats)
        self._barrier = BarrierClosingData(self._settings.database_path, self._hydra_ini_directory)
        self._closing_wind_cache: dict[int, np.ndarray] = {}

    def after_block_grids(self) -> None:
        """
        Interpolate the joint density of lake level and discharge to the block grids.
        """
        # GRIDINTERP2D of the density to the block grids; usually they equal the statistics grids and nothing happens
        if (
            len(self._mblock) != len(self._mstat)
            or len(self._qblock) != len(self._qstat)
            or not (np.allclose(self._mblock, self._mstat) and np.allclose(self._qblock, self._qstat))
        ):
            gi = RegularGridInterpolator(
                (self._mstat, self._qstat), self._density_mq, method="linear", bounds_error=False, fill_value=None
            )
            MM, QQ = np.meshgrid(self._mblock, self._qblock, indexing="ij")
            self._density_mq = gi(np.column_stack([MM.ravel(), QQ.ravel()])).reshape(MM.shape)

    def joint_umqk(self, ir: int) -> np.ndarray:
        """
        Calculate the joint probability density over (u, m, q) per barrier state: g(u | r) g(m, q) P(k | u, m, q, r),
        with g(m, q) Hydra-NL's DICHTH_MQBLOK, the wind conditionally independent given the direction, and the Ramspol
        barrier as a deterministic closing criterion (the closing wind speed) times a constant failure probability.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The joint probability density, with the shape (nu, nm, nq, 2)
        """
        nu, nm, nq = len(self._ublock), len(self._mblock), len(self._qblock)
        density_u = HydraInterpolate.interp1d_clamped(
            self._u_stat, Grid.probability_density(self._u_stat, self._epu[:, ir]), self._ublock
        )
        closing_wind = self.closing_wind_speed(ir)

        joint = np.zeros((nu, nm, nq, 2))
        base = density_u[:, None, None]
        closes = (closing_wind >= 0.0)[None, :, :] & (self._ublock[:, None, None] > closing_wind[None, :, :])
        joint[..., 0] = np.where(closes, self._failure_probability * base, base)
        joint[..., 1] = np.where(closes, (1.0 - self._failure_probability) * base, 0.0)
        # Times g(m, q) and g(r)
        joint *= self._density_mq[None, :, :, None] * self._p_r[ir]
        return joint

    def closing_wind_speed(self, ir: int) -> np.ndarray:
        """
        Return the closing wind speed (BLOKSNELHEID) on the (lake level, discharge) block grid: the wind speed above
        which the closing criterion of the Ramspol barrier is met.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The closing wind speed, with the shape (m, q); -1 where the barrier never closes
        """
        if ir not in self._closing_wind_cache:
            r = float(self._directions[ir])
            self._closing_wind_cache[ir] = BarrierClosingData.closing_wind_speed_on_grid(
                self._barrier, r, self._mblock, self._qblock
            )
        return self._closing_wind_cache[ir]

    @staticmethod
    def lin(x1, x2, y1, y2, x):
        """
        Interpolate or extrapolate linearly between two points (DINTERPOL).

        Parameters
        ----------
        x1, x2 : float
            The coordinates of the two points
        y1, y2 : float
            The values at the two points
        x : float
            The coordinate to interpolate at

        Returns
        -------
        float
            The interpolated value
        """
        return y1 + (x - x1) / (x2 - x1) * (y2 - y1)

    @staticmethod
    def crossing_times(times: np.ndarray, wave: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Determine the crossing times of a wave (TIJDSTIPPENGOLF, 'overschrijdingstijdstippen.f90'), vectorised over the
        levels.

        The comparison is in float32, as Hydra-NL's REAL(4): at levels exactly on a plateau or peak of the wave the
        comparison flips otherwise, which shows in OVKANSMQ.

        Parameters
        ----------
        times : np.ndarray
            The times of the wave shape
        wave : np.ndarray
            The wave shape
        levels : np.ndarray
            The levels

        Returns
        -------
        tuple[np.ndarray, ...]
            Per level the first and last time at which the wave crosses it (linear on the continuous wave shape), and
            whether the wave is above the level (1) or not (-1) at the begin, middle and end of the crossing
        """
        nt = len(times)
        above = np.where(np.float32(wave)[None, :] >= np.float32(levels)[:, None], 1, -1)
        crossing = (above[:, :-1] * above[:, 1:]) == -1
        has_crossing = crossing.any(axis=1)
        s1 = np.full(len(levels), float(times[0]))
        s2 = np.full(len(levels), float(times[-1]))
        begin = np.full(len(levels), -1, dtype=np.int8)
        end = np.full(len(levels), -1, dtype=np.int8)
        middle = above[:, 0].astype(np.int8)
        for j in np.where(has_crossing)[0]:
            # The first crossing block, and the upper index of the last one
            i1 = int(np.argmax(crossing[j]))
            i2 = nt - 1 - int(np.argmax(crossing[j, ::-1]))
            s1[j] = HydraIJsselVechtDeltaIllustrationPoint.lin(
                wave[i1], wave[i1 + 1], times[i1], times[i1 + 1], levels[j]
            )
            s2[j] = HydraIJsselVechtDeltaIllustrationPoint.lin(
                wave[i2 - 1], wave[i2], times[i2 - 1], times[i2], levels[j]
            )
            begin[j], middle[j], end[j] = above[j, i1], above[j, i1 + 1], above[j, i2]
        return s1, s2, begin, middle, end

    @staticmethod
    def joint_durations(
        crossing_m: tuple[np.ndarray, ...], s1q: np.ndarray, s2q: np.ndarray, middle_q: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the joint duration above (lake level, discharge) for one lake level peak wave (OVERSCHRIJDINGSDUREN).

        Parameters
        ----------
        crossing_m : tuple[np.ndarray, ...]
            The result of 'crossing_times' for the lake level wave on the lake level grid
        s1q, s2q : np.ndarray
            The first and last crossing time per (discharge peak, discharge level)
        middle_q : np.ndarray
            Whether the discharge wave is above the level in the middle of the crossing, per (discharge peak, discharge
            level)

        Returns
        -------
        np.ndarray
            The joint duration per (lake level, discharge peak, discharge level), with the shape (nm, number of
            discharge peaks, nq)
        """
        s1m, s2m, _, middle_m, _ = crossing_m
        both = middle_m[:, None, None] == 1
        overlap = np.maximum(
            0.0,
            np.minimum(s2m[:, None, None], s2q[None, :, :]) - np.maximum(s1m[:, None, None], s1q[None, :, :]),
        )
        # The middle of the lake level wave below the level (begin and end above it): the discharge interval minus the
        # middle part
        apart = (s1m[:, None, None] > s2q[None, :, :]) | (s2m[:, None, None] < s1q[None, :, :])
        otherwise = np.where(
            apart,
            s2q[None, :, :] - s1q[None, :, :],
            np.maximum(0.0, s1m[:, None, None] - s1q[None, :, :])
            + np.maximum(0.0, s2q[None, :, :] - s2m[:, None, None]),
        )
        return np.where(middle_q[None, :, :], np.where(both, overlap, otherwise), 0.0)

    @staticmethod
    def joint_density_lake_level_discharge(stats) -> np.ndarray:
        """
        Calculate Hydra-NL's DICHTH_MQBLOK on the block grids (lake level, discharge): OVERSCHRIJDINGSDUREN, the
        OVKANSMQ sum of STAT_GEZAMENLIJK and KANSDICHTHEID twice, on pydra's peak density and wave shapes.

        The global scale factors (BLOKDUURWIND, BASISDUUR / BLOKDUUR) are left out: every probability that is derived
        from this density is normalised.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location

        Returns
        -------
        np.ndarray
            The joint density, with the shape (lake level, discharge)
        """
        mblock = np.asarray(stats.lake_level.ablok, float)
        qblock = np.asarray(stats.discharge.qblok, float)
        # Equal to Hydra-NL's DICHTH_MQPIEK
        peak = np.asarray(stats.density_aq_peak, float)
        dmp = Grid.step_sizes(np.asarray(stats.lake_level.apeak, float))
        dqp = Grid.step_sizes(np.asarray(stats.discharge.qpeak, float))

        ws_m, ws_q = stats.lake_level.wave_shape, stats.discharge.wave_shape
        waves_m = np.asarray(ws_m.wave_shapes, float)  # (nt, number of lake level peaks)
        waves_q = np.asarray(ws_q.wave_shapes, float)
        times_m = np.asarray(ws_m.time, float)
        times_q = np.asarray(ws_q.time, float)
        n_mpeak, n_qpeak = waves_m.shape[1], waves_q.shape[1]

        # Crossing times per peak wave and level
        tm = [
            HydraIJsselVechtDeltaIllustrationPoint.crossing_times(times_m, waves_m[:, i], mblock)
            for i in range(n_mpeak)
        ]
        tq = [
            HydraIJsselVechtDeltaIllustrationPoint.crossing_times(times_q, waves_q[:, i], qblock)
            for i in range(n_qpeak)
        ]
        s1q = np.stack([t[0] for t in tq])  # (number of discharge peaks, nq)
        s2q = np.stack([t[1] for t in tq])
        middle_q = np.stack([t[3] for t in tq]) == 1

        ovkansmq = np.zeros((len(mblock), len(qblock)))
        for ip in range(n_mpeak):
            w = peak[ip, :] * dqp * dmp[ip]
            ovkansmq += np.einsum(
                "k,mkq->mq", w, HydraIJsselVechtDeltaIllustrationPoint.joint_durations(tm[ip], s1q, s2q, middle_q)
            )

        # Differentiated twice: first to the lake level (per discharge column), then to the discharge (per lake level
        # row)
        between = np.empty_like(ovkansmq)
        for iq in range(ovkansmq.shape[1]):
            between[:, iq] = Grid.probability_density_raw(mblock, ovkansmq[:, iq])
        density = np.empty_like(between)
        for im in range(between.shape[0]):
            density[im, :] = Grid.probability_density_raw(qblock, between[im, :])
        return density
