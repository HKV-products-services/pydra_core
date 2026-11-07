"""
The Hydra-NL illustration point for the Rhine-Meuse delta: a 3D loading field with the Europoort barrier. The
factorisation and the easterly wind directions are described in docs/hydra_extensions.qmd.
"""

import numpy as np
import sqlite3

from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from .illustration_points_3d import HydraIllustrationPoint3D
from .library import Grid, HydraInterpolate

# The column of the sea level in 'HydroDynamicInputData' (HRDInputColumnId 4, 'Waterlevel HvH')
_SEA_LEVEL_COLUMN = 4


class HydraRhineMeuseDeltaIllustrationPoint(HydraIllustrationPoint3D):
    """
    The Hydra-NL illustration point for one location in the Rhine-Meuse delta, with the Europoort barrier.

    A location without the slow stochastic discharge and the closing situations of the Europoort barrier is refused
    with a 'NotImplementedError'. The parameters are those of 'HydraIllustrationPoint3D.__init__'.
    """

    # The level axis in the loading table is the sea level 'm'
    _level_axis: str = "m"

    # The table as pydra delivers it, without Hydra-NL's repair after the extension to UMAX
    _hydra_repairs: bool = False

    # Only candidate points on the limit state: no connecting segments between the parts of an isoline
    _isoline_per_part: bool = True

    def check(self, stats) -> None:
        """
        Refuse a location without the slow stochastic discharge and the closing situations of the Europoort barrier.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        if not HydraRhineMeuseDeltaIllustrationPoint.has_rhine_meuse_delta_statistics(stats):
            raise NotImplementedError(
                f"[ERROR] {self.location} is not a location of the Rhine-Meuse delta: the illustration point expects the slow "
                f"stochastic discharge ('q') and the closing situations 0 and 11 of the Europoort barrier."
            )

    def statistics_grids(self, stats) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the statistics grids of the sea level and the discharge.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The statistics grids of the sea level and the discharge
        """
        # The sea level and the discharge give their discretisation directly; there is no separate block axis
        return (
            np.asarray(stats.sea_level.get_discretisation(), float),
            np.asarray(stats.discharge.get_discretisation(), float),
        )

    def prepare(self, stats) -> None:
        """
        Prepare the spring tide sea level and the caches of the densities per direction.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        # The spring tide sea level for the easterly directions, and the densities per direction, which are asked for
        # once per uncertainty combination and level
        self._m_east = HydraRhineMeuseDeltaIllustrationPoint.spring_tide_sea_level(self._settings.database_path)
        self._joint_cache: dict[int, np.ndarray] = {}
        self._bracket_cache: dict[int, np.ndarray] = {}

    def after_block_grids(self) -> None:
        """
        Prepare the cell volume, the density of the discharge and the sea level block of the spring tide level.
        """
        # The cell volume that turns pydra's masses into densities, on the statistics grids
        du = Grid.step_sizes(self._u_stat)
        dm = Grid.step_sizes(self._mstat)
        dq = Grid.step_sizes(self._qstat)
        self._cell_volume = du[:, None, None] * dm[None, :, None] * dq[None, None, :]
        self._block_is_statistics = all(
            len(block) == len(stat) and bool(np.allclose(block, stat))
            for block, stat in (
                (self._ublock, self._u_stat),
                (self._mblock, self._mstat),
                (self._qblock, self._qstat),
            )
        )

        # The density of the discharge that is in pydra's joint, and that 'bracket_umqk' divides out, from the
        # instantaneous exceedance probability. Where the density is zero the mass is zero as well; dividing by infinity
        # keeps those cells at zero.
        epq_stat = np.asarray(self._stats.discharge.get_exceedance_probability(), float)
        gq_stat = Grid.probability_density_raw(np.asarray(self._qstat, float), epq_stat)
        self._gq = np.interp(self._qblock, np.asarray(self._qstat, float), gq_stat)
        self._gq_safe = np.where(self._gq > 0.0, self._gq, np.inf)

        # The sea level block of the spring tide level, where all mass of an easterly direction lies
        self._im_east = HydraRhineMeuseDeltaIllustrationPoint.block_index(self._mblock, float(self._m_east))

    def conditional_probabilities_direct(self, ir: int) -> tuple | None:
        """
        Return the Rosenblatt chain of an easterly direction, as in ONDKANSEN_ILLU_B (see docs/hydra_extensions.qmd).

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        tuple | None
            (F(q|k), F(m|q,k), F(u|m,q,k), P(k|r)) on the block grids; None for a westerly direction, whose chain comes
            from the density
        """
        if HydraIllustrationPoint3D.sector_west(float(self._directions[ir])):
            return None

        nu, nm, nq = len(self._ublock), len(self._mblock), len(self._qblock)
        fq = np.zeros((nq, 2))
        fmq = np.zeros((nm, nq, 2))
        fumq = np.zeros((nu, nm, nq, 2))

        # F(q) from the statistics grid to the block grid, as INTERP1D_D
        fq_stat = 1.0 - np.asarray(self._stats.discharge.get_exceedance_probability(), float)
        fq[:, 0] = np.interp(self._qblock, self._qstat, fq_stat)

        fmq[:, :, 0] = 0.5

        gu_stat = np.asarray(self._stats.wind_speed.get_exceedance_probability(), float)[:, ir]
        fu = np.interp(self._ublock, self._u_stat, 1.0 - gu_stat)
        fumq[:, :, :, 0] = fu[:, None, None]

        return fq, fmq, fumq, np.array([1.0, 0.0])

    def design_level(self, r: float) -> float | None:
        """
        Return the fixed sea level of the design point: at an easterly direction the spring tide level of the database.

        Parameters
        ----------
        r : float
            The wind direction

        Returns
        -------
        float | None
            The spring tide level at an easterly direction, None at a westerly direction
        """
        return None if HydraIllustrationPoint3D.sector_west(r) else self._m_east

    def joint_umqk(self, ir: int) -> np.ndarray:
        """
        Calculate the joint probability density over (u, m, q) per barrier state, from pydra, for a westerly direction.
        An easterly direction has no density over the sea level and gives a 'NotImplementedError'.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The joint probability density, with the shape (nu, nm, nq, 2)
        """
        r = float(self._directions[ir])
        if not HydraIllustrationPoint3D.sector_west(r):
            raise NotImplementedError(
                f"[ERROR] Wind direction {r} lies outside sector west, where the loading table of the Rhine-Meuse delta has no "
                f"sea level axis and there is no joint density over (u, m, q)."
            )
        if ir in self._joint_cache:
            return self._joint_cache[ir]
        joint = np.empty((len(self._ublock), len(self._mblock), len(self._qblock), 2))
        for ik, closing_situation in ((0, 1), (1, 2)):
            mass = np.asarray(self._stats.calculate_probability(r, closing_situation=closing_situation), float)
            density = np.transpose(mass, (0, 2, 1)) / self._cell_volume
            joint[..., ik] = self.to_block_grids(density)
        self._joint_cache[ir] = joint
        return joint

    def bracket_umqk(self, ir: int) -> np.ndarray:
        """
        Calculate Hydra-NL's KANS_UMQKR without the density of the discharge: P(k|u,m,q,r) f(m|r) f(u|m,r) P(r) on the
        block grid (u, m, q, k), for every wind direction. Integrated over (u, m) and summed over the barrier states and the
        directions, it gives the probability mass per discharge that the breakdown uses as SCHALING.

        Hydra-NL multiplies by the density of the discharge only once, for all directions together. Pydra's westerly
        joint has that density in it and the easterly branch built here does not, so the westerly branch divides it
        out; otherwise a later division by a vanishing discharge density blows up the easterly contribution.

        At an easterly direction all mass goes to the open state and to the sea level block of the spring tide level,
        written as a density (divided by the dm of that block), so that one integration rule with du dm dq gives the
        mass back.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            KANS_UMQKR without the density of the discharge, with the shape (nu, nm, nq, 2)
        """
        if ir in self._bracket_cache:
            return self._bracket_cache[ir]
        r = float(self._directions[ir])
        if HydraIllustrationPoint3D.sector_west(r):
            bracket = self.joint_umqk(ir) / self._gq_safe[None, None, :, None]
        else:
            bracket = np.zeros((len(self._ublock), len(self._mblock), len(self._qblock), 2))
            if self._im_east is not None:
                # The wind speed density given the direction, interpolated clamped as Hydra-NL's INTERP1D: there are
                # zeros in the exceedance probability, so no extrapolation
                exceedance = np.maximum(
                    HydraInterpolate.interp1d_clamped(self._u_stat, self._epu[:, ir], self._ublock), 0.0
                )
                density_u = Grid.probability_density_raw(self._ublock, exceedance)
                dm = Grid.step_sizes(self._mblock)
                bracket[:, self._im_east, :, 0] = density_u[:, None] * float(self._p_r[ir]) / dm[self._im_east]
        self._bracket_cache[ir] = bracket
        return bracket

    def to_block_grids(self, field: np.ndarray) -> np.ndarray:
        """
        Interpolate a field from the statistics grids to the block grids (GRIDINTERP in 3D). The block grids share the
        inner nodes of the statistics grids, with the edges clamped to the range of the loading table; if they
        coincide, nothing happens.

        Parameters
        ----------
        field : np.ndarray
            The field on the statistics grids (u, m, q)

        Returns
        -------
        np.ndarray
            The field on the block grids
        """
        if self._block_is_statistics:
            return field
        gi = RegularGridInterpolator(
            (self._u_stat, self._mstat, self._qstat), field, method="linear", bounds_error=False, fill_value=None
        )
        UU, MM, QQ = np.meshgrid(self._ublock, self._mblock, self._qblock, indexing="ij")
        return gi(np.column_stack([UU.ravel(), MM.ravel(), QQ.ravel()])).reshape(UU.shape)

    @staticmethod
    def block_index(grid: np.ndarray, value: float) -> int | None:
        """
        Find the block in which a value lies, grid[i] <= value < grid[i + 1], as Hydra-NL searches it.

        Parameters
        ----------
        grid : np.ndarray
            The block grid
        value : float
            The value

        Returns
        -------
        int | None
            The index of the block; None outside the grid, where Hydra-NL puts the mass nowhere
        """
        for i in range(len(grid) - 1):
            if grid[i] <= value < grid[i + 1]:
                return i
        return None

    @staticmethod
    def spring_tide_sea_level(database_path: str | Path) -> float:
        """
        Read SPRINGTIJ: the sea level at easterly wind directions, the unique value of the sea level over all loading
        rows with a direction outside sector west.

        Read from the database because pydra's 2D loading model of an easterly direction does not carry it. If the value
        is not unique, the assumption of a fixed sea level at easterly wind does not hold for this database, and the
        calculation is refused with a 'NotImplementedError', as Hydra-NL stops with an error.

        Parameters
        ----------
        database_path : str | Path
            The path of the HRD database

        Returns
        -------
        float
            The spring tide sea level
        """
        if not Path(database_path).is_file():
            raise FileNotFoundError(
                f"[ERROR] Database {database_path} not found; the spring tide sea level cannot be read."
            )
        with sqlite3.connect(str(database_path)) as con:
            rows = con.execute(
                "select distinct i.Value from HydroDynamicData d "
                "join HRDWindDirections wd on wd.HRDWindDirectionId = d.HRDWindDirectionId "
                "join HydroDynamicInputData i on i.HydroDynamicDataId = d.HydroDynamicDataId "
                "where i.HRDInputColumnId = ? and not ("
                "  (wd.Direction >= 0.0 and wd.Direction <= 11.25) or wd.Direction > 212.75)",
                (_SEA_LEVEL_COLUMN,),
            ).fetchall()
        values = sorted({round(float(row[0]), 6) for row in rows})
        if len(values) != 1:
            raise NotImplementedError(
                f"[ERROR] The sea level at easterly wind directions is not unique in {database_path} ({values}); the "
                f"spring tide assumption of Hydra-NL does not hold here."
            )
        return values[0]

    @staticmethod
    def has_rhine_meuse_delta_statistics(stats) -> bool:
        """
        Determine whether the statistics have one slow stochastic, the discharge, and the closing situations 0 and 11 of
        the Europoort barrier. Both are needed: an ordinary lower river location also has only the discharge as slow
        stochastic, and every water system has a barrier object, with k = [1] where there is no barrier.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location

        Returns
        -------
        bool
            True for the statistics of the Rhine-Meuse delta
        """
        barrier = getattr(stats, "barrier", None)
        if barrier is None:
            return False
        k = sorted(round(float(x)) for x in barrier.k)
        return set(stats.stochastics_slow) == {"q"} and k == [0, 11]

    @staticmethod
    def is_rhine_meuse_delta(hrd, settings) -> bool:
        """
        Determine whether a location belongs to the Rhine-Meuse delta, without building the illustration point. A
        location without a model or statistics does not.

        Parameters
        ----------
        hrd : HRDatabase
            The database of the location
        settings : Settings
            The settings of the location

        Returns
        -------
        bool
            True for a location in the Rhine-Meuse delta
        """
        try:
            stats = hrd.create_location(settings).get_model().get_statistics()
            return HydraRhineMeuseDeltaIllustrationPoint.has_rhine_meuse_delta_statistics(stats)
        except Exception:  # noqa: BLE001 - a location whose model cannot be built is not this water system
            return False
