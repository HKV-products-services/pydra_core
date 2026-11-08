"""
The Hydra-NL illustration point for the Hollandsche IJssel: the Rhine-Meuse delta with a second barrier, the
Hollandsche IJssel barrier, per state of that barrier. The method is described in docs/hydra_extensions.qmd.
"""

import copy
import sqlite3

import numpy as np

from pathlib import Path

from .illustration_points_3d import HydraIllustrationPoint3D
from .library import Grid, HydraInterpolate
from .rhine_meuse_delta import HydraRhineMeuseDeltaIllustrationPoint
from ..location.model.statistics.water_systems.statistics_hollandsche_ijssel import StatisticsHollandscheIJssel
from ..location.model.statistics.water_systems.statistics_lower_river import StatisticsLowerRiver

# The states of the Hollandsche IJssel barrier, as the closing situations 1 to 3 of the database
HOLLANDSCHE_IJSSEL_STATES = {
    1: "open",
    2: "closed without water burden",
    3: "closed with water burden",
}


class HydraHollandscheIJsselIllustrationPoint(HydraRhineMeuseDeltaIllustrationPoint):
    """
    The Hydra-NL illustration point for one location at the Hollandsche IJssel, for one state of the Hollandsche
    IJssel barrier ('hollandsche_ijssel_state', 1 to 3) and both states of the Europoort barrier (the closing
    situation 1 or 2 of 'design_point').

    The other parameters are those of 'HydraIllustrationPoint3D.__init__'.
    """

    # The water level is made non-decreasing again after the extension to UMAX, as Rekenhart_HIJ does ("Repareer
    # uitgebreide waterstanden", REPAREREN_HIJ in database_uitlezen_HIJ.f90). Within the range of the database the table
    # is already non-decreasing; the extension continues every (m, q) line from the last two wind speeds with its own
    # slope, so at UMAX a higher discharge or sea level can end up below a lower one (up to 14 cm in the demo database).
    _hydra_repairs: bool = True

    def __init__(
        self,
        hrd,
        settings,
        statistical_uncertainty: bool = True,
        model_uncertainty: bool = True,
        response: str = "h",
        profile=None,
        qcrit: float | None = None,
        crest_level_table=None,
        hollandsche_ijssel_state: int = 1,
    ):
        if hollandsche_ijssel_state not in HOLLANDSCHE_IJSSEL_STATES:
            raise ValueError(
                f"[ERROR] Unknown state {hollandsche_ijssel_state!r} of the Hollandsche IJssel barrier; choose from "
                f"{HOLLANDSCHE_IJSSEL_STATES}."
            )
        self.hollandsche_ijssel_state = hollandsche_ijssel_state

        # Rekenhart_HIJ searches the illustration point on the wind speeds of the statistics file (IPPUNTEN_HIJ), not on
        # the grid of USTAP that the frequency line uses; on the coarser grid the design point shifts up to 0.5 m/s
        settings = copy.deepcopy(settings)
        settings.u_step = None
        super().__init__(
            hrd, settings, statistical_uncertainty, model_uncertainty, response, profile, qcrit, crest_level_table
        )

    def check(self, stats) -> None:
        """
        Refuse a location that is not at the Hollandsche IJssel.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        if not isinstance(stats, StatisticsHollandscheIJssel):
            raise NotImplementedError(
                f"[ERROR] {self.location} is not a location at the Hollandsche IJssel: the illustration point expects "
                f"'StatisticsHollandscheIJssel'."
            )

    def loading_tables(self) -> dict[int, int]:
        """
        Return the closing situations of the database for this state of the Hollandsche IJssel barrier, with the
        Europoort barrier open (1 to 3) and closed (4 to 6).

        Returns
        -------
        dict[int, int]
            The closing situation of the database per state of the Europoort barrier
        """
        return {1: self.hollandsche_ijssel_state, 2: self.hollandsche_ijssel_state + 3}

    def prepare(self, stats) -> None:
        """
        Prepare the Rhine-Meuse delta, and read the closing function of the Hollandsche IJssel barrier.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        super().prepare(stats)
        self._alfa = float(self._settings.hollandsche_ijssel_barrier_alfa)
        self._closing_function = HydraHollandscheIJsselIllustrationPoint.read_closing_function(
            self._settings.database_path
        )
        self._weight_cache: dict[int, np.ndarray] = {}
        self._weighted_joint_cache: dict[int, np.ndarray] = {}

    def europoort_probability(self, r: float, closing_situation: int) -> np.ndarray:
        """
        Return pydra's probability mass of a state of the Europoort barrier, without the weight of the Hollandsche
        IJssel barrier: that weight depends on the closing function and is put on the block grids in 'joint_umqk'.

        Parameters
        ----------
        r : float
            The wind direction
        closing_situation : int
            1 (Europoort barrier open) or 2 (closed)

        Returns
        -------
        np.ndarray
            The probability mass on the statistics grids (u, q, m)
        """
        return StatisticsLowerRiver.calculate_probability(self._stats, r, closing_situation=closing_situation)

    def joint_umqk(self, ir: int) -> np.ndarray:
        """
        Calculate the joint probability density over (u, m, q) per state of the Europoort barrier, for this state of the
        Hollandsche IJssel barrier, for a westerly direction (ONDKANSEN_HIJ).

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The joint probability density, with the shape (nu, nm, nq, 2)
        """
        if ir not in self._weighted_joint_cache:
            self._weighted_joint_cache[ir] = super().joint_umqk(ir) * self.hollandsche_ijssel_weight(ir)
        return self._weighted_joint_cache[ir]

    def conditional_probabilities_direct(self, ir: int) -> tuple | None:
        """
        Return the Rosenblatt chain of an easterly direction (ONDKANSEN_HIJ): the Europoort barrier is open and the sea
        level is no stochastic, but the Hollandsche IJssel barrier can close. F(q) and F(u|q) come from the density
        of the wind speed given the direction and that of the discharge, times the weight of this state; F(m|q) is
        0.5, as in the Rhine-Meuse delta.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        tuple | None
            (F(q|k), F(m|q,k), F(u|m,q,k), P(k|r)) on the block grids; None for a westerly direction
        """
        if HydraIllustrationPoint3D.sector_west(float(self._directions[ir])):
            return None

        nu, nm, nq = len(self._ublock), len(self._mblock), len(self._qblock)
        fq = np.zeros((nq, 2))
        fmq = np.zeros((nm, nq, 2))
        fumq = np.zeros((nu, nm, nq, 2))

        # The densities on the block grids, as INTERP1D followed by KANSDICHTHEID
        exceedance_u = np.maximum(HydraInterpolate.interp1d_clamped(self._u_stat, self._epu[:, ir], self._ublock), 0.0)
        density_u = Grid.probability_density_raw(self._ublock, exceedance_u)
        exceedance_q = np.interp(
            self._qblock, self._qstat, np.asarray(self._stats.discharge.get_exceedance_probability(), float)
        )
        density_q = Grid.probability_density_raw(self._qblock, exceedance_q)

        # The sea level does not occur: the density is the same at every sea level, F(m|q) is overwritten
        density_uq = density_u[:, None] * self.hollandsche_ijssel_weight(ir)[:, 0, :, 0] * density_q[None, :]
        if not np.any(density_uq > 0.0):
            return fq, fmq, fumq, np.zeros(2)
        density = np.repeat(density_uq[:, None, :], nm, axis=1)
        fq[:, 0], _, fumq[..., 0] = HydraIllustrationPoint3D.rosenblatt_chain(
            density, self._ublock, self._mblock, self._qblock
        )
        fmq[:, :, 0] = 0.5
        return fq, fmq, fumq, np.array([1.0, 0.0])

    def design_point(self, R: float, level: float, closing_situation: int) -> dict | None:
        """
        Find the design point as 'HydraIllustrationPoint3D.design_point', with None for a state without probability:
        the Hollandsche IJssel barrier closed where the closing function never reaches the closing level.

        Parameters
        ----------
        R : float
            The wind direction
        level : float
            The level
        closing_situation : int
            1 (Europoort barrier open) or 2 (closed)

        Returns
        -------
        dict | None
            The design point, see 'HydraIllustrationPoint3D.design_point'
        """
        ir = self.nearest_ir(R)
        self.conditional_probabilities(ir)
        if not self._barrier_probability_cache[ir][closing_situation - 1] > 0.0:
            return None
        return super().design_point(R, level, closing_situation)

    def hollandsche_ijssel_weight(self, ir: int) -> np.ndarray:
        """
        Calculate the weight of this state of the Hollandsche IJssel barrier on the block grids (ONDKANSEN_HIJ). Where
        the water level at the barrier ('Zmax HOIJK', interpolated and extrapolated as GRIDINEXTRP3D) stays below the
        closing level, the barrier is open; elsewhere it is closed unless it fails, with probability ALFA_HOIJK. The
        water burden plays no part: both closed states have the same weight.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The weight, with the shape (nu, nm, nq, 2), the last axis the state of the Europoort barrier
        """
        if ir in self._weight_cache:
            return self._weight_cache[ir]
        r = float(self._directions[ir])
        u_axis, m_axis, q_axis = self._closing_function["u"], self._closing_function["m"], self._closing_function["q"]
        levels = self._closing_function["levels"].get(round(r, 6))
        if levels is None:
            raise NotImplementedError(
                f"[ERROR] Wind direction {r} is not in the closing function of the Hollandsche IJssel barrier."
            )

        weight = np.zeros((len(self._ublock), len(self._mblock), len(self._qblock), 2))
        for ik in range(2):
            level_at_barrier = HydraIllustrationPoint3D.inextrp1d_first_axis(levels[..., ik], u_axis, self._ublock)
            level_at_barrier = HydraIllustrationPoint3D.inextrp1d_mid_axis(level_at_barrier, m_axis, self._mblock)
            level_at_barrier = HydraIllustrationPoint3D.inextrp1d_last_axis(level_at_barrier, q_axis, self._qblock)
            closes = level_at_barrier >= self._closing_function["closing_level"]
            if self.hollandsche_ijssel_state == 1:
                weight[..., ik] = np.where(closes, self._alfa, 1.0)
            else:
                weight[..., ik] = np.where(closes, 1.0 - self._alfa, 0.0)
        self._weight_cache[ir] = weight
        return weight

    @staticmethod
    def read_closing_function(database_path: str | Path) -> dict:
        """
        Read the closing function of the Hollandsche IJssel barrier from the table 'Sluitfunctie HOIJK', as Hydra-NL's
        SLUITFUNCTIE_HIJ_HOIJK: the water level at the barrier per (u, m, q) and state of the Europoort barrier, for
        every wind direction, and the closing level.

        At an easterly direction the table has one sea level and only the open Europoort barrier; that level is copied
        to all sea levels and both states (INHOUD_HOIJK_OOST). The water level is then made non-decreasing in (u, m, q)
        per state of the Europoort barrier and direction (REPAREER_SLUITF), always, also when the repairs of the
        loading are switched off. Without it the barrier closes far less often with the Europoort barrier closed, where
        the table falls with the sea level.

        Parameters
        ----------
        database_path : str | Path
            The path of the HRD database

        Returns
        -------
        dict
            The keys 'u', 'm', 'q' (the axes of the westerly directions), 'levels' ({r: array (nu, nm, nq, 2)}) and
            'closing_level'
        """
        if not Path(database_path).is_file():
            raise FileNotFoundError(
                f"[ERROR] Database {database_path} not found; the closing function of the Hollandsche IJssel barrier "
                f"cannot be read."
            )
        with sqlite3.connect(str(database_path)) as con:
            rows = np.array(
                con.execute(
                    'select Windrichting, Windsnelheid, Zeewaterstand, Afvoer, Europoortkering, "Zmax HOIJK", '
                    'Sluitwaterstand from "Sluitfunctie HOIJK"'
                ).fetchall(),
                float,
            )
        closing_levels = np.unique(rows[:, 6])
        if len(closing_levels) != 1:
            raise NotImplementedError(
                f"[ERROR] The closing function of the Hollandsche IJssel barrier has more than one closing level "
                f"({closing_levels}); Hydra-NL stops here as well."
            )

        west = np.array([HydraIllustrationPoint3D.sector_west(r) for r in rows[:, 0]])
        u_axis = np.unique(rows[:, 1])
        m_axis = np.unique(rows[west, 2])
        q_axis = np.unique(rows[:, 3])
        k_axis = np.unique(rows[west, 4])
        if len(k_axis) != 2:
            raise NotImplementedError(
                f"[ERROR] The closing function of the Hollandsche IJssel barrier has the Europoort states {k_axis}, "
                f"expected two (open and closed)."
            )

        levels = {}
        for r in np.unique(rows[:, 0]):
            table = np.full((len(u_axis), len(m_axis), len(q_axis), 2), np.nan)
            direction = rows[rows[:, 0] == r]
            iu = np.searchsorted(u_axis, direction[:, 1])
            iq = np.searchsorted(q_axis, direction[:, 3])
            if HydraIllustrationPoint3D.sector_west(r):
                im = np.searchsorted(m_axis, direction[:, 2])
                ik = np.searchsorted(k_axis, direction[:, 4])
                table[iu, im, iq, ik] = direction[:, 5]
            else:
                table[iu, :, iq, :] = direction[:, 5][:, None, None]
            if np.isnan(table).any():
                raise NotImplementedError(
                    f"[ERROR] The closing function of the Hollandsche IJssel barrier is incomplete at wind direction {r}."
                )
            for ik in range(2):
                table[..., ik] = HydraIllustrationPoint3D.repair_3d(table[..., ik])
            levels[round(float(r), 6)] = table
        return {"u": u_axis, "m": m_axis, "q": q_axis, "levels": levels, "closing_level": float(closing_levels[0])}
