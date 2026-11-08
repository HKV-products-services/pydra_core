"""
The Hydra-NL breakdown and percentiles for the Hollandsche IJssel, per state of the Europoort barrier (EPK) and the
Hollandsche IJssel barrier (HIJK). The Fortran chain is described in docs/hydra_extensions.qmd.
"""

import copy
import numpy as np

from collections.abc import Callable
from typing import NamedTuple

from .compiled_lower_river_breakdown import masked_sum
from .hollandsche_ijssel import HOLLANDSCHE_IJSSEL_STATES, HydraHollandscheIJsselIllustrationPoint
from .illustration_points_3d import HydraIllustrationPoint3D
from .library import Grid, Percentiles, WaveProcessing
from ..core.memory import Memory, RememberedCalculation

# The two closing types of the HIJK (IT in Rekenhart_HIJ): the closed state of each, the open state counts in both
CLOSING_TYPES = (2, 3)


class _Piece(NamedTuple):
    """
    What 'masked_sum' needs per (HIJK state, wind direction, EPK state) and does not depend on the level.
    """

    hollandsche_ijssel_state: int  # 1 = open, 2 = closed without water burden, 3 = closed with water burden
    direction: int  # index in '_directions'
    barrier_state: int  # EPK: 0 = open, 1 = closed
    loading: dict  # loading table of the illustration point
    indices: tuple  # interpolation indices and fractions of u, m and q to the block grid
    joint: np.ndarray  # bracket on the block grid (u, m, q), with the weight of the HIJK state


class HydraHollandscheIJsselPercentiles(metaclass=RememberedCalculation):
    """
    The Hydra-NL breakdown of the Hollandsche IJssel for one location: the split over the six states of the EPK and
    the HIJK, per wind direction, and the percentiles of discharge, sea level and wind speed per state.
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
        Prepare the breakdown for one location at the Hollandsche IJssel.

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
            "h" (water level), "hs" (significant wave height), "tspec" (spectral wave period Tm-1,0) or "hbn" (required
            crest level; needs 'profile' and 'qcrit')
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
        # 'WaveShape' changes 'base_duration', so the wave shapes get their own copy of the settings as they came in
        fresh = copy.deepcopy(settings)

        # One illustration point per HIJK state; they share the statistics and the block grids
        self._ips = {
            state: HydraHollandscheIJsselIllustrationPoint(
                hrd, settings, statistical_uncertainty, model_uncertainty, response=response, profile=profile,
                qcrit=qcrit, crest_level_table=crest_level_table, hollandsche_ijssel_state=state,
            )
            for state in HOLLANDSCHE_IJSSEL_STATES
        }
        ip = self._ips[1]
        self.location = ip.location
        stats = ip._stats
        self._water_burden_probability = float(fresh.hollandsche_ijssel_water_burden_probability)

        self._ublock, self._mblock, self._qblock = ip._ublock, ip._mblock, ip._qblock
        self._du, self._dm, self._dq = (Grid.step_sizes(x) for x in (self._ublock, self._mblock, self._qblock))
        self._directions = [float(r) for r in ip._directions]

        # Discharge waves on Hydra-NL's time axis and OVERSCHRIJDINGSDUUR_Q, as in 'HydraRhineMeuseDeltaPercentiles'
        self._block_duration_wind = float(fresh.block_duration)
        self._qpeak = np.asarray(stats.discharge.qpeak, float)
        self._epqpeak = np.asarray(stats.discharge.epqpeak, float)
        self._times, self._waves_q, ws = WaveProcessing.wave_shapes_on_hydra_axis(fresh, "DISCHARGE", self._qpeak)
        self._durations_q = WaveProcessing.exceedance_durations(
            ws, self._qblock, self._waves_q, self._block_duration_wind, clamp=False
        )
        self._block_factors = WaveProcessing.block_factors(self._times)
        self._density_qpeak = Grid.probability_density_raw(self._qpeak, self._epqpeak)
        self._dqpeak = Grid.step_sizes(self._qpeak)

        # The uncertainty combinations per (HIJK state, EPK state). Rekenhart_HIJ has one KANSONZ per combination for
        # all six states, so the weights have to be equal.
        self._combinations = {
            (state, ik): self._ips[state]._combinations_with_probability[ik]
            for state in HOLLANDSCHE_IJSSEL_STATES
            for ik in (0, 1)
        }
        weights = [[c[3] for c in combinations] for combinations in self._combinations.values()]
        if any(len(w) != len(weights[0]) or not np.allclose(w, weights[0], rtol=0.0, atol=1e-12) for w in weights):
            raise ValueError(
                "[ERROR] The weights of the uncertainty combinations differ per state of the barriers; the breakdown "
                "walks the lists in lockstep, because Rekenhart_HIJ has one KANSONZ per combination."
            )
        self._weights = weights[0]

        self._breakdown_cache: dict[float, dict] = {}
        self._pieces_cache: list[_Piece] | None = None
        self._ahead_done = False

    # Building blocks

    def tables_ahead(self) -> None:
        """
        Calculate the crest level tables of every piece for every combination with a probability, once, in worker
        processes ('Memory.calculate_ahead'); 'response_table' then finds them in the store.
        """
        if self.response != "hbn" or self._ahead_done:
            return
        self._ahead_done = True
        active = [ic for ic, weight in enumerate(self._weights) if weight > 0.0]
        work = []
        for piece in self.pieces():
            combinations = self._combinations[(piece.hollandsche_ijssel_state, piece.barrier_state)]
            loading = piece.loading
            tables = (loading["h"], loading["hs"], loading["tspec"], loading["dir"])
            work.append((tables, [combinations[ic][:3] for ic in active]))
        ip = self._ips[1]
        Memory.calculate_ahead(ip._crest_level_table, ip._profile, ip._qcrit, work)

    def pieces(self) -> list[_Piece]:
        """
        Return per (HIJK state, wind direction, EPK state) what 'masked_sum' needs and does not depend on the level.
        The closed EPK only counts in sector west.

        Returns
        -------
        list[_Piece]
            The pieces
        """
        if self._pieces_cache is None:
            self._pieces_cache = []
            for state, ip in self._ips.items():
                for jr, r in enumerate(self._directions):
                    joint = ip.bracket_umqk(jr)
                    for ik in range(2 if HydraIllustrationPoint3D.sector_west(r) else 1):
                        loading = ip._loading[(r, ik + 1)]
                        indices = (
                            *Grid.lower_index_and_fraction_clamped(loading["u"], self._ublock),
                            *Grid.lower_index_and_fraction_clamped(loading["m"], self._mblock),
                            *Grid.lower_index_and_fraction_clamped(loading["q"], self._qblock),
                        )
                        self._pieces_cache.append(
                            _Piece(state, jr, ik, loading, indices, np.ascontiguousarray(joint[..., ik]))
                        )
        return self._pieces_cache

    def frequency_density_q(self, p_q: np.ndarray) -> np.ndarray:
        """
        Calculate the frequency density f_Q(q) (FREQDICHTH_Q, formula 4.6): VERWERKGOLF on the block probability, then
        the frequency density, as 'HydraRhineMeuseDeltaPercentiles.frequency_density_q'.

        Parameters
        ----------
        p_q : np.ndarray
            The block probability P(loading >= level | Q = q) on the discharge block grid

        Returns
        -------
        np.ndarray
            The frequency density on the discharge block grid
        """
        ovk_t = np.empty((len(self._times), len(self._qpeak)))
        for s in range(len(self._qpeak)):
            ovk_t[:, s] = np.interp(self._waves_q[:, s], self._qblock, p_q)
        scaling = WaveProcessing.process_wave_scaling(
            ovk_t, self._block_factors, self._times[1] - self._times[0], self._block_duration_wind
        )
        weight = self._dqpeak * self._density_qpeak * scaling
        return WaveProcessing.frequency_density_1d(self._qblock, weight, self._durations_q)

    # The breakdown

    def breakdown(self, level: float) -> dict:
        """
        Calculate all breakdown arrays at a level, summed over the uncertainty combinations and the two closing types.

        Rekenhart_HIJ calculates per closing type of the HIJK (without or with water burden, weight
        1 - KANS_WATERBEZWAAR and KANS_WATERBEZWAAR) with the open state and the closed state of that type. Each closing type has its own
        block probability p(q) and with it its own f_Q(q); the open state therefore gets a contribution from both.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            The keys 'freqRS' (nr, 2, 3), 'freqQS' (nq, 2, 3), 'freqMS' (nm, 2, 3) and 'freqUS' (nu, 2, 3), with the
            EPK axis (0 = open, 1 = closed) and the HIJK axis (0 = open, 1 = closed without and 2 = closed with water
            burden), plus the grids 'q', 'm', 'u' and 'r'
        """
        level = float(level)
        if level in self._breakdown_cache:
            return self._breakdown_cache[level]

        self.tables_ahead()
        nm, nq, nu, nr = len(self._mblock), len(self._qblock), len(self._ublock), len(self._directions)
        pieces = self.pieces()
        du, dm = np.ascontiguousarray(self._du), np.ascontiguousarray(self._dm)
        ones = np.ones(nq)
        unused_m, unused_u = np.zeros(nm), np.zeros(nu)
        # Per piece the marginals over q, m and u, summed over the combinations and the closing types
        sum_q = np.zeros((len(pieces), nq))
        sum_m = np.zeros((len(pieces), nm))
        sum_u = np.zeros((len(pieces), nu))

        for ic, p_unc in enumerate(self._weights):
            if p_unc <= 0.0:
                continue
            responses, levels = [], []
            for piece in pieces:
                dh, f_hs, f_ts, _ = self._combinations[(piece.hollandsche_ijssel_state, piece.barrier_state)][ic]
                ip = self._ips[piece.hollandsche_ijssel_state]
                responses.append(ip.response_table(piece.loading, dh, f_hs, f_ts))
                levels.append({"h": level - dh, "hs": level / f_hs, "tspec": level / f_ts, "hbn": level}[self.response])

            for closed_state in CLOSING_TYPES:
                type_probability = (
                    self._water_burden_probability if closed_state == 3 else 1.0 - self._water_burden_probability
                )
                in_type = [i for i, piece in enumerate(pieces) if piece.hollandsche_ijssel_state in (1, closed_state)]

                # BLOKKENREKENEN_HIJ, then FREQDICHTH_Q: the block probability of this closing type
                p_q = np.zeros(nq)
                for i in in_type:
                    piece = pieces[i]
                    masked_sum(
                        responses[i], *piece.indices, piece.joint, du, dm, ones, levels[i], p_q, unused_m, unused_u
                    )
                f_q = self.frequency_density_q(np.minimum(p_q, 1.0))

                # BIJDRAGEN_UMQKDR, marginalised per piece
                weight_q = f_q * self._dq * p_unc * type_probability
                for i in in_type:
                    piece = pieces[i]
                    masked_sum(
                        responses[i], *piece.indices, piece.joint, du, dm, weight_q, levels[i], sum_q[i], sum_m[i],
                        sum_u[i],
                    )

        # To the states (BIJDRAGEN_VERDELEN)
        freqRS = np.zeros((nr, 2, 3))
        freqQS = np.zeros((nq, 2, 3))
        freqMS = np.zeros((nm, 2, 3))
        freqUS = np.zeros((nu, 2, 3))
        for i, piece in enumerate(pieces):
            jr, ik, js = piece.direction, piece.barrier_state, piece.hollandsche_ijssel_state - 1
            freqRS[jr, ik, js] += sum_q[i].sum()
            freqQS[:, ik, js] += sum_q[i]
            freqMS[:, ik, js] += sum_m[i]
            freqUS[:, ik, js] += sum_u[i]

        out = {
            "freqRS": freqRS,
            "freqQS": freqQS,
            "freqMS": freqMS,
            "freqUS": freqUS,
            "q": self._qblock,
            "m": self._mblock,
            "u": self._ublock,
            "r": np.asarray(self._directions, float),
        }
        self._breakdown_cache[level] = out
        return out

    # Output tables, as Rekenhart_HIJ prints them

    def state_contributions(self, level: float) -> dict:
        """
        Calculate the tables "Uitsplitsing over de toestanden van de twee stormvloedkeringen" and "Uitsplitsing over de
        windrichtingen": the percentage of the exceedance frequency per state and per (wind direction, state).

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            'states' (2, 3) and 'directions' (nr, 2, 3) in percent, with the axes of 'breakdown', and 'r'
        """
        b = self.breakdown(level)
        total = float(b["freqRS"].sum())
        if total <= 0.0:
            return {"states": np.zeros((2, 3)), "directions": np.zeros_like(b["freqRS"]), "r": b["r"]}
        return {
            "states": 100.0 * b["freqRS"].sum(axis=0) / total,
            "directions": 100.0 * b["freqRS"] / total,
            "r": b["r"],
        }

    def percentiles(self, level: float) -> dict:
        """
        Calculate the tables "Percentielen van de Rijnafvoer", "van de zeewaterstand" and "van de windsnelheid": per
        state and for all states together.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        dict
            'pct' (the percentile labels), and per stochastic ('discharge', 'sea_level', 'wind_speed') a dict with the
            keys (EPK state, HIJK state), 1-based as the database numbers them, and "all"; the value is the list of
            percentiles, or None without frequency, where Hydra-NL prints '-----'
        """
        b = self.breakdown(level)
        out = {"pct": Percentiles.percentile_labels()}
        for name, grid, freq in (
            ("discharge", b["q"], b["freqQS"]),
            ("sea_level", b["m"], b["freqMS"]),
            ("wind_speed", b["u"], b["freqUS"]),
        ):
            table = {}
            for ik in (0, 1):
                for js in (0, 1, 2):
                    values = Percentiles.percentiles(grid, freq[:, ik, js])
                    table[(ik + 1, js + 1)] = None if values is None else [float(v) for v in values]
            values = Percentiles.percentiles(grid, freq.sum(axis=(1, 2)))
            table["all"] = None if values is None else [float(v) for v in values]
            out[name] = table
        return out
