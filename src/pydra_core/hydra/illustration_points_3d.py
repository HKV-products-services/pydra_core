"""
The Hydra-NL illustration point on a 3D loading field with a storm surge barrier: the machinery that every fresh water
system with a barrier shares. A subclass per water system fills the seams; the method and the seams are described in
docs/hydra_extensions.qmd.
"""

import copy
import numpy as np

from collections.abc import Callable
from scipy.interpolate import RegularGridInterpolator

from . import compiled_scans
from .compiled_contour import SENTINEL
from .illustration_points import HydraIllustrationPoint, _BIG
from .library import Contour, Grid, Scaling
from ..core.memory import Memory, RememberedCalculation

_MAXINT = 32767.0

# The three meaningful sections through the 3D loading field; the first element is the horizontal axis
_AXIS_PAIRS = {("M", "U"), ("Q", "U"), ("Q", "M")}


class HydraIllustrationPoint3D(metaclass=RememberedCalculation):
    """
    The Hydra-NL illustration point on a 3D loading field (u, m, q) with barrier states.

    Not to be used on its own: a subclass fills the seams (see docs/hydra_extensions.qmd).
    """

    # The name of the level axis in the loading table. A subclass must set it.
    _level_axis: str = ""

    # Hydra-NL's repairs of the loading table, before the extension to UMAX and again after it. Off: the table as pydra
    # delivers it, only extended linearly to UMAX. The repair after the extension makes every level artificially
    # reachable just below UMAX in the tail.
    _hydra_repairs: bool = True

    # How the scans treat an isoline of several parts. Off (Hydra-NL): CONTOUREN_SAMENVOEGEN joins the parts into one
    # polyline, and the refinement also puts candidate points on the connecting segments between the parts. On: every
    # part is refined on its own, without connecting segments. An illustration point is the most probable point on the
    # limit state; a point on a connecting segment does not lie on it, and if it wins, beta is too small.
    _isoline_per_part: bool = False

    def check(self, stats) -> None:
        """
        Refuse a water system this subclass is not meant for. With the wrong chain the result is a plausible but wrong
        number, so there is no silent pass.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        raise NotImplementedError("[ERROR] A subclass must fill 'check'.")

    def statistics_grids(self, stats) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the statistics grid of the level stochastic and that of the discharge, on which the blocks lie.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The statistics grids of the level and the discharge
        """
        raise NotImplementedError("[ERROR] A subclass must fill 'statistics_grids'.")

    def prepare(self, stats) -> None:
        """
        Prepare the water system before the loading tables (densities, barrier tables, failure probability of the
        barrier in '_failure_probability').

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        """
        raise NotImplementedError("[ERROR] A subclass must fill 'prepare'.")

    def after_block_grids(self) -> None:
        """
        Prepare what needs the block grids. Nothing by default.
        """

    def joint_umqk(self, ir: int) -> np.ndarray:
        """
        Calculate the joint probability density over (u, m, q) per barrier state. This is the only place with the
        factorisation of the water system.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The joint probability density, with the shape (nu, nm, nq, 2)
        """
        raise NotImplementedError("[ERROR] A subclass must fill 'joint_umqk'.")

    def loading_relevant(self, r: float, k: int) -> bool:
        """
        Determine whether the loading table (r, k) is loaded. Everything by default.

        Parameters
        ----------
        r : float
            The wind direction
        k : int
            The closing situation

        Returns
        -------
        bool
            True if the loading table is loaded
        """
        return True

    def conditional_probabilities_direct(self, ir: int) -> tuple | None:
        """
        Return the Rosenblatt chain directly, instead of derived from a density.

        If the loading table is 2D, the level does not occur in the response and its conditional CDF is a constant,
        which no density can give. That constant puts the coordinate Phi^-1(F) at zero, so that the axis adds nothing
        to beta.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        tuple | None
            None by default: 'conditional_probabilities' derives the chain from 'joint_umqk'. A water system without a
            density over all three axes for some wind directions returns (F(q|k), F(m|q,k), F(u|m,q,k), P(k|r)) on the
            block grids.
        """
        return None

    def level_axis_reference(self, loading) -> np.ndarray:
        """
        Return the level axis to broaden a 2D loading table over: that of a 3D table. All 3D tables share the same
        axis.

        Parameters
        ----------
        loading : Loading
            The loading of the location

        Returns
        -------
        np.ndarray
            The level axis
        """
        for lm in loading.model.values():
            if self._level_axis in list(lm.input_variables):
                return np.asarray(getattr(lm, self._level_axis), float)
        raise NotImplementedError(
            f"[ERROR] No loading table has a {self._level_axis!r} axis; this water system is 2D and does not belong to this "
            f"class."
        )

    def design_level(self, r: float) -> float | None:
        """
        Return a fixed value of the level stochastic in the design point. Belongs to
        'conditional_probabilities_direct': if the level adds nothing to beta, the coordinate from the scan is
        arbitrary and Hydra-NL reports a fixed value.

        Parameters
        ----------
        r : float
            The wind direction

        Returns
        -------
        float | None
            The fixed level, or None
        """
        return None

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
        Prepare the illustration point on the 3D loading field of one location.

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
        self._settings = settings
        model = hrd.create_location(settings).get_model()
        loading = model.get_loading()
        if response == "hs":
            # Hydra-NL repairs a wave height line, so the waves are repaired here as well
            loading.repair_loadingmodels("hs")
        stats = model.get_statistics()
        self.check(stats)

        self._u_max = float(getattr(settings, "u_max", 50.0) or 50.0)

        # Wind direction and wind speed are built the same way in every water system
        self._directions = np.asarray(stats.wind_direction.get_discretisation(), float)
        self._p_r = np.asarray(stats.wind_direction.get_probability(), float)
        self._u_stat = np.asarray(stats.wind_speed.get_discretisation(), float)
        self._epu = np.asarray(stats.wind_speed.get_exceedance_probability(), float)  # (u, r)
        self._mstat, self._qstat = self.statistics_grids(stats)
        self._stats = stats
        self.prepare(stats)

        # Loading per (r, k): axes to (u, m, q), repaired, extended to UMAX and repaired again, in the order of
        # Hydra-NL (the table is repaired after reading, UITBREIDING repairs once more after the extrapolation).
        # Without '_hydra_repairs' it is only extended.
        self._loading: dict[tuple[float, int], dict] = {}
        # The response tables of 'response_table', keyed on (id(loading), delta, f_hs, f_ts) without the level: the
        # table does not depend on it. 'id' is safe, every loading dict lives in '_loading' as long as this cache.
        self._response_cache: dict[tuple[int, float, float, float], np.ndarray] = {}
        for (r, k), lm in loading.model.items():
            if not self.loading_relevant(float(r), int(k)):
                continue
            iv = list(lm.input_variables)
            wanted = ["h"] if response == "h" else (["h", "hs"] if response == "hs" else ["h", "hs", "tspec", "dir"])
            arrays = {}
            if self._level_axis in iv:
                perm = [iv.index("u"), iv.index(self._level_axis), iv.index("q")]
                for name in wanted:
                    arrays[name] = np.transpose(np.asarray(getattr(lm, name), float), perm)
                level_axis = np.asarray(getattr(lm, self._level_axis), float)
            else:
                # A 2D table (u, q): the response does not depend on the level, so broadening it over the level axis
                # is exact, every layer is the same
                perm = [iv.index("u"), iv.index("q")]
                level_axis = self.level_axis_reference(loading)
                for name in wanted:
                    flat = np.transpose(np.asarray(getattr(lm, name), float), perm)
                    arrays[name] = np.repeat(flat[:, None, :], len(level_axis), axis=1)
            if self._hydra_repairs:
                arrays["h"] = HydraIllustrationPoint3D.repair_3d(arrays["h"])
            u_ext, arrays = HydraIllustrationPoint3D.extend_u(np.asarray(lm.u, float), arrays, self._u_max)
            if self._hydra_repairs:
                arrays["h"] = HydraIllustrationPoint3D.repair_3d(arrays["h"])
            self._loading[(float(r), int(k))] = {"u": u_ext, "m": level_axis, "q": np.asarray(lm.q, float), **arrays}

        # Block grids (DEELVECTOR of the statistics grids within the range of the loading). The reference is a table
        # with its own level axis: a broadened 2D table borrowed its axis and says nothing about the range.
        loading0 = next(
            (
                b
                for (r, k), b in self._loading.items()
                if self._level_axis in list(loading.model[(r, k)].input_variables)
            ),
            None,
        )
        if loading0 is None:
            loading0 = next(iter(self._loading.values()))
        self._ublock = Grid.sub_vector(self._u_stat, float(loading0["u"][0]), float(loading0["u"][-1]))
        self._mblock = Grid.sub_vector(self._mstat, float(loading0["m"][0]), float(loading0["m"][-1]))
        self._qblock = Grid.sub_vector(self._qstat, float(loading0["q"][0]), float(loading0["q"][-1]))
        self.after_block_grids()

        # The combinations per barrier state: the model uncertainty of the water level may differ per state. Only the
        # deltas shift, the distances and 'comps' do not depend on sigma; that is checked here so that it shows if it
        # ever changes.
        self._combinations_k = [
            HydraIllustrationPoint.uncertainty_combinations_by_distance(stats, response, model_uncertainty, k)
            for k in (1, 2)
        ]
        for a, b in zip(*self._combinations_k, strict=True):
            if not (
                np.isclose(a[4], b[4], rtol=0.0, atol=1e-12) and np.allclose(a[5], b[5], rtol=0.0, atol=1e-12)
            ):
                raise ValueError(
                    "[ERROR] The distance contribution of the model uncertainty differs per barrier state; the search for the "
                    "design point assumes that only the values shift."
                )
        # Taken at once: pydra_core keeps the model uncertainties in class attributes, so a location built later would
        # change them
        self._combinations_with_probability = [
            Scaling.uncertainty_combinations(stats, response, model_uncertainty, k) for k in (1, 2)
        ]
        self._conditional_cache: dict[int, tuple] = {}
        self._barrier_probability_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ probabilities per direction

    def directions(self) -> list[float]:
        """
        Return the wind directions of the statistics.

        Returns
        -------
        list[float]
            The wind directions
        """
        return [float(r) for r in self._directions]

    def nearest_ir(self, R: float) -> int:
        """
        Return the index of the wind direction of the statistics nearest to a direction.

        Parameters
        ----------
        R : float
            The wind direction

        Returns
        -------
        int
            The index of the nearest wind direction
        """
        # Circular distance: north (0) belongs to 360, not to 22.5
        d = np.abs(((self._directions - R + 180.0) % 360.0) - 180.0)
        return int(np.argmin(d))

    def conditional_probabilities(self, ir: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids (ONDKANSEN_ILLU), per barrier state (index 0 open,
        1 closed), derived from 'joint_umqk' in the order q, m, u.

        Also stores the barrier weight P(k|r) in '_barrier_probability_cache', the probability of every barrier state
        given the direction.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            F(q|k), F(m|q,k) and F(u|m,q,k)
        """
        if ir in self._conditional_cache:
            return self._conditional_cache[ir]
        r = float(self._directions[ir])
        nu, nm, nq = len(self._ublock), len(self._mblock), len(self._qblock)

        direct = self.conditional_probabilities_direct(ir)
        if direct is not None:
            fq, fmq, fumq, barrier_probabilities = direct
            self._barrier_probability_cache[ir] = np.asarray(barrier_probabilities, float)
            out = (fq, fmq, fumq)
            self._conditional_cache[ir] = out
            return out

        joint = self.joint_umqk(ir)

        du = Grid.step_sizes(self._ublock)
        dm = Grid.step_sizes(self._mblock)
        dq = Grid.step_sizes(self._qblock)

        n_states = 2 if HydraIllustrationPoint3D.sector_west(r) else 1
        fq = np.zeros((nq, 2))
        fmq = np.zeros((nm, nq, 2))
        fumq = np.zeros((nu, nm, nq, 2))
        barrier_probabilities = np.zeros(2)
        for ik in range(n_states):
            j = joint[..., ik]
            barrier_probabilities[ik] = float(np.einsum("umq,u,m,q->", j, du, dm, dq))
            fq[:, ik], fmq[:, :, ik], fumq[..., ik] = HydraIllustrationPoint3D.rosenblatt_chain(
                j, self._ublock, self._mblock, self._qblock
            )
        # P(k|r) normalised on the sum over the states, not on p_r[ir]: the two should be equal, but the grid
        # integration of g(u|r) and g(m,q) is not exactly 1, and this way the weights always sum to 1
        total = float(barrier_probabilities.sum())
        self._barrier_probability_cache[ir] = (barrier_probabilities / total) if total > 0.0 else barrier_probabilities
        out = (fq, fmq, fumq)
        self._conditional_cache[ir] = out
        return out

    def barrier_weights(self, R: float) -> dict[int, float]:
        """
        Return P(k | r): the probability of every closing situation (1 open, 2 closed) given the wind direction,
        summing to 1. At easterly wind directions the barrier never closes, and only k = 1 exists, with weight 1.

        Parameters
        ----------
        R : float
            The wind direction

        Returns
        -------
        dict[int, float]
            The probability per closing situation
        """
        ir = self.nearest_ir(R)
        self.conditional_probabilities(ir)
        p = self._barrier_probability_cache[ir]
        if not HydraIllustrationPoint3D.sector_west(float(self._directions[ir])):
            return {1: 1.0}
        return {1: float(p[0]), 2: float(p[1])}

    # ------------------------------------------------------------------ the three scans

    def boundary_parts(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, level: float
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Determine the isoline z = level as a list of parts, every part at least two points, in the path order of
        ISOLIJN. With '_isoline_per_part' off that is one part: Hydra-NL's contour, merged if it has several parts.

        Parameters
        ----------
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid, with the shape (len(x), len(y))
        level : float
            The level

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            The x and y coordinates per part
        """
        if not self._isoline_per_part:
            wx, wy = Contour.contour(x, y, z, level)
            return [(wx, wy)] if len(wx) >= 2 else []
        wx, wy = Contour.isoline(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float), level)
        parts = []
        begin = 0
        for end in [*np.flatnonzero((wx <= SENTINEL) | (wy <= SENTINEL)), len(wx)]:
            if end - begin >= 2:
                parts.append((wx[begin:end], wy[begin:end]))
            begin = end + 1
        return parts

    def refined_boundary(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, level: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the candidate points of a scan: every part of 'boundary_parts' refined on its own (FACTOR = 10).

        Parameters
        ----------
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid, with the shape (len(x), len(y))
        level : float
            The level

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The x and y coordinates of the candidate points
        """
        parts = [Contour.refine(wx, wy) for wx, wy in self.boundary_parts(x, y, z, level)]
        if not parts:
            return np.empty(0), np.empty(0)
        return np.concatenate([d[0] for d in parts]), np.concatenate([d[1] for d in parts])

    def scan_u(self, resp, u, m, q, level, oq, omdq, oudmq) -> tuple | None:
        """
        Scan the contours at fixed wind speed (ILLU_PUNTU): per block wind speed a (m, q) contour, with Z on the row of
        the wind speed node.

        The contour of one part is compiled ('compiled_scans.scan_u_parts'); a contour of several parts is merged in
        Python, which is list work, and only the scan after it is compiled.

        Parameters
        ----------
        resp : np.ndarray
            The response table on the (u, m, q) axes of the loading
        u, m, q : np.ndarray
            The axes of the loading
        level : float
            The level
        oq, omdq, oudmq : np.ndarray
            F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids

        Returns
        -------
        tuple | None
            (distance^2, (U, M, Q), (Q^2, M^2, U^2)) of the closest point, or None
        """
        best = None
        respu = HydraIllustrationPoint3D.inextrp1d_first_axis(resp, u, self._ublock)
        oqc = np.ascontiguousarray(oq, float)
        omdqc = np.ascontiguousarray(omdq, float)
        oudmqc = np.ascontiguousarray(oudmq, float)
        mblock = np.ascontiguousarray(self._mblock, float)
        qblock = np.ascontiguousarray(self._qblock, float)
        if self._isoline_per_part:
            found, r = compiled_scans.scan_u_parts(
                np.asarray(m, float), np.asarray(q, float), respu, float(level), mblock, qblock,
                oqc, omdqc, oudmqc, int(self._ik), np.ascontiguousarray(self._ublock, float),
            )
            if found >= 0:
                return HydraIllustrationPoint3D.kernel_result(r) if found else None
        for iu in range(len(self._ublock)):
            for wx, wy in self.boundary_parts(m, q, respu[iu, :, :], level):
                r = compiled_scans.scan_u_kernel(
                    np.ascontiguousarray(wx, float), np.ascontiguousarray(wy, float),
                    mblock, qblock, oqc, omdqc, oudmqc, int(self._ik), int(iu), float(self._ublock[iu]),
                )
                if not np.isfinite(r[0]):
                    continue
                if best is None or r[0] < best[0]:
                    best = HydraIllustrationPoint3D.kernel_result(r)
        return best

    def scan_q(self, resp, u, m, q, level, oq, omdq, oudmq) -> tuple | None:
        """
        Scan the contours at fixed discharge (ILLU_PUNTQ): per block discharge a (u, m) contour, with X on the
        discharge node.

        Parameters
        ----------
        resp : np.ndarray
            The response table on the (u, m, q) axes of the loading
        u, m, q : np.ndarray
            The axes of the loading
        level : float
            The level
        oq, omdq, oudmq : np.ndarray
            F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids

        Returns
        -------
        tuple | None
            (distance^2, (U, M, Q), (Q^2, M^2, U^2)) of the closest point, or None
        """
        best = None
        respq = HydraIllustrationPoint3D.inextrp1d_last_axis(resp, q, self._qblock)
        xq = HydraIllustrationPoint3D.phi_inv_hydra(oq[:, self._ik])
        omdqc = np.ascontiguousarray(omdq, float)
        oudmqc = np.ascontiguousarray(oudmq, float)
        ublock = np.ascontiguousarray(self._ublock, float)
        mblock = np.ascontiguousarray(self._mblock, float)
        if self._isoline_per_part:
            found, r = compiled_scans.scan_q_parts(
                np.asarray(u, float), np.asarray(m, float), respq, float(level), ublock, mblock,
                omdqc, oudmqc, int(self._ik), np.ascontiguousarray(self._qblock, float),
                np.ascontiguousarray(xq, float),
            )
            if found >= 0:
                return HydraIllustrationPoint3D.kernel_result(r) if found else None
        for iq in range(len(self._qblock)):
            for wx, wy in self.boundary_parts(u, m, respq[:, :, iq], level):
                r = compiled_scans.scan_q_kernel(
                    np.ascontiguousarray(wx, float), np.ascontiguousarray(wy, float),
                    ublock, mblock, omdqc, oudmqc, int(self._ik), int(iq), float(self._qblock[iq]), float(xq[iq]),
                )
                if not np.isfinite(r[0]):
                    continue
                if best is None or r[0] < best[0]:
                    best = HydraIllustrationPoint3D.kernel_result(r)
        return best

    def scan_m(self, resp, u, m, q, level, oq, omdq, oudmq) -> tuple | None:
        """
        Scan the contours at fixed level (ILLU_PUNTM): per block level a (u, q) contour, with Y on the row of the level
        node.

        Parameters
        ----------
        resp : np.ndarray
            The response table on the (u, m, q) axes of the loading
        u, m, q : np.ndarray
            The axes of the loading
        level : float
            The level
        oq, omdq, oudmq : np.ndarray
            F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids

        Returns
        -------
        tuple | None
            (distance^2, (U, M, Q), (Q^2, M^2, U^2)) of the closest point, or None
        """
        best = None
        respm = HydraIllustrationPoint3D.inextrp1d_mid_axis(resp, m, self._mblock)
        oqc = np.ascontiguousarray(oq, float)
        omdqc = np.ascontiguousarray(omdq, float)
        oudmqc = np.ascontiguousarray(oudmq, float)
        ublock = np.ascontiguousarray(self._ublock, float)
        qblock = np.ascontiguousarray(self._qblock, float)
        if self._isoline_per_part:
            found, r = compiled_scans.scan_m_parts(
                np.asarray(u, float), np.asarray(q, float), respm, float(level), ublock, qblock,
                oqc, omdqc, oudmqc, int(self._ik), np.ascontiguousarray(self._mblock, float),
            )
            if found >= 0:
                return HydraIllustrationPoint3D.kernel_result(r) if found else None
        for im in range(len(self._mblock)):
            for wx, wy in self.boundary_parts(u, q, respm[:, im, :], level):
                r = compiled_scans.scan_m_kernel(
                    np.ascontiguousarray(wx, float), np.ascontiguousarray(wy, float),
                    ublock, qblock, oqc, omdqc, oudmqc, int(self._ik), int(im), float(self._mblock[im]),
                )
                if not np.isfinite(r[0]):
                    continue
                if best is None or r[0] < best[0]:
                    best = HydraIllustrationPoint3D.kernel_result(r)
        return best

    # ------------------------------------------------------------------ design point

    def response_table(self, loading: dict, delta: float, f_hs: float, f_ts: float) -> np.ndarray:
        """
        Return the response table for the isolines at one uncertainty combination.

        The crest level table depends on (loading, delta, f_hs, f_ts) and not on the level, so it is kept: every
        return period asks for the same tables again. The kept table is shared, not copied, and therefore read-only, so
        that writing into it fails loudly instead of changing every other level.

        Parameters
        ----------
        loading : dict
            The loading of one (r, k), from '_loading'
        delta : float
            The shift of the water level
        f_hs : float
            The factor for the significant wave height
        f_ts : float
            The factor for the spectral wave period

        Returns
        -------
        np.ndarray
            The response table on the (u, m, q) axes of the loading
        """
        if self.response == "h":
            # The level shifts instead: level - delta
            return loading["h"]
        if self.response == "hs":
            # The level shifts instead: level / f_hs
            return loading["hs"]
        key = (id(loading), float(delta), float(f_hs), float(f_ts))
        crest = self._response_cache.get(key)
        if crest is None:
            crest = np.array(
                self._crest_level_table(
                    self._profile, self._qcrit, loading["h"] + delta, loading["hs"], loading["tspec"], loading["dir"],
                    f_hs, f_ts,
                )
            )
            crest.setflags(write=False)
            self._response_cache[key] = crest
        return crest

    def design_point(self, R: float, level: float, closing_situation: int) -> dict | None:
        """
        Find the design point for a wind direction, level and closing situation (ILLUSTRATIEP) with the loop over the
        uncertainty combinations.

        Parameters
        ----------
        R : float
            The wind direction
        level : float
            The level
        closing_situation : int
            1 (open) or 2 (closed)

        Returns
        -------
        dict | None
            The keys 'r', 'closing_situation', 'design' ({"U", "M", "Q"}), 'beta', 'dh', 'alpha2'; for the wave height
            and the crest level also 'h' (the water level in the design point), for the wave height 'f_hs', and for the
            crest level 'waves' and 'uncertainty'. None if there is none, and always for k = 2 at an easterly
            direction, where the barrier never closes.
        """
        ir = self.nearest_ir(R)
        r = float(self._directions[ir])
        ik = {1: 0, 2: 1}[closing_situation]
        if ik == 1 and not HydraIllustrationPoint3D.sector_west(r):
            return None
        loading = self._loading[(r, ik + 1)]
        oq, omdq, oudmq = self.conditional_probabilities(ir)
        self._ik = ik

        u, m, q = loading["u"], loading["m"], loading["q"]
        # The best candidate holds the total distance^2, the point (U, M, Q), the uncertainty combination, the squared
        # coordinates (Q^2, M^2, U^2) and the model uncertainty terms
        best = None
        for delta, f_hs, f_ts, f_tp, distance2, comps in self._combinations_k[ik]:
            if best is not None and distance2 >= best[0]:
                # Sorted on distance: the remaining combinations cannot give a smaller distance
                break
            resp = self.response_table(loading, delta, f_hs, f_ts)
            lvl = {"h": level - delta, "hs": level / f_hs, "hbn": level}[self.response]
            candidate = None
            if len(self._mblock) > 1 and len(self._ublock) > 1:
                candidate = HydraIllustrationPoint3D.smallest(
                    candidate, self.scan_q(resp, u, m, q, lvl, oq, omdq, oudmq)
                )
            if len(self._qblock) > 1 and len(self._ublock) > 1:
                candidate = HydraIllustrationPoint3D.smallest(
                    candidate, self.scan_m(resp, u, m, q, lvl, oq, omdq, oudmq)
                )
            if len(self._qblock) > 1 and len(self._mblock) > 1:
                candidate = HydraIllustrationPoint3D.smallest(
                    candidate, self.scan_u(resp, u, m, q, lvl, oq, omdq, oudmq)
                )
            if candidate is None:
                continue
            total = candidate[0] + distance2
            if best is None or total < best[0]:
                best = (total, candidate[1], (delta, f_hs, f_ts, f_tp), candidate[2], comps)
        if best is None:
            return None
        d2, (U, M, Q), (delta, f_hs, f_ts, f_tp), (q2, m2, u2), comps = best
        beta = float(np.sqrt(min(d2, _MAXINT)))
        fixed_level = self.design_level(r)
        if fixed_level is not None:
            M = float(fixed_level)
        out = {
            "r": r,
            "closing_situation": closing_situation,
            "design": {"U": U, "M": M, "Q": Q},
            "beta": beta,
            "dh": delta,
            "alpha2": HydraIllustrationPoint.alpha2_decomposition(
                [("Q", "Q", q2), ("M", "M", m2), ("U", "U", u2)], comps, d2, self.response, self.model_uncertainty
            ),
        }
        pt = np.array([U, M, Q])
        if self.response in ("hs", "hbn"):
            out["h"] = float(
                HydraIllustrationPoint3D.trilinear(loading["u"], loading["m"], loading["q"], loading["h"], pt)
            ) + (delta if self.response == "hbn" else 0.0)
        if self.response == "hs":
            out["f_hs"] = f_hs
        if self.response == "hbn":
            sin = np.sin(np.pi * loading["dir"] / 180.0)
            cos = np.cos(np.pi * loading["dir"] / 180.0)
            wave_direction = (
                np.degrees(
                    np.arctan2(
                        HydraIllustrationPoint3D.trilinear(loading["u"], loading["m"], loading["q"], sin, pt),
                        HydraIllustrationPoint3D.trilinear(loading["u"], loading["m"], loading["q"], cos, pt),
                    )
                )
                % 360.0
            )
            out["waves"] = {
                "h": out["h"],
                "Hm0": float(
                    HydraIllustrationPoint3D.trilinear(loading["u"], loading["m"], loading["q"], loading["hs"], pt)
                )
                * f_hs,
                "Tspec": float(
                    HydraIllustrationPoint3D.trilinear(loading["u"], loading["m"], loading["q"], loading["tspec"], pt)
                )
                * f_ts,
                "dir": float(wave_direction),
            }
            out["uncertainty"] = {"h": delta, "f_hs": f_hs, "f_tspec": f_ts, "f_tp": f_tp}
        return out

    def design_grid(
        self,
        R: float,
        level: float,
        closing_situation: int,
        axes: tuple[str, str] = ("M", "U"),
        n_x: int = 100,
        n_y: int = 100,
    ) -> dict | None:
        """
        Calculate the limit state field as a 2D section through the design point of this closing situation.

        The loading is 3D (Q, M, U), so a flat section fixes one stochastic, at its value in the design point. The
        section then goes through the design point, and the contour 'level' of the field coincides with the limit
        state. 'axes' chooses the pair: ("M", "U"), ("Q", "U") or ("Q", "M"), the first one horizontal. The ranges are
        the block grids over which the port integrates.

        The returned 'beta' is the 3D beta of the design point, not the beta that can be read from the section: the
        third coordinate is missing in the plane, so the distance to the origin seen there is smaller.

        Parameters
        ----------
        R : float
            The wind direction
        level : float
            The level
        closing_situation : int
            1 (open) or 2 (closed)
        axes : tuple[str, str], optional
            The pair of axes. Default is ("M", "U").
        n_x : int, optional
            The number of values on the horizontal axis. Default is 100.
        n_y : int, optional
            The number of values on the vertical axis. Default is 100.

        Returns
        -------
        dict | None
            The keys 'x_values', 'y_values', 'response' ([i_x][j_y]), 'slow_label' (the horizontal axis), 'design'
            ({"x", "y"}), 'beta', 'alpha2', 'closing_situation', 'p_closing_situation', 'axes', 'fixed'
            ({"key", "value"}), 'design_3d' and 'beta_is_3d'. None if there is no design point.
        """
        if self.response not in ("h", "hbn"):
            raise NotImplementedError(f"[ERROR] The section exists for the response 'h' or 'hbn', not {self.response!r}.")
        pair = tuple(axes)
        if pair not in _AXIS_PAIRS:
            raise ValueError(f"[ERROR] Unknown pair of axes {pair!r}; choose from {sorted(_AXIS_PAIRS)}.")
        dp = self.design_point(R, float(level), closing_situation)
        if dp is None:
            return None

        ir = self.nearest_ir(R)
        r = float(self._directions[ir])
        loading = self._loading[(r, closing_situation)]
        delta = float(dp["dh"])
        o = dp.get("uncertainty") or {}
        # The field is built so that the limit state lies at 'level': for the water level the model uncertainty
        # shifts the field (h + delta) instead of the level (level - delta)
        resp = self.response_table(loading, delta, float(o.get("f_hs", 1.0)), float(o.get("f_tspec", 1.0)))
        if self.response == "h":
            resp = resp + delta

        gi = RegularGridInterpolator(
            (loading["u"], loading["m"], loading["q"]), resp, method="linear", bounds_error=False, fill_value=None
        )

        ranges = {"U": self._ublock, "M": self._mblock, "Q": self._qblock}
        fixed_key = ({"U", "M", "Q"} - set(pair)).pop()
        x_values = np.linspace(float(ranges[pair[0]][0]), float(ranges[pair[0]][-1]), n_x)
        y_values = np.linspace(float(ranges[pair[1]][0]), float(ranges[pair[1]][-1]), n_y)
        XX, YY = np.meshgrid(x_values, y_values, indexing="ij")
        columns = {
            pair[0]: XX.ravel(),
            pair[1]: YY.ravel(),
            fixed_key: np.full(XX.size, float(dp["design"][fixed_key])),
        }
        H = gi(np.column_stack([columns["U"], columns["M"], columns["Q"]])).reshape(XX.shape)

        return {
            "x_values": x_values.tolist(),
            "y_values": y_values.tolist(),
            "response": H.tolist(),
            "slow_label": pair[0],
            "design": {"x": float(dp["design"][pair[0]]), "y": float(dp["design"][pair[1]])},
            "beta": dp["beta"],
            "alpha2": dp.get("alpha2"),
            "closing_situation": closing_situation,
            "p_closing_situation": self.barrier_weights(R).get(closing_situation),
            "axes": list(pair),
            "fixed": {"key": fixed_key, "value": float(dp["design"][fixed_key])},
            "design_3d": {k: float(v) for k, v in dp["design"].items()},
            "beta_is_3d": True,
        }

    def all_directions(self, level: float) -> list[dict]:
        """
        Find the design point per wind direction and closing situation at a level, where there is one.

        Parameters
        ----------
        level : float
            The level

        Returns
        -------
        list[dict]
            The design points of 'design_point'
        """
        out = []
        for r in self.directions():
            for closing_situation in (1, 2):
                dp = self.design_point(r, float(level), closing_situation)
                if dp is not None:
                    out.append(dp)
        return out

    # ------------------------------------------------------------------ block probability

    def uncertainty_combinations(self, closing_situation: int = 1) -> list[tuple[float, float, float, float]]:
        """
        Return the uncertainty combinations with their probability KANSONZ, summing to 1. In a water system with the
        same model uncertainty in both states the two lists are equal.

        Parameters
        ----------
        closing_situation : int, optional
            1 (open) or 2 (closed). Default is 1.

        Returns
        -------
        list[tuple[float, float, float, float]]
            The combinations (delta_h, f_Hm0, f_Tspec, probability)
        """
        return self._combinations_with_probability[closing_situation - 1]

    def response_on_blocks(self, ir: int, ik: int, combination: tuple[float, float, float]) -> np.ndarray:
        """
        Interpolate the response to the block grid (u, m, q). The loading is on the axes of the database (u extended to
        UMAX), the density on the block grids; for an integral both must be on the same grid.

        Parameters
        ----------
        ir : int
            The index of the wind direction
        ik : int
            The barrier state index (0 open, 1 closed)
        combination : tuple[float, float, float]
            The uncertainty combination (delta_h, f_hs, f_ts)

        Returns
        -------
        np.ndarray
            The response on the block grid
        """
        r = float(self._directions[ir])
        loading = self._loading[(r, ik + 1)]
        delta, f_hs, f_ts = combination
        resp = self.response_table(loading, delta, f_hs, f_ts)
        resp = HydraIllustrationPoint3D.inextrp1d_first_axis(resp, loading["u"], self._ublock)
        resp = HydraIllustrationPoint3D.inextrp1d_mid_axis(resp, loading["m"], self._mblock)
        return HydraIllustrationPoint3D.inextrp1d_last_axis(resp, loading["q"], self._qblock)

    def p_failure_block(self, R: float, level: float) -> dict:
        """
        Calculate the conditional failure probability per loading block at a wind direction, split over the closing
        situations:

            P|r = sum_k P(k|r) * P(failure | k, r),   P(failure | k, r) = int g_k(u,m,q) 1{response > level}
                                                                         / int g_k(u,m,q)

        with g_k from 'joint_umqk' and the integrals as trapezium sums over the block grid on which the design point
        is searched. Normalised on the sum of the grid integrals, not on p_r[ir], as in 'conditional_probabilities'.
        The model uncertainty is the outer discrete sum over KANSONZ.

        This is a block probability, not a yearly frequency: with two slow stochastics there is no single number of
        blocks per year.

        Parameters
        ----------
        R : float
            The wind direction
        level : float
            The level

        Returns
        -------
        dict
            The keys 'r', 'p', 'closing_situations' ({k: {'p_closing_situation', 'p_failure', 'contribution'}}) and
            'combinations' (per combination 'dh', 'f_hs', 'f_tspec', 'probability', 'p_failure')
        """
        ir = self.nearest_ir(R)
        r = float(self._directions[ir])
        level = float(level)
        joint = self.joint_umqk(ir)
        du = Grid.step_sizes(self._ublock)
        dm = Grid.step_sizes(self._mblock)
        dq = Grid.step_sizes(self._qblock)
        n_states = 2 if HydraIllustrationPoint3D.sector_west(r) else 1

        denominator = np.zeros(2)
        for ik in range(n_states):
            denominator[ik] = float(np.einsum("umq,u,m,q->", joint[..., ik], du, dm, dq))
        total = float(denominator.sum())
        if total <= 0.0:
            raise ValueError("[ERROR] No probability mass on this block grid for this wind direction.")

        numerator = np.zeros(2)
        rows: list[dict] = []
        for ic, (dh0, f_hs0, f_ts0, p_combination) in enumerate(self.uncertainty_combinations()):
            if p_combination <= 0.0:
                continue
            t_combination = 0.0
            for ik in range(n_states):
                # The delta (and the wave factors) come from the table of this barrier state; the weight is the same,
                # KANSONZ does not depend on the state
                dh, f_hs, f_ts, _ = self.uncertainty_combinations(ik + 1)[ic]
                lvl = {"h": level - dh, "hs": level / f_hs, "hbn": level}[self.response]
                failure = self.response_on_blocks(ir, ik, (dh, f_hs, f_ts)) > lvl
                t = float(np.einsum("umq,u,m,q->", joint[..., ik] * failure, du, dm, dq))
                numerator[ik] += p_combination * t
                t_combination += t
            rows.append(
                {
                    "dh": float(dh0), "f_hs": float(f_hs0), "f_tspec": float(f_ts0),
                    "probability": float(p_combination), "p_failure": t_combination / total,
                }
            )

        per_state = {}
        for ik in range(n_states):
            per_state[ik + 1] = {
                "p_closing_situation": denominator[ik] / total,
                "p_failure": (numerator[ik] / denominator[ik]) if denominator[ik] > 0.0 else 0.0,
                "contribution": numerator[ik] / total,
            }
        return {"r": r, "p": float(numerator.sum() / total), "closing_situations": per_state, "combinations": rows}

    @staticmethod
    def sector_west(r: float) -> bool:
        """
        Determine whether the barrier can close at a wind direction (SECTORWEST): only at westerly wind directions.

        The same bounds for both barriers ported so far. That is a coincidence between two independent water systems,
        not a shared property: a barrier with other bounds needs a seam here.

        Parameters
        ----------
        r : float
            The wind direction

        Returns
        -------
        bool
            True for a westerly wind direction
        """
        return (0.0 <= r <= 11.25) or (212.75 < r <= 360.0)

    @staticmethod
    def rosenblatt_chain(
        j: np.ndarray, ublock: np.ndarray, mblock: np.ndarray, qblock: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the Rosenblatt chain q, m, u from one density on the (u, m, q) block grid, as cumulative trapezium
        sums: the three tables of Hydra-NL's ONDKANSEN_ILLU routines, without an assumption on what is conditioned in.

        Parameters
        ----------
        j : np.ndarray
            The density on the block grid, with the shape (nu, nm, nq). Its normalisation does not matter, all three
            tables are quotients in which a constant factor drops out.
        ublock, mblock, qblock : np.ndarray
            The block grids

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            F(q) (nq,), F(m|q) (nm, nq) and F(u|m,q) (nu, nm, nq)
        """
        du, dm = Grid.step_sizes(ublock), Grid.step_sizes(mblock)
        Fq = np.zeros(len(qblock))
        Fmq = np.zeros((len(mblock), len(qblock)))
        Fumq = np.zeros(j.shape)

        density_mq = np.einsum("umq,u->mq", j, du)
        with np.errstate(invalid="ignore", divide="ignore"):
            gu = np.where(density_mq[None, :, :] > 0.0, j / density_mq[None, :, :], 0.0)
        Fumq[1:] = np.cumsum((gu[:-1] + gu[1:]) / 2.0 * np.diff(ublock)[:, None, None], axis=0)

        density_q = np.einsum("mq,m->q", density_mq, dm)
        with np.errstate(invalid="ignore", divide="ignore"):
            gm = np.where(density_q[None, :] > 0.0, density_mq / density_q[None, :], 0.0)
        Fmq[1:] = np.cumsum((gm[:-1] + gm[1:]) / 2.0 * np.diff(mblock)[:, None], axis=0)

        total = float(density_q @ Grid.step_sizes(qblock))
        gq = density_q / total if total > 0.0 else np.zeros_like(density_q)
        Fq[1:] = np.cumsum((gq[:-1] + gq[1:]) / 2.0 * np.diff(qblock))
        return Fq, Fmq, Fumq

    @staticmethod
    def repair_3d(h: np.ndarray) -> np.ndarray:
        """
        Repair the loading table (REPAREER_ZOET): a running maximum along every axis of (u, m, q), so that every cell is
        at least its predecessor.

        Parameters
        ----------
        h : np.ndarray
            The loading table on (u, m, q)

        Returns
        -------
        np.ndarray
            The repaired loading table
        """
        out = np.array(h, float, copy=True)
        for ax in range(3):
            out = np.maximum.accumulate(out, axis=ax)
        return out

    @staticmethod
    def extend_u(u: np.ndarray, arrays: dict[str, np.ndarray], u_max: float) -> tuple[np.ndarray, dict]:
        """
        Extend the loading along the wind speed (UITBREIDING): extra nodes up to UMAX (node placement as in the
        Fortran, NINT in float32), the water level and the wave parameters extended linearly with the last slope of the
        table. The wave parameters are clipped at zero.

        A wave direction extended beyond 360 degrees is brought back by 360: at a wind direction of 360 the table can
        end just below 360 with a rising slope, and the overtopping calculation refuses an angle above 360. Every
        direction that already lies within 0 to 360 is left as it is.

        Parameters
        ----------
        u : np.ndarray
            The wind speeds of the loading table
        arrays : dict[str, np.ndarray]
            The loading fields on (u, m, q)
        u_max : float
            The largest wind speed (UMAX)

        Returns
        -------
        tuple[np.ndarray, dict]
            The extended wind speeds and the extended loading fields
        """
        u = np.asarray(u, float)
        if u[-1] >= u_max:
            return u, arrays
        deltau = u[-1] - u[-2]
        n_new = max(1, Grid.nint32((u_max - u[-1]) / deltau))
        unew = u[-1] + (u_max - u[-1]) * np.arange(1, n_new + 1) / n_new
        out: dict[str, np.ndarray] = {}
        for name, arr in arrays.items():
            arr = np.asarray(arr, float)
            slope = (arr[-1] - arr[-2]) / deltau
            ext = [arr[-1] + slope * (uv - u[-1]) for uv in unew]
            new = np.concatenate([arr] + [e[None, ...] for e in ext], axis=0)
            if name != "h":
                new = np.maximum(new, 0.0)
            if name == "dir":
                new = np.where(new > 360.0, new - 360.0, new)
            out[name] = new
        return np.concatenate([u, unew]), out

    @staticmethod
    def phi_inv_hydra(probability: np.ndarray) -> np.ndarray:
        """
        Calculate Phi^-1 with the clips of the Fortran: probability <= 5e-16 gives -MAXINT, 1 - probability <= 5e-16
        gives +MAXINT.

        Parameters
        ----------
        probability : np.ndarray
            The non-exceedance probability

        Returns
        -------
        np.ndarray
            The standard normal coordinate
        """
        probability = np.asarray(probability, float)
        out = np.where(probability <= 5.0e-16, -_MAXINT, np.where(1.0 - probability <= 5.0e-16, _MAXINT, 0.0))
        mid = (probability > 5.0e-16) & (1.0 - probability > 5.0e-16)
        if mid.any():
            out = np.array(out, float)
            out[mid] = np.clip(HydraIllustrationPoint.phi_inv(probability[mid]), -_BIG, _BIG)
        return out

    @staticmethod
    def inextrp1d_last_axis(arr: np.ndarray, x: np.ndarray, xn: np.ndarray) -> np.ndarray:
        """
        Interpolate the last axis linearly on new nodes (INTERP1D, clamped).

        Parameters
        ----------
        arr : np.ndarray
            The values
        x : np.ndarray
            The nodes of the last axis
        xn : np.ndarray
            The new nodes

        Returns
        -------
        np.ndarray
            The values on the new nodes
        """
        j, fr = Grid.lower_index_and_fraction_clamped(x, xn)
        return arr[..., j] * (1.0 - fr) + arr[..., j + 1] * fr

    @staticmethod
    def inextrp1d_mid_axis(arr: np.ndarray, x: np.ndarray, xn: np.ndarray) -> np.ndarray:
        """
        Interpolate the middle axis of a 3D array linearly on new nodes (INTERP1D, clamped).

        Parameters
        ----------
        arr : np.ndarray
            The values, with three axes
        x : np.ndarray
            The nodes of the middle axis
        xn : np.ndarray
            The new nodes

        Returns
        -------
        np.ndarray
            The values on the new nodes
        """
        j, fr = Grid.lower_index_and_fraction_clamped(x, xn)
        return arr[:, j, :] * (1.0 - fr)[None, :, None] + arr[:, j + 1, :] * fr[None, :, None]

    @staticmethod
    def inextrp1d_first_axis(arr: np.ndarray, x: np.ndarray, xn: np.ndarray) -> np.ndarray:
        """
        Interpolate the first axis of a 3D array linearly on new nodes (INTERP1D, clamped).

        Parameters
        ----------
        arr : np.ndarray
            The values, with three axes
        x : np.ndarray
            The nodes of the first axis
        xn : np.ndarray
            The new nodes

        Returns
        -------
        np.ndarray
            The values on the new nodes
        """
        j, fr = Grid.lower_index_and_fraction_clamped(x, xn)
        return arr[j, :, :] * (1.0 - fr)[:, None, None] + arr[j + 1, :, :] * fr[:, None, None]

    @staticmethod
    def trilinear(u: np.ndarray, m: np.ndarray, q: np.ndarray, arr: np.ndarray, pt: np.ndarray) -> float:
        """
        Interpolate trilinearly on (u, m, q) (INTERP3D).

        Parameters
        ----------
        u, m, q : np.ndarray
            The axes
        arr : np.ndarray
            The values on (u, m, q)
        pt : np.ndarray
            The point (u, m, q)

        Returns
        -------
        float
            The interpolated value
        """
        gi = RegularGridInterpolator((u, m, q), arr, method="linear", bounds_error=False, fill_value=None)
        return float(gi(pt[None, :])[0])

    @staticmethod
    def kernel_result(r: np.ndarray) -> tuple:
        """
        Group a scan kernel result.

        Parameters
        ----------
        r : np.ndarray
            The kernel result (distance^2, U, M, Q, Q^2, M^2, U^2)

        Returns
        -------
        tuple
            (distance^2, (U, M, Q), (Q^2, M^2, U^2))
        """
        return (r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6]))

    @staticmethod
    def smallest(a, b):
        """
        Return the scan result with the smallest distance.

        Parameters
        ----------
        a, b : tuple | None
            Scan results as 'kernel_result', or None

        Returns
        -------
        tuple | None
            The result with the smallest distance^2; None if both are None
        """
        if a is None:
            return b
        if b is None:
            return a
        return a if a[0] <= b[0] else b
