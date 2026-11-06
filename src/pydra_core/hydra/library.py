"""
The ported Hydra-NL routines that do not depend on a water system: grids and densities, interpolation, contours, wave
processing, percentiles and the scaling of the breakdown. Why some routines exist twice is described in
docs/hydra_extensions.qmd.
"""

import copy
import math
import numpy as np

from .compiled_contour import NUMBA_AVAILABLE, SENTINEL
from ..common.enum import WaveShapeType
from ..location.model.statistics.stochastics.wave_shape import WaveShape

if NUMBA_AVAILABLE:
    from . import compiled_contour, compiled_interpolation


# PERCENTIELVULLEN (NP=7): the percentiles that Hydra-NL prints in the UITSPL and UR tables
PERCENTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

# Subdivision factor for refining parts of an isoline (Fortran FACTOR)
FACTOR = 10

# The breakdown families that SCHAALUITSPLITS each scales to their own sum. The names are the Hydra-NL names of the
# axes (Q discharge, M level, U wind speed, R wind direction, K barrier state) and thus not bound to an area.
FAMILIES = ("freqQK", "freqMK", "freqRK", "freqUK", "freqURK", "freqUR")


class Grid:
    """
    Grids and probability densities as Hydra-NL builds them.
    """

    @staticmethod
    def step_sizes(x: np.ndarray) -> np.ndarray:
        """
        Calculate STAPGROOTTE ('Stapgrootte.f90') for all indices: (x[i+1] - x[i-1]) / 2 inside, half a step at the
        edges.

        Parameters
        ----------
        x : np.ndarray
            The grid

        Returns
        -------
        np.ndarray
            The step size per node
        """
        x = np.asarray(x, float)
        d = np.empty_like(x)
        d[1:-1] = (x[2:] - x[:-2]) / 2.0
        d[0] = (x[1] - x[0]) / 2.0
        d[-1] = (x[-1] - x[-2]) / 2.0
        return d

    @staticmethod
    def lower_index_and_fraction(x: np.ndarray, xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the lower index of points in a grid and the fraction in between, as INEXTRP1D (V_INDICES_ONDER and
        CALC_FRACS). The index is clamped within the grid, the fraction is not, so beyond the grid it extrapolates.

        Parameters
        ----------
        x : np.ndarray
            The grid
        xn : np.ndarray
            The points

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The lower index and the fraction per point
        """
        j = np.clip(np.searchsorted(x, xn) - 1, 0, len(x) - 2)
        return j, (xn - x[j]) / (x[j + 1] - x[j])

    @staticmethod
    def lower_index_and_fraction_clamped(x: np.ndarray, xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the lower index and fraction as 'lower_index_and_fraction', but with the fraction clamped between 0
        and 1, as INTERP1D (no extrapolation).

        Parameters
        ----------
        x : np.ndarray
            The grid
        xn : np.ndarray
            The points

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The lower index and the clamped fraction per point
        """
        j, fr = Grid.lower_index_and_fraction(x, xn)
        return j, np.clip(fr, 0.0, 1.0)

    @staticmethod
    def probability_density_raw(x: np.ndarray, ovkans: np.ndarray) -> np.ndarray:
        """
        Calculate Hydra's bare KANSDICHTHEID: -STAPGROOTTE(ovkans) / STAPGROOTTE(x), without guard. This is the form
        used by the breakdown and the frequency density chain, and by STAT_GEZAMENLIJK when differentiating twice.

        Parameters
        ----------
        x : np.ndarray
            The grid
        ovkans : np.ndarray
            The exceedance probability on the grid

        Returns
        -------
        np.ndarray
            The probability density on the grid
        """
        return -Grid.step_sizes(ovkans) / Grid.step_sizes(x)

    @staticmethod
    def probability_density(x: np.ndarray, ovkans: np.ndarray) -> np.ndarray:
        """
        Calculate KANSDICHTHEID with the guard of 'windinfo_Zoet': density 0 where the exceedance probability is 0.
        According to Hydra-NL's own comment, without it 'strange illustration points' arise. It is a choice about
        content, not a rounding, which is why it is a separate function next to 'probability_density_raw'.

        Parameters
        ----------
        x : np.ndarray
            The grid
        ovkans : np.ndarray
            The exceedance probability on the grid

        Returns
        -------
        np.ndarray
            The probability density on the grid
        """
        dens = Grid.probability_density_raw(x, ovkans)
        dens[np.asarray(ovkans, float) <= 0.0] = 0.0
        return dens

    @staticmethod
    def fill_vector(vmin: float, vmax: float, step: float) -> np.ndarray:
        """
        Fill a grid as VECTORVULLEN, with NINT rounding of the number of steps; the last node is exactly 'vmax'.

        Parameters
        ----------
        vmin : float
            The first node
        vmax : float
            The last node
        step : float
            The step size

        Returns
        -------
        np.ndarray
            The grid
        """
        n = int(math.floor((vmax - vmin) / step + 0.5)) + 1
        return np.append(vmin + step * np.arange(n - 1, dtype=float), vmax)

    @staticmethod
    def nint32(x: float) -> int:
        """
        Round as Fortran NINT on REAL(4). The value is converted to float32 first, because Hydra-NL places its nodes in
        singles: (2.2 - 1.5) / 0.2 = 3.4999998f rounds to 3, where doubles give 3.5 and round to 4.

        Parameters
        ----------
        x : float
            The value

        Returns
        -------
        int
            The rounded value
        """
        return int(np.rint(np.float32(x)))

    @staticmethod
    def sub_vector(x: np.ndarray, ymin: float, ymax: float) -> np.ndarray:
        """
        Take the part of a statistics vector within [ymin, ymax], with the clamped edges added (DEELVECTOR).

        Parameters
        ----------
        x : np.ndarray
            The statistics vector
        ymin : float
            The lower bound
        ymax : float
            The upper bound

        Returns
        -------
        np.ndarray
            The part of the vector, with the clamped edges
        """
        x = np.asarray(x, float)
        if len(x) == 0:
            return np.array([ymax], dtype=float)
        first = None
        for i in range(len(x)):
            if ymin < x[i]:
                # Fortran MAX(IX,2), 0-based
                first = max(i, 1)
                break
        last = None
        for i in range(len(x) - 1, -1, -1):
            if ymax > x[i]:
                last = min(i, len(x) - 2)
                break
        y = [max(ymin, float(x[0]))]
        y.extend(float(v) for v in x[first : last + 1])
        y.append(min(ymax, float(x[-1])))
        return np.asarray(y, float)


class HydraInterpolate:
    """
    Interpolation as in Hydra-NL (INTERP1D, INEXTRP1D).
    """

    @staticmethod
    def interp1d_clamped(xp: np.ndarray, fp: np.ndarray, x) -> np.ndarray:
        """
        Interpolate linearly as Fortran INTERP1D, clamped to the edge value beyond the range.

        Parameters
        ----------
        xp : np.ndarray
            The grid
        fp : np.ndarray
            The values on the grid
        x : array_like
            The points to interpolate at

        Returns
        -------
        np.ndarray
            The interpolated values
        """
        return np.interp(np.asarray(x, float), np.asarray(xp, float), np.asarray(fp, float))

    @staticmethod
    def inextrp1d_interp(x, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        """
        Interpolate and extrapolate linearly (INEXTRP1D) with 'np.interp' plus an explicit edge slope beyond the range.
        Note the order of the arguments: '(x, xp, fp)' here and '(xp, fp, x)' for 'inextrp1d_fractions'.

        Parameters
        ----------
        x : array_like
            The points to interpolate at
        xp : np.ndarray
            The grid
        fp : np.ndarray
            The values on the grid

        Returns
        -------
        np.ndarray
            The interpolated values
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.interp(x, xp, fp)
        lo, hi = x < xp[0], x > xp[-1]
        if lo.any():
            y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        if hi.any():
            y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return y

    @staticmethod
    def inextrp1d_fractions_python(xp: np.ndarray, fp: np.ndarray, x) -> np.ndarray:
        """
        Interpolate and extrapolate linearly (INEXTRP1D) in Hydra's own form: a lower index with a fraction that is not
        clamped. This is the pattern of INITIALISEER_INTERPOLATIE and INEXTRP1D_INIT (I_ONDER, FRACS): the lower index
        is clamped to the grid, the fraction is not, so the edge sections extrapolate linearly.

        This is the reference for 'compiled_interpolation.inextrp1d_fractions'; change both together.

        Parameters
        ----------
        xp : np.ndarray
            The grid
        fp : np.ndarray
            The values on the grid
        x : array_like
            The points to interpolate at

        Returns
        -------
        np.ndarray
            The interpolated values
        """
        x = np.atleast_1d(np.asarray(x, float))
        xp = np.asarray(xp, float)
        fp = np.asarray(fp, float)
        j = np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2)
        fr = (x - xp[j]) / (xp[j + 1] - xp[j])
        return fp[j] * (1.0 - fr) + fp[j + 1] * fr

    @staticmethod
    def inextrp1d_fractions(xp: np.ndarray, fp: np.ndarray, x) -> np.ndarray:
        """
        Interpolate and extrapolate linearly (INEXTRP1D) as 'inextrp1d_fractions_python', compiled when numba is
        available.

        Parameters
        ----------
        xp : np.ndarray
            The grid
        fp : np.ndarray
            The values on the grid
        x : array_like
            The points to interpolate at

        Returns
        -------
        np.ndarray
            The interpolated values
        """
        if NUMBA_AVAILABLE:
            y = compiled_interpolation.inextrp1d_fractions(xp, fp, x)
            if y is not None:
                return y
        return HydraInterpolate.inextrp1d_fractions_python(xp, fp, x)


# Port of 'libsrc/Bibliotheek/Hydra_contour.f90' and 'blokkans.f90' (VLAKKEN_CONTOUR). ISOLIJN follows one
# continuous, ordered contour instead of a sorted point cloud. That prevents a local non-monotonicity in the load
# (the ripple of the overtopping library at high wind speeds) from flipping the edge section.
class Contour:
    """
    Contours (ISOLIJN, marching squares) and their processing.
    """

    @staticmethod
    def interpol(z1: float, z2: float, c1: float, c2: float, cval: float) -> float:
        """
        Interpolate the coordinate where the contour value lies on the edge between two grid points (Fortran
        INTERPOL).

        Parameters
        ----------
        z1, z2 : float
            The values at the two grid points
        c1, c2 : float
            The coordinates of the two grid points
        cval : float
            The value of the contour

        Returns
        -------
        float
            The coordinate of the contour on the edge
        """
        return c1 + (cval - z1) / (z2 - z1) * (c2 - c1)

    @staticmethod
    def next_cell(cell, mx, my, z, cval, ilog):
        """
        Determine the next cell the contour passes through (VOLGENDEVAK, 0-based indices).

        Parameters
        ----------
        cell : tuple
            The x and y indices of the four corners of the current cell
        mx, my : int
            The number of grid points on the x and y axes
        z : np.ndarray
            The values on the grid
        cval : float
            The value of the contour
        ilog : np.ndarray
            The horizontal edges the contour still has to start from, updated in place

        Returns
        -------
        tuple
            The x and y indices of the four corners of the next cell
        """
        ix, iy = cell
        dx = ix[1] - ix[0]
        dy = iy[1] - iy[0]
        if dx == 1:
            ilog[ix[0], iy[0]] = False
        right = z[ix[2], iy[2]] < cval <= z[ix[1], iy[1]]
        left = z[ix[0], iy[0]] < cval <= z[ix[3], iy[3]]
        if right and left:
            hm = sum(z[ix[i], iy[i]] for i in range(4)) / 4.0
            if hm <= cval:
                left = False
            else:
                right = False
        if right:
            nix = (ix[2], ix[1], ix[1] + dx, ix[2] + dx)
            niy = (iy[2], iy[1], iy[1] + dy, iy[2] + dy)
        elif left:
            nix = (ix[0], ix[3], ix[3] - dx, ix[0] - dx)
            niy = (iy[0], iy[3], iy[3] - dy, iy[0] - dy)
        else:
            # Upwards
            nix = (ix[3], ix[2], ix[2] - dy, ix[3] - dy)
            niy = (iy[3], iy[2], iy[2] + dx, iy[3] + dx)
        return (nix, niy)

    @staticmethod
    def follow_line(cell, mx, my, x, y, z, cval, ilog, wx, wy):
        """
        Follow the contour from a start cell through the grid (VOLGLIJN); the points go to wx and wy, closed by the
        sentinel.

        Parameters
        ----------
        cell : tuple
            The x and y indices of the four corners of the start cell
        mx, my : int
            The number of grid points on the x and y axes
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid
        cval : float
            The value of the contour
        ilog : np.ndarray
            The horizontal edges the contour still has to start from, updated in place
        wx, wy : list[float]
            The x and y coordinates of the contour, added to in place
        """
        ix, iy = cell
        zxb = Contour.interpol(z[ix[0], iy[0]], z[ix[1], iy[1]], x[ix[0]], x[ix[1]], cval)
        zyb = Contour.interpol(z[ix[0], iy[0]], z[ix[1], iy[1]], y[iy[0]], y[iy[1]], cval)
        wx.append(zxb)
        wy.append(zyb)
        new_cell = cell
        zxnew, zynew = zxb, zyb
        cell_number = 1
        proceed = True
        while proceed:
            old_cell = new_cell
            zxold, zyold = zxnew, zynew
            new_cell = Contour.next_cell(old_cell, mx, my, z, cval, ilog)
            nix, niy = new_cell
            zxnew = Contour.interpol(z[nix[0], niy[0]], z[nix[1], niy[1]], x[nix[0]], x[nix[1]], cval)
            zynew = Contour.interpol(z[nix[0], niy[0]], z[nix[1], niy[1]], y[niy[0]], y[niy[1]], cval)
            if not (zxnew == zxold and zynew == zyold):
                wx.append(zxnew)
                wy.append(zynew)
            if cell_number > 1:
                proceed = not (zxb == zxnew and zyb == zynew)
            proceed = proceed and (0 <= nix[2] < mx) and (0 <= niy[2] < my)
            cell_number = 2
        wx.append(SENTINEL)
        wy.append(SENTINEL)

    @staticmethod
    def isoline_python(x: np.ndarray, y: np.ndarray, z: np.ndarray, cval: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the ordered contour z(x, y) = cval (ISOLIJN, marching squares). The parts of the contour are separated
        by the sentinel; consecutive duplicate points are removed.

        This is the reference for 'compiled_contour.isoline'; change both together.

        Parameters
        ----------
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid, with the shape (len(x), len(y))
        cval : float
            The value of the contour

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The x and y coordinates of the contour
        """
        mx, my = len(x), len(y)
        ilog = np.zeros((mx - 1, my), dtype=bool)
        for iyi in range(my):
            for ixi in range(mx - 1):
                ilog[ixi, iyi] = z[ixi, iyi] < cval <= z[ixi + 1, iyi]
        wx: list[float] = []
        wy: list[float] = []

        # Edges: bottom, right, top, left
        iy = 0
        for ix in range(mx - 1):
            if z[ix, iy] < cval <= z[ix + 1, iy]:
                ilog[ix, iy] = False
                Contour.follow_line(
                    ((ix, ix + 1, ix + 1, ix), (iy, iy, iy + 1, iy + 1)), mx, my, x, y, z, cval, ilog, wx, wy
                )
        ix = mx - 1
        for iy in range(my - 1):
            if z[ix, iy] < cval <= z[ix, iy + 1]:
                Contour.follow_line(
                    ((ix, ix, ix - 1, ix - 1), (iy, iy + 1, iy + 1, iy)), mx, my, x, y, z, cval, ilog, wx, wy
                )
        iy = my - 1
        for ix in range(mx - 1, 0, -1):
            if z[ix, iy] < cval <= z[ix - 1, iy]:
                Contour.follow_line(
                    ((ix, ix - 1, ix - 1, ix), (iy, iy, iy - 1, iy - 1)), mx, my, x, y, z, cval, ilog, wx, wy
                )
        ix = 0
        for iy in range(my - 1, 0, -1):
            if z[ix, iy] < cval <= z[ix, iy - 1]:
                Contour.follow_line(
                    ((ix, ix, ix + 1, ix + 1), (iy, iy - 1, iy - 1, iy)), mx, my, x, y, z, cval, ilog, wx, wy
                )

        # Closed contours
        for iy in range(1, my - 1):
            for ix in range(mx - 1):
                if ilog[ix, iy]:
                    ilog[ix, iy] = False
                    Contour.follow_line(
                        ((ix, ix + 1, ix + 1, ix), (iy, iy, iy + 1, iy + 1)), mx, my, x, y, z, cval, ilog, wx, wy
                    )
        if not wx:
            return np.empty(0), np.empty(0)

        # Remove the last sentinel and consecutive duplicate points
        wx = wx[:-1]
        wy = wy[:-1]
        ox, oy = [wx[0]], [wy[0]]
        for i in range(1, len(wx)):
            if not (wx[i] == wx[i - 1] and wy[i] == wy[i - 1]):
                ox.append(wx[i])
                oy.append(wy[i])
        return np.asarray(ox, float), np.asarray(oy, float)

    @staticmethod
    def isoline(x: np.ndarray, y: np.ndarray, z: np.ndarray, cval: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the ordered contour z(x, y) = cval (ISOLIJN) as 'isoline_python', compiled when numba is available.

        Parameters
        ----------
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid, with the shape (len(x), len(y))
        cval : float
            The value of the contour

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The x and y coordinates of the contour
        """
        if NUMBA_AVAILABLE:
            result = compiled_contour.isoline(x, y, z, cval)
            if result is not None:
                return result
        return Contour.isoline_python(x, y, z, cval)

    @staticmethod
    def merge_along_axis(wu: np.ndarray, ws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Merge the parts of a contour as a simplified CONTOUREN_SAMENVOEGEN: remove the sentinels and parts of a single
        point, sort on the second axis, and thin out the middle points where three or more points have the same value on
        that axis. The forcing to strictly increasing values and the interpolation are done by 'flatten_contour'. The
        full port is 'merge_contours'.

        Parameters
        ----------
        wu, ws : np.ndarray
            The coordinates of the contour on the first and second axis, with the sentinel between the parts

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The merged coordinates on the first and second axis
        """
        parts, current = [], []
        for u, s in zip(wu, ws):
            if s <= SENTINEL or u <= SENTINEL:
                if len(current) >= 2:
                    parts.append(current)
                current = []
            else:
                current.append((u, s))
        if len(current) >= 2:
            parts.append(current)
        points = [p for part in parts for p in part]
        if len(points) < 2:
            return np.empty(0), np.empty(0)
        points.sort(key=lambda p: p[1])
        u = np.array([p[0] for p in points], float)
        s = np.array([p[1] for p in points], float)

        # Three or more points on the same s: remove the middle ones
        keep = np.ones(len(s), dtype=bool)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j - i >= 2:
                for k in range(i + 1, j):
                    keep[k] = False
            i = j + 1
        return u[keep], s[keep]

    @staticmethod
    def merge_contours(wx: np.ndarray, wy: np.ndarray, ymin: float, ymax: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Merge the parts of a contour into one polyline (CONTOUREN_SAMENVOEGEN, 'contourbewerking.f90').

        The steps are in the order of the Fortran: remove parts of a single point; remove the middle one of three or
        more consecutive equal y values; remove 'flat' parts (first y equal to last y) if their end x values are equal
        or the y value lies on the edge of the grid; sort the parts on their first y; remove back-and-forth duplicates
        (three or four times the same y). Parts are not reversed: the reversal branch in the Fortran compares indices
        and is never taken in practice.

        Parameters
        ----------
        wx, wy : np.ndarray
            The x and y coordinates of the contour, with the sentinel between the parts
        ymin, ymax : float
            The edges of the grid on the y axis

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The x and y coordinates of the polyline
        """
        parts: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for x, y in zip(wx, wy):
            if x <= SENTINEL or y <= SENTINEL:
                if current:
                    parts.append(current)
                current = []
            else:
                current.append((float(x), float(y)))
        if current:
            parts.append(current)
        parts = [p for p in parts if len(p) >= 2]

        # Remove the middle one of three or more consecutive equal y values, per part
        cleaned = []
        for p in parts:
            keep = [p[0]]
            for i in range(1, len(p) - 1):
                if not (p[i][1] == p[i - 1][1] == p[i + 1][1]):
                    keep.append(p[i])
            keep.append(p[-1])
            cleaned.append(keep)
        parts = [p for p in cleaned if len(p) >= 2]

        # Remove flat parts
        kept = []
        for p in parts:
            if p[0][1] == p[-1][1] and (p[0][0] == p[-1][0] or p[0][1] in (ymin, ymax)):
                continue
            kept.append(p)
        parts = kept
        if not parts:
            return np.empty(0), np.empty(0)

        # Sort the parts on their first y and join them
        parts.sort(key=lambda p: p[0][1])
        points = [pt for p in parts for pt in p]

        # Back-and-forth duplicates: three or four times the same y
        if len(parts) > 1:
            i = 0
            while i < len(points) - 2:
                if points[i][1] == points[i + 2][1]:
                    if i + 3 < len(points) and points[i][1] == points[i + 3][1]:
                        xs = [points[i + j][0] for j in range(4)]
                        dup = [False] * 4
                        for a in range(4):
                            for b in range(a + 1, 4):
                                if xs[a] == xs[b]:
                                    dup[a] = dup[b] = True
                        if all(dup):
                            del points[i : i + 4]
                        else:
                            idx = [j for j in range(4) if dup[j]][:2]
                            for j in reversed(idx):
                                del points[i + j]
                        continue
                    # Three times the same y: remove one of the duplicate x values
                    if points[i][0] == points[i + 1][0]:
                        del points[i + 1]
                    else:
                        del points[i + 2]
                    i += 2
                    continue
                i += 1
        if len(points) < 2:
            return np.empty(0), np.empty(0)
        return np.asarray([p[0] for p in points], float), np.asarray([p[1] for p in points], float)

    @staticmethod
    def contour(x: np.ndarray, y: np.ndarray, z: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the contour of a level (ISOLIJN), followed by CONTOUREN_SAMENVOEGEN if the contour consists of more
        than one part.

        The merging is only done for a contour of more than one part (Hydra-NL tests on 'MINVAL(W) <= -9999'). A contour
        of one part keeps the order of the path that ISOLIJN followed. Keeping that order matters: sorting on the second
        axis makes an edge section monotonic in the first and makes the extrapolation flip there.

        Parameters
        ----------
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid, with the shape (len(x), len(y))
        level : float
            The level of the contour

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The x and y coordinates of the contour
        """
        wx, wy = Contour.isoline(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float), level)
        if wx.size == 0:
            return wx, wy
        if wy.min() <= SENTINEL or wx.min() <= SENTINEL:
            return Contour.merge_contours(wx, wy, float(y[0]), float(y[-1]))
        return wx, wy

    @staticmethod
    def refine(wx: np.ndarray, wy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Refine the contour with FACTOR = 10: each section in ten pieces, with the end point added.

        Parameters
        ----------
        wx, wy : np.ndarray
            The x and y coordinates of the contour

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The refined x and y coordinates
        """
        if len(wx) < 2:
            return np.empty(0), np.empty(0)
        fr = np.arange(FACTOR) / FACTOR
        xs = (wx[:-1, None] + (wx[1:, None] - wx[:-1, None]) * fr[None, :]).ravel()
        ys = (wy[:-1, None] + (wy[1:, None] - wy[:-1, None]) * fr[None, :]).ravel()
        return np.append(xs, wx[-1]), np.append(ys, wy[-1])

    @staticmethod
    def flatten_contour(wu: np.ndarray, ws: np.ndarray, umax: float, s_grid: np.ndarray) -> np.ndarray | None:
        """
        Determine the characteristic wind speed on a grid from a contour (VLAKKEN_CONTOUR, 'blokkans.f90').

        Parameters
        ----------
        wu : np.ndarray
            The wind speeds of the contour
        ws : np.ndarray
            The values of the slow stochastic of the contour
        umax : float
            The largest wind speed of the statistics (UMAX)
        s_grid : np.ndarray
            The grid of the slow stochastic

        Returns
        -------
        np.ndarray | None
            The characteristic wind speed on 's_grid'; None means 'no contour', and the caller then decides with the
            corner point rule
        """
        if len(ws) == 0:
            return None
        wu = np.array(wu, float)
        ws = np.array(ws, float)
        if len(ws) == 1 or np.isclose(ws[0], ws[-1]):
            # Horizontal contour: UMAX below the crossing ws, 0 above
            return np.where(s_grid < ws[0], umax, 0.0)
        if ws[0] > ws[-1]:
            ws = ws[::-1].copy()
            wu = wu[::-1].copy()
        eps = 2e-6
        for ip in range(len(ws) - 1):
            if ws[ip + 1] <= ws[ip]:
                ws[ip + 1] = ws[ip] * (1.0 + eps)
        u_characteristic = HydraInterpolate.inextrp1d_interp(s_grid, ws, wu)
        return np.clip(u_characteristic, 0.0, umax)

    @staticmethod
    def process_contour(
        wu: np.ndarray, wm: np.ndarray, s_grid: np.ndarray, u_stat: np.ndarray, ovkansu_r: np.ndarray, umax: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a contour into the characteristic wind speed and its exceedance probability (CONTOURVERWERKING,
        'contour.f90').

        Force strictly increasing contour values along the second axis (+1e-6), expand to 's_grid' (INEXTRP1D, linearly
        extrapolating), clip the characteristic wind speed to [0, UMAX] and read the exceedance probability, with the
        guard of the repaired 'contour.f90': a characteristic wind speed above UMAX gives a probability of exactly 0.

        Parameters
        ----------
        wu : np.ndarray
            The wind speeds of the contour
        wm : np.ndarray
            The values of the slow stochastic of the contour
        s_grid : np.ndarray
            The grid of the slow stochastic
        u_stat : np.ndarray
            The wind speeds of the statistics grid
        ovkansu_r : np.ndarray
            The exceedance probabilities of the wind speed on the statistics grid, for this wind direction
        umax : float
            The largest wind speed of the statistics (UMAX)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The characteristic wind speed and its exceedance probability on 's_grid'
        """
        wm = np.array(wm, float, copy=True)
        wu = np.asarray(wu, float)
        for ip in range(len(wm) - 1):
            if wm[ip + 1] <= wm[ip]:
                wm[ip + 1] = wm[ip] + 1.0e-6
        u_characteristic = np.maximum(HydraInterpolate.inextrp1d_fractions(wm, wu, s_grid), 0.0)
        ovk = np.maximum(
            HydraInterpolate.inextrp1d_fractions(u_stat, ovkansu_r, np.minimum(u_characteristic, umax)), 0.0
        )
        ovk[u_characteristic > umax] = 0.0
        return np.minimum(u_characteristic, umax), ovk


class WaveProcessing:
    """
    Wave processing (VERWERKGOLF).
    """

    @staticmethod
    def block_factors(times: np.ndarray) -> np.ndarray:
        """
        Calculate the block duration factors of VERWERKGOLF ('verwerkgolf.f90'): half a first block, and the second to
        last and last block according to the EINDDUUR convention.

        Parameters
        ----------
        times : np.ndarray
            The times of the wave shape

        Returns
        -------
        np.ndarray
            The factor per time
        """
        nt = len(times)
        block_duration = times[1] - times[0]
        end_duration = times[-1] - times[-2]
        f = np.ones(nt)
        f[0] = 0.5
        if nt == 2:
            # Wave shape of one block: last block half
            f[1] = 0.5
        elif end_duration < block_duration / 2.0:
            f[nt - 2] = ((block_duration / 2.0) + end_duration) / block_duration
            f[nt - 1] = 0.0
        else:
            f[nt - 2] = 1.0
            f[nt - 1] = (end_duration - (block_duration / 2.0)) / block_duration
        return f

    @staticmethod
    def wave_shapes_on_hydra_axis(
        settings, wave_type: str, peaks: np.ndarray, rise: float = 0.0, transition: float | None = None
    ):
        """
        Build the wave shapes on Hydra-NL's time axis: VECTORVULLEN(0, base duration, step) with NINT rounding.

        'WaveShape' builds the same trapezoids, but cuts the time axis at a multiple of the discretisation step (base
        duration 30 h gives the axis [0, 12, 24]); Hydra-NL calculates on [0, 12, 24, 30] with the centre of the
        trapezoid at half the base duration. For equal results the wave shapes are therefore built here on Hydra-NL's
        axis, with the builder of 'WaveShape'.

        Pass fresh settings: 'WaveShape' changes 'base_duration', so the settings of a location that has already been
        built are no longer the original ones.

        Parameters
        ----------
        settings : Settings
            Fresh settings of the location
        wave_type : str
            The name of a 'WaveShapeType' ("LAKE_LEVEL", "DISCHARGE", ...)
        peaks : np.ndarray
            The peak values
        rise : float, optional
            The climate rise of the peak values. Default is 0.0.
        transition : float | None, optional
            The shift of the wave shape (VERSCHUIVING), or None. Default is None.

        Returns
        -------
        tuple
            The times, the wave shapes (time, peak) and the 'WaveShape' object for the exceedance durations
        """
        base_duration = float(settings.base_duration)
        step = float(settings.waveshape_time_step)
        # Fortran NINT
        n = int(math.floor(base_duration / step + 0.5)) + 1
        times = np.array([i * step for i in range(n - 1)] + [base_duration])
        ws = WaveShape(copy.copy(settings), type=getattr(WaveShapeType, wave_type))
        ws.time, ws.ntime, ws.base_duration = times, len(times), base_duration
        ws.initialise_wave_shapes(np.asarray(peaks, float), climate_change=rise or 0.0)
        if transition:
            ws.transition_wave(transition)
        return times, np.asarray(ws.get_wave_shapes(), float), ws

    @staticmethod
    def exceedance_durations(
        ws, block: np.ndarray, wave_shapes: np.ndarray, block_duration_wind: float, clamp: bool
    ) -> np.ndarray:
        """
        Calculate how long a trapezoid lies above each block level (OVERSCHRIJDINGSDUUR_*), in units of the wind block
        duration.

        The two variants in Hydra-NL differ in content: the level variant (formula 6.3) clamps the duration to at least
        one block duration as soon as the trapezoid crosses the level, the discharge variant (formula 4.6) does not. If
        the level lies above the peak, the duration is 0.

        Parameters
        ----------
        ws : WaveShape
            The wave shape object of 'wave_shapes_on_hydra_axis'
        block : np.ndarray
            The block levels
        wave_shapes : np.ndarray
            The wave shapes, with the shape (time, peak)
        block_duration_wind : float
            The block duration of the wind
        clamp : bool
            True for the level variant, False for the discharge variant

        Returns
        -------
        np.ndarray
            The exceedance durations, with the shape (peak, block)
        """
        durations = ws.exceedance_timestamps(np.asarray(block, float))
        crossed = np.asarray(block, float)[None, :] <= np.asarray(wave_shapes, float).max(axis=0)[:, None]
        if clamp:
            durations = np.maximum(durations, block_duration_wind)
        return np.where(crossed, durations, 0.0) / block_duration_wind

    @staticmethod
    def frequency_density_1d(grid: np.ndarray, peak_weight: np.ndarray, durations: np.ndarray) -> np.ndarray:
        """
        Calculate the frequency density with one slow stochastic (FREQDICHTH*): sum over the peaks, then differentiate.

            OVFREQ(x) = sum over the peaks of delta_peak * density_peak * SCHALING_peak * DUREN(peak, x)
            f(x)      = -d/dx OVFREQ(x)

        This is both FREQDICHTHM (lakes, formula 6.3) and FREQDICHTHQ (discharge, formula 4.6 [Geerse, 2005.088]): two
        Fortran routines that differ only in their DUREN and their SCHALING index, which are both in the arguments here.

        Parameters
        ----------
        grid : np.ndarray
            The grid of the slow stochastic
        peak_weight : np.ndarray
            The product delta_peak * density_peak * SCHALING_peak per peak
        durations : np.ndarray
            The exceedance durations, with the shape (peak, grid)

        Returns
        -------
        np.ndarray
            The frequency density on the grid
        """
        ovfreq = (np.asarray(durations, float) * np.asarray(peak_weight, float)[:, None]).sum(axis=0)
        return Grid.probability_density_raw(grid, ovfreq)

    @staticmethod
    def process_wave_scaling(
        ovk_t: np.ndarray, factors: np.ndarray, block_duration: float, block_duration_wind: float
    ) -> np.ndarray:
        """
        Calculate only SCHALING of VERWERKGOLF; see 'process_wave'.

        Parameters
        ----------
        ovk_t : np.ndarray
            The block probability along the wave shape(s), with time as the first axis
        factors : np.ndarray
            The block duration factor per time
        block_duration : float
            The block duration of the wave shape
        block_duration_wind : float
            The block duration of the wind

        Returns
        -------
        np.ndarray
            SCHALING, with the shape of 'ovk_t' without the time axis
        """
        return WaveProcessing.process_wave(ovk_t, factors, block_duration, block_duration_wind)[1]

    @staticmethod
    def process_wave(
        ovk_t: np.ndarray, factors: np.ndarray, block_duration: float, block_duration_wind: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process the block probability along the wave shape(s) into OVKANS_KS and SCHALING (VERWERKGOLF).

        OVKANS_KS = 1 - prod (1 - p)^factor is the trapezoid probability: the exceedance probability over the whole
        wave, which COMBINEERGOLVEN_* sums over the peaks to the frequency line. SCHALING = GQM / G_SLANG is the
        saturated trapezoid probability divided by the linear sum, for the breakdown.

        'ovk_t' is the block probability along the wave shape(s), with time as the first axis followed by one axis per
        peak stochastic: (t, s) for one slow stochastic, (t, s, k) for two. The number of peak axes does not matter
        otherwise; 'factors' is broadcast over the time axis. How the block probability is evaluated on the wave shape
        (linearly along one wave, bilinearly on two simultaneous waves) is up to the caller; the saturation afterwards
        is the same for every water system.

        The MAXKANS branch (GQM set to 1 where the block probability reaches 1) and the lower bound of G_SLANG are as in
        the Fortran.

        Parameters
        ----------
        ovk_t : np.ndarray
            The block probability along the wave shape(s), with time as the first axis
        factors : np.ndarray
            The block duration factor per time
        block_duration : float
            The block duration of the wave shape
        block_duration_wind : float
            The block duration of the wind

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            OVKANS_KS and SCHALING, with the shape of 'ovk_t' without the time axis
        """
        ovk_t = np.minimum(np.asarray(ovk_t, float), 1.0)
        if block_duration != block_duration_wind:
            ovk_t = 1.0 - (1.0 - ovk_t) ** (block_duration / block_duration_wind)
        factors = np.asarray(factors, float).reshape((len(factors),) + (1,) * (ovk_t.ndim - 1))
        non_exceedance = (1.0 - ovk_t) ** factors
        ovkans_ks = 1.0 - np.prod(non_exceedance, axis=0)
        gqm = ovkans_ks.copy()
        # MAXKANS branch, only for GQM and not for OVKANS_KS
        gqm[np.isclose(ovk_t.max(axis=0), 1.0)] = 1.0
        g_slang = (1.0 - non_exceedance).sum(axis=0)
        return ovkans_ks, np.where(g_slang < 1.0e-7, 1.0, gqm / np.maximum(g_slang, 1.0e-30))


class Percentiles:
    """
    Percentiles of the breakdown (PERCENTIELENS).
    """

    @staticmethod
    def percentiles(s: np.ndarray, breakdown: np.ndarray) -> np.ndarray | None:
        """
        Calculate the percentiles of a breakdown (PERCENTIELENS): cumulate with the half bin convention, normalise and
        interpolate.

        Parameters
        ----------
        s : np.ndarray
            The grid of the stochastic
        breakdown : np.ndarray
            The breakdown on the grid

        Returns
        -------
        np.ndarray | None
            The value of the stochastic per percentile in 'PERCENTILES'; None if there is no mass, where Hydra-NL prints
            '-----'
        """
        ns = len(s)
        if breakdown.sum() <= 0.0:
            return None
        F = np.zeros(ns)
        if ns > 2:
            F[1] = breakdown[0] + breakdown[1] / 2.0
            for i in range(2, ns - 1):
                F[i] = F[i - 1] + (breakdown[i - 1] + breakdown[i]) / 2.0
            F[ns - 1] = F[ns - 2] + breakdown[ns - 2] / 2.0 + breakdown[ns - 1]
        else:
            F[1] = breakdown[0] + breakdown[1]
        if F[-1] <= 0.0:
            return None
        F = F / F[-1]
        out = np.empty(len(PERCENTILES))
        for ip, w in enumerate(PERCENTILES):
            D = ns - 1
            # Fortran IS = 2 .. NS-1
            for i in range(1, ns - 1):
                if F[i] > w:
                    D = i
                    break
            f0, f1 = F[D - 1], F[D]
            out[ip] = s[D] if f1 == f0 else s[D - 1] + (s[D] - s[D - 1]) * (w - f0) / (f1 - f0)
        return out

    @staticmethod
    def percentiles_per_barrier_state(s: np.ndarray, freqSK: np.ndarray) -> dict:
        """
        Calculate the percentiles in total and per barrier state (PERCENTIELEN_KERING).

        Parameters
        ----------
        s : np.ndarray
            The grid of the stochastic
        freqSK : np.ndarray
            The breakdown with the shape (grid, 2), column 0 = open and column 1 = closed: the numbering of the barrier
            states on the output side, as every water system with a barrier reports them

        Returns
        -------
        dict
            The percentiles for "open+closed", "open" and "closed"; None if the frequency is 0
        """
        freqK = freqSK.sum(axis=0)
        total = Percentiles.percentiles(s, freqSK.sum(axis=1))
        out = {"open+closed": None if total is None else [float(v) for v in total]}
        for k, name in ((0, "open"), (1, "closed")):
            pk = Percentiles.percentiles(s, freqSK[:, k]) if freqK[k] > 0.0 else None
            out[name] = None if pk is None else [float(v) for v in pk]
        return out

    @staticmethod
    def percentile_labels() -> list[int]:
        """
        Return the percentiles as whole percentages, as the tables print them in their header.

        Returns
        -------
        list[int]
            The percentiles in percent
        """
        return [int(round(p * 100)) for p in PERCENTILES]

    @staticmethod
    def breakdown_percentiles(freqRK: np.ndarray, stochastics: dict) -> dict:
        """
        Calculate the T00x_UITSPL output: the percentages per barrier state plus the percentiles per stochastic.

        Parameters
        ----------
        freqRK : np.ndarray
            The breakdown over wind direction and barrier state
        stochastics : dict
            {output name: (grid, freqSK)}. The caller chooses the names, because they depend on the water system: the
            same level axis is a lake level in one area and a sea water level in another.

        Returns
        -------
        dict
            The percentile labels, the percentages per barrier state and the percentiles per stochastic
        """
        freqK = np.asarray(freqRK, float).sum(axis=0)
        total = float(freqK.sum())
        out = {
            "pct": Percentiles.percentile_labels(),
            "barrier": {
                "open_pct": 100.0 * freqK[0] / total if total > 0 else 0.0,
                "closed_pct": 100.0 * freqK[1] / total if total > 0 else 0.0,
            },
        }
        for name, (grid, freqSK) in stochastics.items():
            out[name] = Percentiles.percentiles_per_barrier_state(grid, freqSK)
        return out

    @staticmethod
    def wind_percentiles_per_direction(breakdown: dict, barrier_state: str | None = None) -> dict:
        """
        Calculate the UR table: per wind direction the percentage (PERCENTAGE_UR: 100 * FREQR(r) / total FREQ) and the
        wind speed percentiles. A closed barrier only exists where it can close; elsewhere the frequency is 0 and there
        are no percentiles.

        Parameters
        ----------
        breakdown : dict
            The breakdown, with 'freqRK', 'freqUR', 'freqURK', 'r' and 'u'
        barrier_state : str | None, optional
            None for the UR table, "open" for the UR1 table and "closed" for the UR2 table. Default is None.

        Returns
        -------
        dict
            The percentile labels and per wind direction the percentage and the wind speed percentiles
        """
        total = float(breakdown["freqRK"].sum())
        if barrier_state is None:
            freq_r = breakdown["freqRK"].sum(axis=1)
            fur = breakdown["freqUR"]
        else:
            k = {"open": 0, "closed": 1}[barrier_state]
            freq_r = breakdown["freqRK"][:, k]
            fur = breakdown["freqURK"][:, :, k]
        directions = []
        for jr, r in enumerate(breakdown["r"]):
            pu = Percentiles.percentiles(breakdown["u"], fur[:, jr]) if freq_r[jr] > 0.0 else None
            directions.append(
                {
                    "r": float(r),
                    "percentage": (100.0 * freq_r[jr] / total) if total > 0.0 else 0.0,
                    "wind_speed": None if pu is None else [float(v) for v in pu],
                }
            )
        return {"pct": Percentiles.percentile_labels(), "directions": directions}

    @staticmethod
    def direction_contributions(breakdown: dict) -> dict:
        """
        Split the contributions over open and closed for the illustration point tables: per barrier state the
        percentage of that state plus the percentages of the exceedance frequency per wind direction (from FREQRKD, as
        'IPuitvoer_Zoet' prints them).

        Parameters
        ----------
        breakdown : dict
            The breakdown, with 'freqRK' and 'r'

        Returns
        -------
        dict
            Per barrier state the percentage and the percentage per wind direction
        """
        total = float(breakdown["freqRK"].sum())
        out = {}
        for k, name in ((0, "open"), (1, "closed")):
            per_r = {
                float(r): (100.0 * breakdown["freqRK"][jr, k] / total if total > 0 else 0.0)
                for jr, r in enumerate(breakdown["r"])
            }
            out[name] = {
                "pct": 100.0 * breakdown["freqRK"][:, k].sum() / total if total > 0 else 0.0,
                "directions": per_r,
            }
        return out


class Scaling:
    """
    Scaling of the breakdown (SCHAALUITSPLITS), the levels of the frequency line and the uncertainty combinations.
    """

    @staticmethod
    def scale_breakdown(breakdown: dict, target_frequency: float, families=FAMILIES) -> dict:
        """
        Scale the breakdown to the frequency line (SCHAALUITSPLITS), the last step of Hydra-NL's breakdown, for every water system ('uitsplitsingen_B.f90' and
        'uitsplitsingen_VIJ.f90' call the same routine, with their own copies for salt water and the Volkerak-Zoommeer).
        It is needed as soon as an absolute number is taken from the breakdown.

        Hydra-NL determines the exceedance frequency and the breakdown along two different routes, which do not give the
        same number:

          * the frequency line from the probability route: block probabilities combined over the wave shapes
            (COMBINEERGOLVEN_* and TRAPEZIUMKANS, then times NGEGEVENSBLOK). This is also the route of pydra;
          * the breakdown from the frequency density route: FREQDICHTH* and SPLITSEN*, with
            BIJDRAGE = ... * REAL(NGEGEVENSBLOK).

        Hydra-NL reconciles the two by scaling: per family the own sum FREQ, and everything multiplied by
        SCHALING = UITSPLITSING%FREQ(IN) / FREQ. For FREQ = 0 Hydra-NL sets MOGELIJK = .FALSE.; here the family is then
        not scaled (zero stays zero). Percentiles and percentages normalise the absolute scale away, so this step only
        matters for a frequency or return period read from the breakdown. The density route lies a few percent above the
        line, depending on the level, which is why Hydra-NL scales per level and not with one constant.

        Parameters
        ----------
        breakdown : dict
            The breakdown; it is not changed
        target_frequency : float
            The frequency line at this level; unit in is unit out
        families : tuple, optional
            The names of the breakdown arrays in 'breakdown' that are each scaled to their own sum. Default is
            'FAMILIES'.

        Returns
        -------
        dict
            A copy of the breakdown with the scaled families, 'target_frequency' and the scale per family
        """
        target = float(target_frequency)
        out = dict(breakdown)
        scales: dict[str, float] = {}
        for family in families:
            arr = np.asarray(breakdown[family], float)
            total = float(arr.sum())
            # FREQ = 0: Hydra-NL marks the breakdown as 'not possible'
            scale = target / total if total > 0.0 else 1.0
            scales[family] = scale
            out[family] = arr * scale
        out["target_frequency"] = target
        out["scale"] = scales
        return out

    @staticmethod
    def frequency_at_level(line: dict, level: float) -> float | None:
        """
        Interpolate the frequency line log-linearly at a level. There is deliberately no extrapolation: beyond the range
        of the line the frequency is an assumption, not a result, and it is used as the target value of
        'scale_breakdown'.

        Parameters
        ----------
        line : dict
            Any dict with 'level' and 'frequency'
        level : float
            The level

        Returns
        -------
        float | None
            The frequency at the level; None beyond the range of the line
        """
        lv = np.asarray(line["level"], float)
        fr = np.asarray(line["frequency"], float)
        ok = np.isfinite(lv) & np.isfinite(fr) & (fr > 0.0)
        lv, fr = lv[ok], fr[ok]
        if lv.size < 2 or not (lv.min() <= float(level) <= lv.max()):
            return None
        order = np.argsort(lv)
        return float(np.exp(np.interp(float(level), lv[order], np.log(fr[order]))))

    @staticmethod
    def generate_levels(
        frequency, requested_frequencies, hstart: float, hstep: float, hd: float, fmax: float
    ) -> list[float]:
        """
        Choose the levels of the frequency line as Hydra-NL does itself (GENEREER_HOOGTES_ZOET, 'hulpdijken_Zoet.f90',
        HULPDIJKENMETHODE 0).

        First a coarse search from 'hstart' in steps 'hstep' until the line comes below FBOVEN and above FONDER, then
        per bound an interpolation (log f), rounded to decimetres, until a search value comes back. In between an even
        grid with step 'hd'.

        Parameters
        ----------
        frequency : Callable
            The frequency at a level, called once per level; the caller keeps the results, so that the grid levels from
            the search are not calculated again
        requested_frequencies : list[float]
            The frequencies the user asked for
        hstart : float
            The first level of the coarse search
        hstep : float
            The step of the coarse search
        hd : float
            The step of the grid between the bounds
        fmax : float
            The sum of NGEGEVENSBLOK * KANSMASSA, the highest frequency the grid can reach

        Returns
        -------
        list[float]
            The levels of the frequency line
        """
        npmax = 50
        eps = 2.0e-6
        fonder = max(1.0e-1, max(requested_frequencies, default=0.0))
        fboven = 1.0e-6 if requested_frequencies and min(requested_frequencies) < 1.0e-5 else 1.0e-5

        def interpol(x1, x2, y1, y2, x):
            """
            Interpolate linearly (INTERPOL).

            Parameters
            ----------
            x1, x2 : float
                The coordinates of the two search points
            y1, y2 : float
                The levels of the two search points
            x : float
                The coordinate to interpolate at

            Returns
            -------
            float
                The interpolated level
            """
            # Hydra-NL also divides by zero here (INTERPOL) and continues with Inf or NaN
            if x2 == x1:
                raise ValueError(
                    f"[ERROR] When choosing the levels of the frequency line, the search points {y1} and {y2} m have "
                    f"the same frequency; there is nothing to interpolate in between."
                )
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

        hp = [float(hstart)]
        fp = [frequency(hp[0])]
        while fp[-1] > fboven and len(hp) < npmax:
            hp.append(round(hp[-1] + hstep, 6))
            fp.append(frequency(hp[-1]))
        fonder = min(fmax / 1.1, fonder)
        while max(fp) < fonder and len(hp) < npmax:
            hp.append(round(min(hp) - hstep, 6))
            fp.append(frequency(hp[-1]))

        # Upper bound: rounded up to decimetres
        proceed = True
        while proceed and len(hp) < npmax:
            h1, f1, h2, f2 = min(hp), max(fp), max(hp), min(fp)
            for h, f in zip(hp, fp):
                if f < fboven:
                    h2, f2 = min(h, h2), max(f, f2)
                else:
                    h1, f1 = max(h, h1), min(f, f1)
            if f1 > 0.0 and f2 > 0.0:
                hn = interpol(math.log(f1), math.log(f2), h1, h2, math.log(fboven))
            else:
                hn = interpol(f1, f2, h1, h2, fboven)
            hn = math.ceil(10.0 * hn) / 10.0
            proceed = min(abs(h - hn) for h in hp) > eps
            if proceed:
                hp.append(hn)
                fp.append(frequency(hn))

        # Lower bound: rounded down to decimetres. In the branch without logarithm Hydra-NL interpolates to FBOVEN, not
        # to FONDER; this is kept as it is.
        proceed = True
        while proceed and len(hp) < npmax:
            h1, f1, h2, f2 = min(hp), max(fp), max(hp), min(fp)
            for h, f in zip(hp, fp):
                if f > fonder:
                    h1, f1 = max(h, h1), min(f, f1)
                else:
                    h2, f2 = min(h, h2), max(f, f2)
            if f1 > 0.0 and f2 > 0.0:
                hn = interpol(math.log(f1), math.log(f2), h1, h2, math.log(fonder))
            else:
                hn = interpol(f1, f2, h1, h2, fboven)
            hn = math.floor(10.0 * hn) / 10.0
            proceed = min(abs(h - hn) for h in hp) > eps
            if proceed:
                hp.append(hn)
                fp.append(frequency(hn))

        # Lower bound = highest level with F > FONDER, upper bound = lowest level with F < FBOVEN
        h1, h2 = min(hp), max(hp)
        for h, f in zip(hp, fp):
            if f > fonder:
                h1 = max(h, h1)
            elif f < fboven:
                h2 = min(h, h2)
        if abs(h2 - h1) <= eps:
            h2 = h1 + hstep
        # NINT
        nn = math.floor((h2 - h1) / hd + 0.5) + 1
        return [round(h1 + i * hd, 6) for i in range(nn - 1)] + [h2]

    @staticmethod
    def uncertainty_combinations(
        stats, response: str, model_uncertainty: bool, closing_situation: int = 1
    ) -> list[tuple[float, float, float, float]]:
        """
        Return the combinations of model uncertainties with their probability, with the probabilities summing to 1. This is the integration table of Hydra-NL (ONZWAARDEN and KANSONZ). The
        probabilities come from 'discretise' (delta_h, via the bin edges) and
        'iterate_model_uncertainty_wave_conditions' (waves, including the correlation between Hs and Tspec). Without
        model uncertainty there is one combination with weight 1.

        'closing_situation' (1-based): in a water system with a storm surge barrier the model uncertainty may differ per
        barrier state. The HRD table 'UncertaintyModelFactor' has a column 'ClosingSituationId', and Hydra-NL shifts the
        water level per state with its own delta from ONZ_WS_K(IONZ, k) when WS_PAR_PER_KERING is on. With the default,
        the uncertainty of the first state is used for all states.

        The probabilities do not depend on the barrier state, the values do: the bin edges lie in standard normal units
        and only depend on the number of classes, not on mu and sigma ('modelonzekerheden.f90'). Lists for different
        states thus have the same weights in the same order.

        This only holds for the water level, not for the waves: ONZ_GH and ONZ_GP come from one column without barrier
        state ('modelonzekerheden.f90'), so the wave iterator keeps its own default.

        Not every HRD has a water level uncertainty; the coastal databases have none, and there
        'get_model_uncertainty("h")' returns None. Then delta_h = 0, as in the HBN calculation of pydra.

        Parameters
        ----------
        stats : Statistics
            The statistics of the location
        response : str
            The result variable; only "hbn" uses the wave uncertainties
        model_uncertainty : bool
            False gives one combination with weight 1
        closing_situation : int, optional
            The barrier state (1-based) of the water level uncertainty. Default is 1.

        Returns
        -------
        list[tuple[float, float, float, float]]
            The combinations (delta_h, f_Hm0, f_Tspec, KANSONZ)
        """
        if not model_uncertainty:
            return [(0.0, 1.0, 1.0, 1.0)]
        model_uncertainties = stats.get_model_uncertainties()
        mu_h = model_uncertainties.get_model_uncertainty("h", closing_situation)
        if mu_h is None:
            dh, p_dh = np.array([0.0]), np.array([1.0])
        else:
            dh, edges = mu_h.discretise(model_uncertainties.step_size["h"])
            dh = np.atleast_1d(dh).astype(float)
            # Imported here, so that the worker processes for crest levels do not load scipy.stats
            from scipy.stats import norm

            cdf = norm.cdf(np.asarray(edges, dtype=float))
            p_dh = cdf[1:] - cdf[:-1]
        if response == "hbn":
            waves = list(model_uncertainties.iterate_model_uncertainty_wave_conditions(wave_period="tspec"))
        else:
            waves = [(1.0, 1.0, 1.0)]
        return [(float(d), float(fhs), float(fts), float(pd * pw)) for d, pd in zip(dh, p_dh) for fhs, fts, pw in waves]
