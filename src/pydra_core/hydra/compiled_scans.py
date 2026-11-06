"""
Compiled scans of the illustration points on a 3D loading field with a storm surge barrier (ILLU_PUNTU, ILLU_PUNTQ and
ILLU_PUNTM): per contour point the distance to the origin in the standard normal space, and the point with the
smallest distance. Bit for bit equal to the Python route; the reasons are in docs/hydra_extensions.qmd.
"""

import ctypes
import numpy as np

from numba import njit
from numba.extending import get_cython_function_address

from .compiled_contour import SENTINEL, SPACE_PER_POINT, isoline_kernel
from .library import FACTOR

_ndtri_address = get_cython_function_address("scipy.special.cython_special", "ndtri")
_ndtri = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(_ndtri_address)

# The value of a standard normal coordinate that stands for "beyond the grid" (Hydra-NL MAXINT)
MAXINT = 32767.0

# The clamp of the standard normal coordinate and of the probability before 'ndtri'
BIG = 40.0
EPS = 1e-12

# What a scan kernel returns if the contour yields nothing: 'np.inf' loses from every real distance
EMPTY = np.inf


@njit(cache=True, nogil=True)
def bracket(grid, v):
    """
    Find the interval of one value on a grid, as 'searchsorted(grid, v, "left")' clamped to [1, n - 1].

    Parameters
    ----------
    grid : np.ndarray
        The ascending grid
    v : float
        The value

    Returns
    -------
    int
        The index of the upper end of the interval
    """
    n = grid.shape[0]
    if v != v:
        # NaN: numpy places it at the end
        return n - 1
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if grid[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    if lo < 1:
        return 1
    if lo > n - 1:
        return n - 1
    return lo


@njit(cache=True, nogil=True)
def lin(x1, x2, y1, y2, x):
    """
    Interpolate linearly at one point (DINTERPOL).

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


@njit(cache=True, nogil=True)
def extremes2(a, b):
    """
    Determine whether two probabilities differ at least 0.99, (max - min) >= 0.99. NaN gives False, as numpy.

    Parameters
    ----------
    a, b : float
        The probabilities

    Returns
    -------
    bool
        True if the probabilities differ at least 0.99
    """
    if a != a or b != b:
        return False
    return (max(a, b) - min(a, b)) >= 0.99


@njit(cache=True, nogil=True)
def extremes4(a, b, c, d):
    """
    Determine whether four probabilities differ at least 0.99, (max - min) >= 0.99. NaN gives False, as numpy.

    Parameters
    ----------
    a, b, c, d : float
        The probabilities

    Returns
    -------
    bool
        True if the probabilities differ at least 0.99
    """
    if a != a or b != b or c != c or d != d:
        return False
    high = max(max(a, b), max(c, d))
    low = min(min(a, b), min(c, d))
    return (high - low) >= 0.99


@njit(nogil=True)
def phi_hydra(k):
    """
    Calculate the standard normal coordinate of a probability as Hydra-NL computes it, including the double clamp.

    NaN gives 0.0 and not NaN, as the Python route with 'np.where' does: all three branches are false for a NaN, so
    the value falls through to the default 0.0.

    Parameters
    ----------
    k : float
        The non-exceedance probability

    Returns
    -------
    float
        The standard normal coordinate
    """
    if k != k:
        return 0.0
    if k <= 5.0e-16:
        return -MAXINT
    if 1.0 - k <= 5.0e-16:
        return MAXINT
    x = k
    if x < EPS:
        x = EPS
    elif x > 1.0 - EPS:
        x = 1.0 - EPS
    v = _ndtri(x)
    if v < -BIG:
        v = -BIG
    elif v > BIG:
        v = BIG
    return v


@njit(cache=True, nogil=True)
def refine_into(wx, wy, xs, ys):
    """
    Refine a contour as 'Contour.refine' (FACTOR = 10), into preallocated buffers.

    Every section in ten pieces, then the last point: the same order as the 'ravel()' of the numpy route, per section
    i all ten fractions j.

    Parameters
    ----------
    wx, wy : np.ndarray
        The x and y coordinates of the contour
    xs, ys : np.ndarray
        The buffers of the refined x and y coordinates, filled in place

    Returns
    -------
    int
        The number of refined points
    """
    n = wx.shape[0]
    p = 0
    for i in range(n - 1):
        for j in range(FACTOR):
            fr = j / FACTOR
            xs[p] = wx[i] + (wx[i + 1] - wx[i]) * fr
            ys[p] = wy[i] + (wy[i + 1] - wy[i]) * fr
            p += 1
    xs[p] = wx[n - 1]
    ys[p] = wy[n - 1]
    return p + 1


@njit(nogil=True)
def scan_u_kernel(wx, wy, mblock, qblock, oq, omdq, oudmq, ik, iu, u_value):
    """
    Find the contour point closest to the origin for one block wind speed (ILLU_PUNTU): the contour runs over (lake
    level, discharge).

    Parameters
    ----------
    wx, wy : np.ndarray
        The lake levels and discharges of one part of the contour
    mblock, qblock : np.ndarray
        The block grids of the lake level and the discharge
    oq, omdq, oudmq : np.ndarray
        F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids
    ik : int
        The barrier state (0 open, 1 closed)
    iu : int
        The index of the block wind speed
    u_value : float
        The block wind speed

    Returns
    -------
    tuple
        (squared distance, u, m, q, X^2, Y^2, Z^2) of the closest point; the distance is 'EMPTY' if the part has fewer
        than two points
    """
    n = wx.shape[0]
    if n < 2:
        return (EMPTY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cap = FACTOR * (n - 1) + 1
    ms = np.empty(cap)
    qs = np.empty(cap)
    n_points = refine_into(wx, wy, ms, qs)

    best = EMPTY
    bi = -1
    bX2 = 0.0
    bY2 = 0.0
    bZ2 = 0.0
    for p in range(n_points):
        mv = ms[p]
        qv = qs[p]
        dqi = bracket(qblock, qv)
        q1 = qblock[dqi - 1]
        q2 = qblock[dqi]
        kx1 = oq[dqi - 1, ik]
        kx2 = oq[dqi, ik]
        if extremes2(kx1, kx2):
            X = MAXINT
        else:
            X = phi_hydra(lin(q1, q2, kx1, kx2, qv))

        dmi = bracket(mblock, mv)
        m1 = mblock[dmi - 1]
        m2 = mblock[dmi]
        vy0 = omdq[dmi - 1, dqi - 1, ik]
        vy1 = omdq[dmi - 1, dqi, ik]
        vy2 = omdq[dmi, dqi - 1, ik]
        vy3 = omdq[dmi, dqi, ik]
        if extremes4(vy0, vy1, vy2, vy3):
            Y = MAXINT
        else:
            ky1 = lin(q1, q2, vy0, vy1, qv)
            ky2 = lin(q1, q2, vy2, vy3, qv)
            Y = phi_hydra(lin(m1, m2, ky1, ky2, mv))

        vz0 = oudmq[iu, dmi - 1, dqi - 1, ik]
        vz1 = oudmq[iu, dmi - 1, dqi, ik]
        vz2 = oudmq[iu, dmi, dqi - 1, ik]
        vz3 = oudmq[iu, dmi, dqi, ik]
        if extremes4(vz0, vz1, vz2, vz3):
            Z = MAXINT
        else:
            kz1 = lin(q1, q2, vz0, vz1, qv)
            kz2 = lin(q1, q2, vz2, vz3, qv)
            Z = phi_hydra(lin(m1, m2, kz1, kz2, mv))

        X2 = X * X
        Y2 = Y * Y
        Z2 = Z * Z
        d2 = X2 + Y2 + Z2
        # As 'np.argmin': the first minimum wins
        if bi < 0 or d2 < best:
            best = d2
            bi = p
            bX2 = X2
            bY2 = Y2
            bZ2 = Z2
    return (best, u_value, ms[bi], qs[bi], bX2, bY2, bZ2)


@njit(nogil=True)
def scan_q_kernel(wx, wy, ublock, mblock, omdq, oudmq, ik, iq, q_value, xq):
    """
    Find the contour point closest to the origin for one block discharge (ILLU_PUNTQ): the contour runs over (wind
    speed, lake level).

    Parameters
    ----------
    wx, wy : np.ndarray
        The wind speeds and lake levels of one part of the contour
    ublock, mblock : np.ndarray
        The block grids of the wind speed and the lake level
    omdq, oudmq : np.ndarray
        F(m|q,k) and F(u|m,q,k) on the block grids
    ik : int
        The barrier state (0 open, 1 closed)
    iq : int
        The index of the block discharge
    q_value : float
        The block discharge
    xq : float
        The standard normal coordinate of the discharge, computed by the caller: it does not depend on the contour
        point

    Returns
    -------
    tuple
        (squared distance, u, m, q, X^2, Y^2, Z^2) of the closest point; the distance is 'EMPTY' if the part has fewer
        than two points
    """
    n = wx.shape[0]
    if n < 2:
        return (EMPTY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cap = FACTOR * (n - 1) + 1
    us = np.empty(cap)
    ms = np.empty(cap)
    n_points = refine_into(wx, wy, us, ms)

    X = xq
    X2 = X * X
    best = EMPTY
    bi = -1
    bY2 = 0.0
    bZ2 = 0.0
    for p in range(n_points):
        uv = us[p]
        mv = ms[p]
        dmi = bracket(mblock, mv)
        m1 = mblock[dmi - 1]
        m2 = mblock[dmi]
        k1 = omdq[dmi - 1, iq, ik]
        k2 = omdq[dmi, iq, ik]
        if extremes2(k1, k2):
            Y = MAXINT
        else:
            Y = phi_hydra(lin(m1, m2, k1, k2, mv))

        dui = bracket(ublock, uv)
        v0 = oudmq[dui - 1, dmi - 1, iq, ik]
        v1 = oudmq[dui - 1, dmi, iq, ik]
        v2 = oudmq[dui, dmi - 1, iq, ik]
        v3 = oudmq[dui, dmi, iq, ik]
        if extremes4(v0, v1, v2, v3):
            Z = MAXINT
        else:
            km1 = lin(m1, m2, v0, v1, mv)
            km2 = lin(m1, m2, v2, v3, mv)
            Z = phi_hydra(lin(ublock[dui - 1], ublock[dui], km1, km2, uv))

        Y2 = Y * Y
        Z2 = Z * Z
        d2 = X2 + Y2 + Z2
        if bi < 0 or d2 < best:
            best = d2
            bi = p
            bY2 = Y2
            bZ2 = Z2
    return (best, us[bi], ms[bi], q_value, X2, bY2, bZ2)


@njit(nogil=True)
def scan_m_kernel(wx, wy, ublock, qblock, oq, omdq, oudmq, ik, im, m_value):
    """
    Find the contour point closest to the origin for one block lake level (ILLU_PUNTM): the contour runs over (wind
    speed, discharge).

    Parameters
    ----------
    wx, wy : np.ndarray
        The wind speeds and discharges of one part of the contour
    ublock, qblock : np.ndarray
        The block grids of the wind speed and the discharge
    oq, omdq, oudmq : np.ndarray
        F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids
    ik : int
        The barrier state (0 open, 1 closed)
    im : int
        The index of the block lake level
    m_value : float
        The block lake level

    Returns
    -------
    tuple
        (squared distance, u, m, q, X^2, Y^2, Z^2) of the closest point; the distance is 'EMPTY' if the part has fewer
        than two points
    """
    n = wx.shape[0]
    if n < 2:
        return (EMPTY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cap = FACTOR * (n - 1) + 1
    us = np.empty(cap)
    qs = np.empty(cap)
    n_points = refine_into(wx, wy, us, qs)

    best = EMPTY
    bi = -1
    bX2 = 0.0
    bY2 = 0.0
    bZ2 = 0.0
    for p in range(n_points):
        uv = us[p]
        qv = qs[p]
        dqi = bracket(qblock, qv)
        q1 = qblock[dqi - 1]
        q2 = qblock[dqi]
        kx1 = oq[dqi - 1, ik]
        kx2 = oq[dqi, ik]
        if extremes2(kx1, kx2):
            X = MAXINT
        else:
            X = phi_hydra(lin(q1, q2, kx1, kx2, qv))

        ky1 = omdq[im, dqi - 1, ik]
        ky2 = omdq[im, dqi, ik]
        if extremes2(ky1, ky2):
            Y = MAXINT
        else:
            Y = phi_hydra(lin(q1, q2, ky1, ky2, qv))

        dui = bracket(ublock, uv)
        v0 = oudmq[dui - 1, im, dqi - 1, ik]
        v1 = oudmq[dui - 1, im, dqi, ik]
        v2 = oudmq[dui, im, dqi - 1, ik]
        v3 = oudmq[dui, im, dqi, ik]
        if extremes4(v0, v1, v2, v3):
            Z = MAXINT
        else:
            km1 = lin(q1, q2, v0, v1, qv)
            km2 = lin(q1, q2, v2, v3, qv)
            Z = phi_hydra(lin(ublock[dui - 1], ublock[dui], km1, km2, uv))

        X2 = X * X
        Y2 = Y * Y
        Z2 = Z * Z
        d2 = X2 + Y2 + Z2
        if bi < 0 or d2 < best:
            best = d2
            bi = p
            bX2 = X2
            bY2 = Y2
            bZ2 = Z2
    return (best, us[bi], m_value, qs[bi], bX2, bY2, bZ2)


# Whole scan per isoline part
# The scan runs over a few hundred slices, each with its own isoline. Below is the whole loop: isoline, duplicate
# points removed, split on the sentinels, the kernel per part, the best part kept. A full isoline buffer returns -1:
# the caller then does the whole scan along the Python route.


@njit(nogil=True)
def parts(x, y, z, level, bufx, bufy, ox, oy, begins, ends):
    """
    Split the isoline of a level into parts, without consecutive duplicate points.

    Parameters
    ----------
    x, y : np.ndarray
        The coordinates of the grid
    z : np.ndarray
        The values on the grid, with the shape (x, y)
    level : float
        The level of the isoline
    bufx, bufy : np.ndarray
        The buffers of the isoline, filled in place
    ox, oy : np.ndarray
        The buffers of the isoline without duplicate points, filled in place
    begins, ends : np.ndarray
        The begin and end of each part in 'ox' and 'oy', filled in place

    Returns
    -------
    int
        The number of parts, or -1 if the buffer is full
    """
    n = isoline_kernel(x, y, np.ascontiguousarray(z), level, bufx, bufy)
    if n < 0:
        return -1
    # Without the last sentinel
    m = n - 1
    if m <= 0:
        return 0
    ox[0] = bufx[0]
    oy[0] = bufy[0]
    k = 1
    for i in range(1, m):
        if not (bufx[i] == bufx[i - 1] and bufy[i] == bufy[i - 1]):
            ox[k] = bufx[i]
            oy[k] = bufy[i]
            k += 1
    n_parts = 0
    begin = 0
    for i in range(k + 1):
        if i < k and not (ox[i] <= SENTINEL or oy[i] <= SENTINEL):
            continue
        if i - begin >= 2:
            begins[n_parts] = begin
            ends[n_parts] = i
            n_parts += 1
        begin = i + 1
    return n_parts


@njit(nogil=True)
def buffers(nx, ny):
    """
    Allocate the buffers of 'parts' for a grid.

    Parameters
    ----------
    nx, ny : int
        The number of grid points on the x and y axes

    Returns
    -------
    tuple
        The buffers bufx, bufy, ox, oy, begins and ends
    """
    cap = SPACE_PER_POINT * nx * ny + 16
    return np.empty(cap), np.empty(cap), np.empty(cap), np.empty(cap), np.empty(cap, np.int64), np.empty(cap, np.int64)


@njit(nogil=True)
def scan_u_parts(m, q, respu, level, mblock, qblock, oq, omdq, oudmq, ik, ublock):
    """
    Find the contour point closest to the origin over all block wind speeds (ILLU_PUNTU).

    Parameters
    ----------
    m, q : np.ndarray
        The lake levels and discharges of the response table
    respu : np.ndarray
        The response on the block wind speeds, with the shape (u, m, q)
    level : float
        The level of the contour
    mblock, qblock : np.ndarray
        The block grids of the lake level and the discharge
    oq, omdq, oudmq : np.ndarray
        F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids
    ik : int
        The barrier state (0 open, 1 closed)
    ublock : np.ndarray
        The block grid of the wind speed

    Returns
    -------
    tuple
        (found, result): found is 1 if a point was found, 0 if not and -1 if the isoline buffer was full; result as
        'scan_u_kernel'
    """
    bufx, bufy, ox, oy, begins, ends = buffers(m.shape[0], q.shape[0])
    found = 0
    best = (EMPTY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for iu in range(ublock.shape[0]):
        n_parts = parts(m, q, respu[iu, :, :], level, bufx, bufy, ox, oy, begins, ends)
        if n_parts < 0:
            return -1, best
        for d in range(n_parts):
            r = scan_u_kernel(
                ox[begins[d] : ends[d]].copy(),
                oy[begins[d] : ends[d]].copy(),
                mblock,
                qblock,
                oq,
                omdq,
                oudmq,
                ik,
                iu,
                ublock[iu],
            )
            if not np.isfinite(r[0]):
                continue
            if found == 0 or r[0] < best[0]:
                best = r
                found = 1
    return found, best


@njit(nogil=True)
def scan_q_parts(u, m, respq, level, ublock, mblock, omdq, oudmq, ik, qblock, xq):
    """
    Find the contour point closest to the origin over all block discharges (ILLU_PUNTQ).

    Parameters
    ----------
    u, m : np.ndarray
        The wind speeds and lake levels of the response table
    respq : np.ndarray
        The response on the block discharges, with the shape (u, m, q)
    level : float
        The level of the contour
    ublock, mblock : np.ndarray
        The block grids of the wind speed and the lake level
    omdq, oudmq : np.ndarray
        F(m|q,k) and F(u|m,q,k) on the block grids
    ik : int
        The barrier state (0 open, 1 closed)
    qblock : np.ndarray
        The block grid of the discharge
    xq : np.ndarray
        The standard normal coordinate of each block discharge

    Returns
    -------
    tuple
        (found, result): found is 1 if a point was found, 0 if not and -1 if the isoline buffer was full; result as
        'scan_q_kernel'
    """
    bufx, bufy, ox, oy, begins, ends = buffers(u.shape[0], m.shape[0])
    found = 0
    best = (EMPTY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for iq in range(qblock.shape[0]):
        n_parts = parts(u, m, respq[:, :, iq], level, bufx, bufy, ox, oy, begins, ends)
        if n_parts < 0:
            return -1, best
        for d in range(n_parts):
            r = scan_q_kernel(
                ox[begins[d] : ends[d]].copy(),
                oy[begins[d] : ends[d]].copy(),
                ublock,
                mblock,
                omdq,
                oudmq,
                ik,
                iq,
                qblock[iq],
                xq[iq],
            )
            if not np.isfinite(r[0]):
                continue
            if found == 0 or r[0] < best[0]:
                best = r
                found = 1
    return found, best


@njit(nogil=True)
def scan_m_parts(u, q, respm, level, ublock, qblock, oq, omdq, oudmq, ik, mblock):
    """
    Find the contour point closest to the origin over all block lake levels (ILLU_PUNTM).

    Parameters
    ----------
    u, q : np.ndarray
        The wind speeds and discharges of the response table
    respm : np.ndarray
        The response on the block lake levels, with the shape (u, m, q)
    level : float
        The level of the contour
    ublock, qblock : np.ndarray
        The block grids of the wind speed and the discharge
    oq, omdq, oudmq : np.ndarray
        F(q|k), F(m|q,k) and F(u|m,q,k) on the block grids
    ik : int
        The barrier state (0 open, 1 closed)
    mblock : np.ndarray
        The block grid of the lake level

    Returns
    -------
    tuple
        (found, result): found is 1 if a point was found, 0 if not and -1 if the isoline buffer was full; result as
        'scan_m_kernel'
    """
    bufx, bufy, ox, oy, begins, ends = buffers(u.shape[0], q.shape[0])
    found = 0
    best = (EMPTY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for im in range(mblock.shape[0]):
        n_parts = parts(u, q, respm[:, im, :], level, bufx, bufy, ox, oy, begins, ends)
        if n_parts < 0:
            return -1, best
        for d in range(n_parts):
            r = scan_m_kernel(
                ox[begins[d] : ends[d]].copy(),
                oy[begins[d] : ends[d]].copy(),
                ublock,
                qblock,
                oq,
                omdq,
                oudmq,
                ik,
                im,
                mblock[im],
            )
            if not np.isfinite(r[0]):
                continue
            if found == 0 or r[0] < best[0]:
                best = r
                found = 1
    return found, best
