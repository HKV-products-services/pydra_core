"""
The block probability P(H >= h | m, q) as a sum over all wind segments where the loading is above the level, as a
compiled loop. How this relates to Hydra-NL's BLOKKENREKENEN_VIJ is described in docs/hydra_extensions.qmd.
"""

import numpy as np

from numba import njit

from .library import Grid


@njit(cache=True, nogil=True)
def probability_above(x, u_stat, ovkansu, umax):
    """
    Calculate P(U >= x) with INEXTRP1D on the statistics grid; 0 beyond UMAX.

    Parameters
    ----------
    x : float
        The wind speed
    u_stat : np.ndarray
        The wind speeds of the statistics grid
    ovkansu : np.ndarray
        The exceedance probabilities of the wind speed on the statistics grid
    umax : float
        The largest wind speed of the statistics (UMAX)

    Returns
    -------
    float
        The exceedance probability of the wind speed
    """
    if x > umax:
        return 0.0
    n = u_stat.shape[0]
    lo = 0
    hi = n
    # bisect_left, as np.searchsorted
    while lo < hi:
        mid = (lo + hi) // 2
        if u_stat[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    j = min(max(lo - 1, 0), n - 2)
    fr = (x - u_stat[j]) / (u_stat[j + 1] - u_stat[j])
    return ovkansu[j] * (1.0 - fr) + ovkansu[j + 1] * fr


@njit(cache=True, nogil=True)
def edge_crossing(x0, x1, h0, h1, lvl):
    """
    Find the position on a cell edge where h crosses the level.

    Parameters
    ----------
    x0, x1 : float
        The positions of the ends of the edge
    h0, h1 : float
        The loading at the ends of the edge
    lvl : float
        The level

    Returns
    -------
    float
        The position of the crossing, or NaN if h does not cross the level on this edge
    """
    if (h0 >= lvl) == (h1 >= lvl):
        return np.nan
    return x0 + (lvl - h0) / (h1 - h0) * (x1 - x0)


@njit(cache=True, nogil=True)
def crossing(u0, u1, ha0, hb0, ha1, hb1, f_m, h0, h1, lvl):
    """
    Find the wind speed where the line 'lake level = f_m' crosses the isoline, in the cell (u0, u1) x (a, b).

    As Hydra-NL's isoline (ISOLIJN, CONTOURVERWERKING): the piece of isoline in the cell is straight between the two
    points where it crosses the cell edges. If the cell does not have exactly two edge crossings (a saddle), or the
    intersection falls outside the cell, the crossing is linear in the wind between h0 and h1, the values on the
    line itself.

    Parameters
    ----------
    u0, u1 : float
        The wind speeds of the cell
    ha0, hb0 : float
        The loading at the lake level nodes a and b at wind speed u0
    ha1, hb1 : float
        The loading at the lake level nodes a and b at wind speed u1
    f_m : float
        The fraction of the line between the lake level nodes a and b
    h0, h1 : float
        The loading on the line at wind speeds u0 and u1
    lvl : float
        The level

    Returns
    -------
    float
        The wind speed of the crossing
    """
    # The four edges: wind u0 and u1 (crossing in f), lake level node a and b (crossing in u)
    n = 0
    pu0 = pf0 = pu1 = pf1 = 0.0
    for edge in range(4):
        if edge == 0:
            pu, pf = u0, edge_crossing(0.0, 1.0, ha0, hb0, lvl)
        elif edge == 1:
            pu, pf = u1, edge_crossing(0.0, 1.0, ha1, hb1, lvl)
        elif edge == 2:
            pu, pf = edge_crossing(u0, u1, ha0, ha1, lvl), 0.0
        else:
            pu, pf = edge_crossing(u0, u1, hb0, hb1, lvl), 1.0
        if np.isnan(pu) or np.isnan(pf):
            continue
        if n == 0:
            pu0, pf0 = pu, pf
        else:
            pu1, pf1 = pu, pf
        n += 1
    if n == 2 and pf1 != pf0:
        t = (f_m - pf0) / (pf1 - pf0)
        if 0.0 <= t <= 1.0:
            return pu0 + t * (pu1 - pu0)
    return u0 + (lvl - h0) / (h1 - h0) * (u1 - u0)


@njit(cache=True, nogil=True)
def scan(u, resp, jm, fm, jq, fq, lvl, fill_segments, begin, end, u_stat, ovkansu, umax, out):
    """
    Walk along the wind nodes per block cell and find where h crosses the level. With 'fill_segments' the segments
    go into 'begin' and 'end'; without, P(U >= begin) - P(U >= end) is summed per segment into 'out'.

    Parameters
    ----------
    u : np.ndarray
        The wind speeds of the response table
    resp : np.ndarray
        The response table, with the shape (u, m, q)
    jm, fm : np.ndarray
        The lower index and fraction of each block lake level on the lake level axis of the table
    jq, fq : np.ndarray
        The lower index and fraction of each block discharge on the discharge axis of the table
    lvl : float
        The level
    fill_segments : bool
        True to fill 'begin' and 'end', False to fill 'out'
    begin, end : np.ndarray
        The begin and end of the segments per block cell, filled in place with 'fill_segments'
    u_stat : np.ndarray
        The wind speeds of the statistics grid
    ovkansu : np.ndarray
        The exceedance probabilities of the wind speed on the statistics grid
    umax : float
        The largest wind speed of the statistics (UMAX)
    out : np.ndarray
        The block probability per block cell, filled in place without 'fill_segments'
    """
    nu = u.shape[0]
    for im in range(jm.shape[0]):
        a = jm[im]
        f_m = fm[im]
        for iq in range(jq.shape[0]):
            c = jq[iq]
            f_q = fq[iq]
            p = 0.0
            n = 0
            above = False
            u_begin = u[0]
            h_prev = 0.0
            ha_prev = 0.0
            hb_prev = 0.0
            for k in range(nu):
                h_a = resp[k, a, c] * (1.0 - f_q) + resp[k, a, c + 1] * f_q
                h_b = resp[k, a + 1, c] * (1.0 - f_q) + resp[k, a + 1, c + 1] * f_q
                h = h_a * (1.0 - f_m) + h_b * f_m
                now_above = h >= lvl
                if k == 0:
                    if fill_segments and now_above:
                        begin[im, iq, 0] = u[0]
                    above = now_above
                elif now_above != above:
                    u_cross = crossing(u[k - 1], u[k], ha_prev, hb_prev, h_a, h_b, f_m, h_prev, h, lvl)
                    if now_above:
                        u_begin = u_cross
                        if fill_segments:
                            begin[im, iq, n] = u_cross
                    elif fill_segments:
                        end[im, iq, n] = u_cross
                        n += 1
                    else:
                        p += probability_above(u_begin, u_stat, ovkansu, umax) - probability_above(
                            u_cross, u_stat, ovkansu, umax
                        )
                    above = now_above
                h_prev = h
                ha_prev = h_a
                hb_prev = h_b
            if fill_segments:
                if above:
                    end[im, iq, n] = np.inf
            else:
                if above:
                    p += probability_above(u_begin, u_stat, ovkansu, umax)
                out[im, iq] = max(p, 0.0)


def block_probability(u, m, q, resp, mblock, qblock, lvl, u_stat, ovkansu, umax) -> np.ndarray:
    """
    Calculate the block probability P(H >= lvl | m, q) on the block grid.

    Parameters
    ----------
    u, m, q : np.ndarray
        The wind speeds, lake levels and discharges of the response table
    resp : np.ndarray
        The response table, with the shape (u, m, q)
    mblock, qblock : np.ndarray
        The block grids of the lake level and the discharge
    lvl : float
        The level
    u_stat : np.ndarray
        The wind speeds of the statistics grid
    ovkansu : np.ndarray
        The exceedance probabilities of the wind speed on the statistics grid
    umax : float
        The largest wind speed of the statistics (UMAX)

    Returns
    -------
    np.ndarray
        The block probability, with the shape (len(mblock), len(qblock))
    """
    jm, fm = Grid.lower_index_and_fraction(np.asarray(m, float), np.asarray(mblock, float))
    jq, fq = Grid.lower_index_and_fraction(np.asarray(q, float), np.asarray(qblock, float))
    out = np.empty((len(mblock), len(qblock)))
    empty = np.empty((0, 0, 0))
    scan(
        np.ascontiguousarray(u, float),
        np.ascontiguousarray(resp, float),
        jm,
        fm,
        jq,
        fq,
        float(lvl),
        False,
        empty,
        empty,
        np.ascontiguousarray(u_stat, float),
        np.ascontiguousarray(ovkansu, float),
        float(umax),
        out,
    )
    return out


def segments(u, m, q, resp, mblock, qblock, lvl) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the wind segments with h >= lvl per block cell. The table goes to the block grid in the same way as in
    'block_probability'.

    Parameters
    ----------
    u, m, q : np.ndarray
        The wind speeds, lake levels and discharges of the response table
    resp : np.ndarray
        The response table, with the shape (u, m, q)
    mblock, qblock : np.ndarray
        The block grids of the lake level and the discharge
    lvl : float
        The level

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The begin and end of the segments, with the shape (len(mblock), len(qblock), K); a segment up to the edge of
        the table has end = inf, unused places are NaN
    """
    jm, fm = Grid.lower_index_and_fraction(np.asarray(m, float), np.asarray(mblock, float))
    jq, fq = Grid.lower_index_and_fraction(np.asarray(q, float), np.asarray(qblock, float))
    # More segments do not fit between the wind nodes
    k = len(u) // 2 + 1
    begin = np.full((len(mblock), len(qblock), k), np.nan)
    end = np.full_like(begin, np.nan)
    empty = np.empty(0)
    scan(
        np.ascontiguousarray(u, float),
        np.ascontiguousarray(resp, float),
        jm,
        fm,
        jq,
        fq,
        float(lvl),
        True,
        begin,
        end,
        empty,
        empty,
        0.0,
        np.empty((0, 0)),
    )
    # Usually one or two segments are used: the rest is dropped
    used = max(int((~np.isnan(begin)).sum(axis=-1).max()), 1)
    return begin[:, :, :used], end[:, :, :used]


@njit(cache=True, nogil=True)
def on_waves(ovkans_qm, mio, mfr, qio, qfr):
    """
    Interpolate the block probability (m, q) bilinearly on the simultaneous lake level and discharge waves, for
    VERWERKGOLF.

    Parameters
    ----------
    ovkans_qm : np.ndarray
        The block probability, with the shape (m, q)
    mio, mfr : np.ndarray
        The lower index and fraction of the lake level wave on the block grid, with the shape (t, s)
    qio, qfr : np.ndarray
        The lower index and fraction of the discharge wave on the block grid, with the shape (t, k)

    Returns
    -------
    np.ndarray
        The block probability on the waves, with the shape (t, s, k)
    """
    nt, ns = mio.shape
    nk = qio.shape[1]
    out = np.empty((nt, ns, nk))
    for t in range(nt):
        for s in range(ns):
            im = mio[t, s]
            mf = mfr[t, s]
            for k in range(nk):
                iq = qio[t, k]
                qf = qfr[t, k]
                fy1 = ovkans_qm[im, iq] * (1.0 - mf) + ovkans_qm[im + 1, iq] * mf
                fy2 = ovkans_qm[im, iq + 1] * (1.0 - mf) + ovkans_qm[im + 1, iq + 1] * mf
                out[t, s, k] = fy1 + (fy2 - fy1) * qf
    return out


@njit(cache=True, nogil=True)
def trapezium_probability_on_waves(ovkans_qm, mio, mfr, qio, qfr, factors, block_duration, block_duration_wind):
    """
    Calculate OVKANS_KS (s, k) of 'WaveProcessing.process_wave' on 'on_waves', without the scaling and without the
    intermediate series.

    Parameters
    ----------
    ovkans_qm : np.ndarray
        The block probability, with the shape (m, q)
    mio, mfr : np.ndarray
        The lower index and fraction of the lake level wave on the block grid, with the shape (t, s)
    qio, qfr : np.ndarray
        The lower index and fraction of the discharge wave on the block grid, with the shape (t, k)
    factors : np.ndarray
        The trapezium weight of each time step
    block_duration : float
        The block duration of the waves
    block_duration_wind : float
        The block duration of the wind

    Returns
    -------
    np.ndarray
        The exceedance probability over the waves, with the shape (s, k)
    """
    nt, ns = mio.shape
    nk = qio.shape[1]
    out = np.empty((ns, nk))
    for s in range(ns):
        for k in range(nk):
            below = 1.0
            for t in range(nt):
                im = mio[t, s]
                mf = mfr[t, s]
                iq = qio[t, k]
                qf = qfr[t, k]
                fy1 = ovkans_qm[im, iq] * (1.0 - mf) + ovkans_qm[im + 1, iq] * mf
                fy2 = ovkans_qm[im, iq + 1] * (1.0 - mf) + ovkans_qm[im + 1, iq + 1] * mf
                p = min(fy1 + (fy2 - fy1) * qf, 1.0)
                if block_duration != block_duration_wind:
                    p = 1.0 - (1.0 - p) ** (block_duration / block_duration_wind)
                below *= (1.0 - p) ** factors[t]
            out[s, k] = 1.0 - below
    return out
