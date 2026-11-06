"""
The masked contributions of the Rhine-Meuse delta breakdown (FREQ_UMQKR, FREQDICHTHEID_B) in one compiled loop,
without intermediate arrays on the whole (u, m, q) grid.
"""

import numpy as np

from numba import njit


@njit(cache=True, nogil=True)
def masked_sum(resp, ju, fu, jm, fm, jq, fq, joint_k, du, dm, weight_q, level, out_q, out_m, out_u):
    """
    Add joint_k * du * dm * weight(q) where the response is above 'level' to 'out_q', 'out_m' and 'out_u'.

    Parameters
    ----------
    resp : np.ndarray
        The response table on the table grid, with the shape (u, m, q)
    ju, fu : np.ndarray
        The interval index and fraction of each block grid point on the u axis of the table
    jm, fm : np.ndarray
        The interval index and fraction of each block grid point on the m axis of the table
    jq, fq : np.ndarray
        The interval index and fraction of each block grid point on the q axis of the table
    joint_k : np.ndarray
        The probability density on the block grid, with the shape (u, m, q)
    du, dm : np.ndarray
        The cell widths of the block grid on the u and m axes
    weight_q : np.ndarray
        The weight of each block grid point on the q axis
    level : float
        The level the response has to exceed
    out_q, out_m, out_u : np.ndarray
        The marginals over q, m and u, added to in place
    """
    nm_tab, nq_tab = resp.shape[1], resp.shape[2]
    r_u = np.empty((nm_tab, nq_tab))
    r_um = np.empty((jm.shape[0], nq_tab))
    sum_q = np.empty(jq.shape[0])
    for a in range(ju.shape[0]):
        # u axis
        for b in range(nm_tab):
            for c in range(nq_tab):
                r_u[b, c] = resp[ju[a], b, c] * (1.0 - fu[a]) + resp[ju[a] + 1, b, c] * fu[a]
        # m axis
        for b in range(jm.shape[0]):
            for c in range(nq_tab):
                r_um[b, c] = r_u[jm[b], c] * (1.0 - fm[b]) + r_u[jm[b] + 1, c] * fm[b]
        # q axis, mask and sum; partial sums per u value and per row first, which is faster than adding every point
        # to the output
        sum_q[:] = 0.0
        sum_u = 0.0
        for b in range(jm.shape[0]):
            w_um = du[a] * dm[b]
            sum_m = 0.0
            for c in range(jq.shape[0]):
                response = r_um[b, jq[c]] * (1.0 - fq[c]) + r_um[b, jq[c] + 1] * fq[c]
                if response > level:
                    v = joint_k[a, b, c] * w_um * weight_q[c]
                    sum_q[c] += v
                    sum_m += v
            out_m[b] += sum_m
            sum_u += sum_m
        out_u[a] += sum_u
        for c in range(jq.shape[0]):
            out_q[c] += sum_q[c]
