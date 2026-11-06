"""
The compiled loops of the hydra subpackage against a reference in numpy or plain Python. Where the operations are
elementwise in the same order, the results must be bit for bit equal; only sums in a different order get a relative
tolerance.
"""

import numpy as np
import pytest
from scipy.special import ndtri

from pydra_core.hydra import compiled_contour, compiled_interpolation, library
from pydra_core.hydra import compiled_lower_river_breakdown as breakdown
from pydra_core.hydra import compiled_scans as scans
from pydra_core.hydra import compiled_wind_segments as wind

SEEDS = range(40)


def probabilities(rng, *shape):
    return rng.uniform(0.0, 1.0, shape) ** rng.uniform(0.2, 5.0)


def loading_field(rng, noise):
    nu, nm, nq = rng.integers(3, 12, 3)
    u = np.sort(rng.uniform(0.0, 40.0, nu))
    m = np.sort(rng.uniform(-1.0, 2.0, nm))
    q = np.sort(rng.uniform(100.0, 16000.0, nq))
    resp = 0.05 * u[:, None, None] + m[None, :, None] + 1e-4 * q[None, None, :] + noise * rng.normal(size=(nu, nm, nq))
    return u, m, q, resp


def block_grids(rng, m, q):
    mblock = np.sort(rng.uniform(m[0] - 0.3, m[-1] + 0.3, rng.integers(2, 8)))
    qblock = np.sort(rng.uniform(q[0] - 500.0, q[-1] + 500.0, rng.integers(2, 8)))
    return mblock, qblock


# Interpolation and contour


@pytest.mark.parametrize("seed", SEEDS)
def test_interpolation_compiled_equals_python(seed):
    rng = np.random.default_rng(seed)
    xp = np.sort(rng.uniform(0.0, 10.0, rng.integers(2, 20)))
    fp = rng.normal(size=len(xp))
    x = rng.uniform(-2.0, 12.0, 50)
    x[:3] = [np.nan, xp[0], xp[-1]]
    compiled = compiled_interpolation.inextrp1d_fractions(xp, fp, x)
    python = library.HydraInterpolate.inextrp1d_fractions_python(xp, fp, x)
    np.testing.assert_array_equal(compiled, python)


@pytest.mark.parametrize("seed", SEEDS)
def test_isoline_compiled_equals_python(seed):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 10.0, rng.integers(3, 15)))
    y = np.sort(rng.uniform(0.0, 10.0, rng.integers(3, 15)))
    z = rng.normal(size=(len(x), len(y))).cumsum(0).cumsum(1)
    level = float(np.quantile(z, rng.uniform(0.1, 0.9)))
    compiled_x, compiled_y = compiled_contour.isoline(x, y, z, level)
    python_x, python_y = library.Contour.isoline_python(x, y, z, level)
    np.testing.assert_array_equal(compiled_x, python_x)
    np.testing.assert_array_equal(compiled_y, python_y)


# Scans of the illustration points


def phi_hydra_reference(k):
    k = np.asarray(k, float)
    out = np.zeros_like(k)
    out[k <= 5.0e-16] = -scans.MAXINT
    out[1.0 - k <= 5.0e-16] = scans.MAXINT
    mid = (k > 5.0e-16) & (1.0 - k > 5.0e-16)
    out[mid] = np.clip(ndtri(np.clip(k[mid], scans.EPS, 1.0 - scans.EPS)), -scans.BIG, scans.BIG)
    return out


def test_phi_hydra_equals_scipy_ndtri():
    rng = np.random.default_rng(0)
    k = np.concatenate(
        [
            rng.uniform(0.0, 1.0, 10000),
            10.0 ** rng.uniform(-300.0, -1.0, 2000),
            1.0 - 10.0 ** rng.uniform(-16.0, -1.0, 2000),
            [np.nan, 0.0, 5e-16, 1e-12, 0.5, 1.0 - 1e-12, 1.0 - 5e-16, 1.0],
        ]
    )
    compiled = np.array([scans.phi_hydra(v) for v in k])
    np.testing.assert_array_equal(compiled, phi_hydra_reference(k))


def lin_reference(x1, x2, y1, y2, x):
    return y1 + (x - x1) / (x2 - x1) * (y2 - y1)


def coordinate_from_two(p1, p2, x1, x2, x):
    """
    Standard normal coordinate of the probability interpolated between p1 and p2, or MAXINT if they span 0.99.
    """
    extreme = (np.maximum(p1, p2) - np.minimum(p1, p2)) >= 0.99
    return np.where(extreme, scans.MAXINT, phi_hydra_reference(lin_reference(x1, x2, p1, p2, x)))


def coordinate_from_four(p11, p12, p21, p22, x1, x2, x, y1, y2, y):
    """
    As 'coordinate_from_two', bilinear: first along x, then along y.
    """
    high = np.maximum(np.maximum(p11, p12), np.maximum(p21, p22))
    low = np.minimum(np.minimum(p11, p12), np.minimum(p21, p22))
    along_y = lin_reference(y1, y2, lin_reference(x1, x2, p11, p12, x), lin_reference(x1, x2, p21, p22, x), y)
    return np.where(high - low >= 0.99, scans.MAXINT, phi_hydra_reference(along_y))


def scan_u_reference(wx, wy, mblock, qblock, oq, omdq, oudmq, ik, iu, u_value):
    ms, qs = library.Contour.refine(wx, wy)
    dqi = np.clip(np.searchsorted(qblock, qs), 1, len(qblock) - 1)
    dmi = np.clip(np.searchsorted(mblock, ms), 1, len(mblock) - 1)
    q1, q2 = qblock[dqi - 1], qblock[dqi]
    m1, m2 = mblock[dmi - 1], mblock[dmi]
    X = coordinate_from_two(oq[dqi - 1, ik], oq[dqi, ik], q1, q2, qs)
    Y = coordinate_from_four(
        omdq[dmi - 1, dqi - 1, ik], omdq[dmi - 1, dqi, ik], omdq[dmi, dqi - 1, ik], omdq[dmi, dqi, ik],
        q1, q2, qs, m1, m2, ms,
    )  # fmt: skip
    Z = coordinate_from_four(
        oudmq[iu, dmi - 1, dqi - 1, ik], oudmq[iu, dmi - 1, dqi, ik], oudmq[iu, dmi, dqi - 1, ik], oudmq[iu, dmi, dqi, ik],
        q1, q2, qs, m1, m2, ms,
    )  # fmt: skip
    d2 = X * X + Y * Y + Z * Z
    i = int(np.argmin(d2))
    return (d2[i], u_value, ms[i], qs[i], X[i] * X[i], Y[i] * Y[i], Z[i] * Z[i])


def scan_case(rng):
    """
    Block grids and probability tables for the scans, with a NaN, a 1 and a 0 in some cases.
    """
    _, m, q, _ = loading_field(rng, 0.0)
    mblock, qblock = block_grids(rng, m, q)
    ublock = np.sort(rng.uniform(0.0, 40.0, rng.integers(2, 8)))
    oq = probabilities(rng, len(qblock), 3)
    omdq = probabilities(rng, len(mblock), len(qblock), 3)
    oudmq = probabilities(rng, len(ublock), len(mblock), len(qblock), 3)
    if rng.uniform() < 0.3:
        oq.flat[0] = np.nan
        omdq.flat[1] = 1.0
        oudmq.flat[2] = 0.0
    return ublock, mblock, qblock, oq, omdq, oudmq, int(rng.integers(0, 3))


@pytest.mark.parametrize("seed", SEEDS)
def test_scan_u_kernel_equals_numpy_reference(seed):
    rng = np.random.default_rng(seed)
    ublock, mblock, qblock, oq, omdq, oudmq, ik = scan_case(rng)
    n = int(rng.integers(2, 8))
    wx = rng.uniform(mblock[0] - 0.2, mblock[-1] + 0.2, n)
    wy = rng.uniform(qblock[0] - 300.0, qblock[-1] + 300.0, n)
    iu = int(rng.integers(0, len(ublock)))
    compiled = scans.scan_u_kernel(wx, wy, mblock, qblock, oq, omdq, oudmq, ik, iu, ublock[iu])
    reference = scan_u_reference(wx, wy, mblock, qblock, oq, omdq, oudmq, ik, iu, ublock[iu])
    np.testing.assert_array_equal(np.array(compiled), np.array(reference))


@pytest.mark.parametrize("seed", SEEDS)
def test_scan_q_and_m_kernels_compiled_equal_interpreted(seed):
    rng = np.random.default_rng(seed)
    ublock, mblock, qblock, oq, omdq, oudmq, ik = scan_case(rng)
    n = int(rng.integers(1, 8))
    wu = rng.uniform(ublock[0] - 2.0, ublock[-1] + 2.0, n)
    wm = rng.uniform(mblock[0] - 0.2, mblock[-1] + 0.2, n)
    wq = rng.uniform(qblock[0] - 300.0, qblock[-1] + 300.0, n)
    iq = int(rng.integers(0, len(qblock)))
    im = int(rng.integers(0, len(mblock)))
    xq = float(rng.normal())

    arguments_q = (wu, wm, ublock, mblock, omdq, oudmq, ik, iq, qblock[iq], xq)
    compiled = scans.scan_q_kernel(*arguments_q)
    interpreted = scans.scan_q_kernel.py_func(*arguments_q)
    np.testing.assert_array_equal(np.array(compiled), np.array(interpreted))

    arguments_m = (wu, wq, ublock, qblock, oq, omdq, oudmq, ik, im, mblock[im])
    compiled = scans.scan_m_kernel(*arguments_m)
    interpreted = scans.scan_m_kernel.py_func(*arguments_m)
    np.testing.assert_array_equal(np.array(compiled), np.array(interpreted))


def isoline_parts(x, y, z, level):
    """
    The parts of the isoline with at least two points, split on the sentinels.
    """
    wx, wy = library.Contour.isoline_python(x, y, z, level)
    parts = []
    begin = 0
    for i in range(len(wx) + 1):
        if i < len(wx) and not (wx[i] <= compiled_contour.SENTINEL or wy[i] <= compiled_contour.SENTINEL):
            continue
        if i - begin >= 2:
            parts.append((wx[begin:i], wy[begin:i]))
        begin = i + 1
    return parts


@pytest.mark.parametrize("seed", SEEDS)
def test_scan_over_parts_equals_best_kernel_per_part(seed):
    rng = np.random.default_rng(seed)
    ublock, mblock, qblock, oq, omdq, oudmq, ik = scan_case(rng)
    respu = rng.normal(size=(len(ublock), len(mblock), len(qblock))).cumsum(1)
    level = float(rng.normal())

    best = None
    for iu in range(len(ublock)):
        for px, py in isoline_parts(mblock, qblock, respu[iu], level):
            result = scans.scan_u_kernel(px, py, mblock, qblock, oq, omdq, oudmq, ik, iu, ublock[iu])
            if np.isfinite(result[0]) and (best is None or result[0] < best[0]):
                best = result

    found, compiled = scans.scan_u_parts(mblock, qblock, respu, level, mblock, qblock, oq, omdq, oudmq, ik, ublock)
    assert found == (0 if best is None else 1)
    if best is not None:
        np.testing.assert_array_equal(np.array(compiled), np.array(best))


# Block probability over wind segments


def probability_above_reference(x, u_stat, ovkansu, umax):
    if x > umax:
        return 0.0
    return float(library.HydraInterpolate.inextrp1d_fractions_python(u_stat, ovkansu, x)[0])


def wind_statistics(rng):
    u_stat = np.linspace(0.0, 45.0, 30)
    ovkansu = np.sort(probabilities(rng, 30))[::-1]
    return u_stat, ovkansu, float(rng.uniform(30.0, 45.0))


@pytest.mark.parametrize("seed", SEEDS)
def test_block_probability_equals_sum_over_segments(seed):
    rng = np.random.default_rng(seed)
    u, m, q, resp = loading_field(rng, 0.3)
    mblock, qblock = block_grids(rng, m, q)
    level = float(np.quantile(resp, rng.uniform(0.1, 0.9)))
    u_stat, ovkansu, umax = wind_statistics(rng)

    compiled = wind.block_probability(u, m, q, resp, mblock, qblock, level, u_stat, ovkansu, umax)
    begin, end = wind.segments(u, m, q, resp, mblock, qblock, level)
    for im in range(len(mblock)):
        for iq in range(len(qblock)):
            p = 0.0
            for b, e in zip(begin[im, iq], end[im, iq], strict=True):
                if np.isnan(b):
                    continue
                p_end = 0.0 if np.isinf(e) else probability_above_reference(e, u_stat, ovkansu, umax)
                p += probability_above_reference(b, u_stat, ovkansu, umax) - p_end
            assert compiled[im, iq] == max(p, 0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_block_probability_rising_with_wind_is_probability_above_crossing(seed):
    rng = np.random.default_rng(seed)
    u, m, q, _ = loading_field(rng, 0.0)
    h_u = np.sort(rng.uniform(0.0, 5.0, len(u)))
    resp = np.broadcast_to(h_u[:, None, None], (len(u), len(m), len(q))).copy()
    mblock, qblock = block_grids(rng, m, q)
    level = float(rng.uniform(h_u[0], h_u[-1]))
    u_stat, ovkansu, umax = wind_statistics(rng)

    k = int(np.searchsorted(h_u, level))
    if k == 0:
        expected = probability_above_reference(u[0], u_stat, ovkansu, umax)
    else:
        u_cross = u[k - 1] + (level - h_u[k - 1]) / (h_u[k] - h_u[k - 1]) * (u[k] - u[k - 1])
        expected = probability_above_reference(u_cross, u_stat, ovkansu, umax)
    compiled = wind.block_probability(u, m, q, resp, mblock, qblock, level, u_stat, ovkansu, umax)
    np.testing.assert_allclose(compiled, expected, rtol=1e-12, atol=0.0)


def wave_case(rng):
    nm, nq = rng.integers(2, 8, 2)
    nt, ns, nk = rng.integers(2, 9, 3)
    ovkans_qm = probabilities(rng, nm, nq)
    mio = rng.integers(0, nm - 1, (nt, ns))
    qio = rng.integers(0, nq - 1, (nt, nk))
    mfr = rng.uniform(-0.2, 1.2, (nt, ns))
    qfr = rng.uniform(-0.2, 1.2, (nt, nk))
    return ovkans_qm, mio, mfr, qio, qfr


def on_waves_reference(ovkans_qm, mio, mfr, qio, qfr):
    im, mf = mio[:, :, None], mfr[:, :, None]
    iq, qf = qio[:, None, :], qfr[:, None, :]
    fy1 = ovkans_qm[im, iq] * (1.0 - mf) + ovkans_qm[im + 1, iq] * mf
    fy2 = ovkans_qm[im, iq + 1] * (1.0 - mf) + ovkans_qm[im + 1, iq + 1] * mf
    return fy1 + (fy2 - fy1) * qf


@pytest.mark.parametrize("seed", SEEDS)
def test_on_waves_equals_numpy_reference(seed):
    rng = np.random.default_rng(seed)
    case = wave_case(rng)
    np.testing.assert_array_equal(wind.on_waves(*case), on_waves_reference(*case))


@pytest.mark.parametrize("block_durations", [(12.0, 12.0), (24.0, 12.0)])
@pytest.mark.parametrize("seed", SEEDS)
def test_trapezium_probability_equals_numpy_reference(seed, block_durations):
    rng = np.random.default_rng(seed)
    case = wave_case(rng)
    factors = rng.uniform(0.5, 1.0, case[1].shape[0])
    block_duration, block_duration_wind = block_durations

    p = np.minimum(on_waves_reference(*case), 1.0)
    if block_duration != block_duration_wind:
        p = 1.0 - (1.0 - p) ** (block_duration / block_duration_wind)
    below = np.ones(p.shape[1:])
    for t in range(p.shape[0]):
        below *= (1.0 - p[t]) ** factors[t]

    compiled = wind.trapezium_probability_on_waves(*case, factors, block_duration, block_duration_wind)
    np.testing.assert_array_equal(compiled, 1.0 - below)


# Masked breakdown of the Rhine-Meuse delta


@pytest.mark.parametrize("seed", SEEDS)
def test_masked_sum_equals_numpy_reference(seed):
    rng = np.random.default_rng(seed)
    _, _, _, resp = loading_field(rng, 0.3)
    nu, nm, nq = resp.shape
    na, nb, nc = rng.integers(2, 10, 3)
    ju, jm, jq = rng.integers(0, nu - 1, na), rng.integers(0, nm - 1, nb), rng.integers(0, nq - 1, nc)
    fu, fm, fq = rng.uniform(0.0, 1.0, na), rng.uniform(0.0, 1.0, nb), rng.uniform(0.0, 1.0, nc)
    joint = probabilities(rng, na, nb, nc)
    du, dm, weight_q = rng.uniform(0.0, 1.0, na), rng.uniform(0.0, 1.0, nb), rng.uniform(0.0, 1.0, nc)
    level = float(np.quantile(resp, rng.uniform(0.1, 0.9)))

    r_u = resp[ju] * (1.0 - fu[:, None, None]) + resp[ju + 1] * fu[:, None, None]
    r_um = r_u[:, jm] * (1.0 - fm[None, :, None]) + r_u[:, jm + 1] * fm[None, :, None]
    response = r_um[:, :, jq] * (1.0 - fq) + r_um[:, :, jq + 1] * fq
    contribution = np.where(response > level, joint * du[:, None, None] * dm[None, :, None] * weight_q, 0.0)

    out_q, out_m, out_u = np.zeros(nc), np.zeros(nb), np.zeros(na)
    breakdown.masked_sum(resp, ju, fu, jm, fm, jq, fq, joint, du, dm, weight_q, level, out_q, out_m, out_u)
    np.testing.assert_allclose(out_q, contribution.sum(axis=(0, 1)), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(out_m, contribution.sum(axis=(0, 2)), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(out_u, contribution.sum(axis=(1, 2)), rtol=1e-12, atol=0.0)
