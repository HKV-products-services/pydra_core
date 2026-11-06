"""
Compiled counterpart of 'HydraInterpolate.inextrp1d_fractions' (INEXTRP1D), bit for bit equal to the numpy route.
numba is optional: without it, the caller uses the numpy route.
"""

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True, nogil=True)
    def fractions_kernel(xp, fp, x, out):
        """
        Fill 'out' with INEXTRP1D(xp, fp, x).

        Parameters
        ----------
        xp : np.ndarray
            The grid, one-dimensional and contiguous
        fp : np.ndarray
            The values on the grid, one-dimensional and contiguous
        x : np.ndarray
            The points to interpolate at, one-dimensional and contiguous
        out : np.ndarray
            The interpolated values, with the length of 'x', filled in place
        """
        n = xp.shape[0]
        jmax = n - 2
        for k in range(x.shape[0]):
            xv = x[k]
            if xv != xv:
                # NaN: numpy places it at the end
                i = n
            else:
                # bisect_left, as side='left'
                lo = 0
                hi = n
                while lo < hi:
                    mid = (lo + hi) // 2
                    if xp[mid] < xv:
                        lo = mid + 1
                    else:
                        hi = mid
                i = lo
            # The clamp of np.clip(., 0, n - 2)
            j = i - 1
            if j < 0:
                j = 0
            elif j > jmax:
                j = jmax
            fr = (xv - xp[j]) / (xp[j + 1] - xp[j])
            out[k] = fp[j] * (1.0 - fr) + fp[j + 1] * fr

    def inextrp1d_fractions(xp, fp, x):
        """
        Interpolate linearly with extrapolation (INEXTRP1D), with the same signature and output as
        'HydraInterpolate.inextrp1d_fractions_python', but compiled.

        Parameters
        ----------
        xp : array_like
            The grid
        fp : array_like
            The values on the grid
        x : array_like
            The points to interpolate at

        Returns
        -------
        np.ndarray | None
            The interpolated values, with the shape of 'x'; None if the input is outside the range of the kernel (grids
            that are not one-dimensional, a grid shorter than two points, or an 'fp' shorter than 'xp'), so the caller
            uses the numpy route
        """
        xa = np.atleast_1d(np.asarray(x, float))
        xpa = np.asarray(xp, float)
        fpa = np.asarray(fp, float)
        if xpa.ndim != 1 or fpa.ndim != 1 or xpa.shape[0] < 2 or fpa.shape[0] < xpa.shape[0]:
            return None
        flat = np.ascontiguousarray(xa).ravel()
        out = np.empty(flat.shape[0])
        fractions_kernel(np.ascontiguousarray(xpa), np.ascontiguousarray(fpa), flat, out)
        return out.reshape(xa.shape)
