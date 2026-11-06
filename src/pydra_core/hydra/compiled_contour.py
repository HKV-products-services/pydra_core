"""
Compiled counterpart of 'Contour.isoline_python' (ISOLIJN, marching squares), bit for bit equal to the Python route.
numba is optional: without it, or when the output buffer is too small, the caller uses the Python route.
"""

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# Separator between the parts of a contour in the ISOLIJN output
SENTINEL = -9999.0

# Space per grid point in the output buffer. Marching squares yields at most two points per cell, so four leaves room
# for the sentinels. On overflow the Python route takes over.
SPACE_PER_POINT = 4


if NUMBA_AVAILABLE:

    @njit(cache=True, nogil=True)
    def interpol(z1, z2, c1, c2, cval):
        """
        Interpolate the coordinate where the value 'cval' lies on the edge between two grid points.

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

    @njit(cache=True, nogil=True)
    def next_cell(ix, iy, nix, niy, z, cval, ilog):
        """
        Determine the next cell of the contour (VOLGENDEVAK), with the four corner indices in separate arrays instead
        of tuples.

        Parameters
        ----------
        ix, iy : np.ndarray
            The x and y indices of the four corners of the current cell
        nix, niy : np.ndarray
            The x and y indices of the four corners of the next cell, filled in place
        z : np.ndarray
            The values on the grid
        cval : float
            The value of the contour
        ilog : np.ndarray
            The horizontal edges the contour still has to start from, updated in place
        """
        dx = ix[1] - ix[0]
        dy = iy[1] - iy[0]
        if dx == 1:
            ilog[ix[0], iy[0]] = False
        right = z[ix[2], iy[2]] < cval <= z[ix[1], iy[1]]
        left = z[ix[0], iy[0]] < cval <= z[ix[3], iy[3]]
        if right and left:
            # Same order of summation as Python's sum() over the generator: 0 + z0 + z1 + z2 + z3
            hm = 0.0
            for i in range(4):
                hm += z[ix[i], iy[i]]
            hm = hm / 4.0
            if hm <= cval:
                left = False
            else:
                right = False
        if right:
            nix[0] = ix[2]
            nix[1] = ix[1]
            nix[2] = ix[1] + dx
            nix[3] = ix[2] + dx
            niy[0] = iy[2]
            niy[1] = iy[1]
            niy[2] = iy[1] + dy
            niy[3] = iy[2] + dy
        elif left:
            nix[0] = ix[0]
            nix[1] = ix[3]
            nix[2] = ix[3] - dx
            nix[3] = ix[0] - dx
            niy[0] = iy[0]
            niy[1] = iy[3]
            niy[2] = iy[3] - dy
            niy[3] = iy[0] - dy
        else:
            # Upwards
            nix[0] = ix[3]
            nix[1] = ix[2]
            nix[2] = ix[2] - dy
            nix[3] = ix[3] - dy
            niy[0] = iy[3]
            niy[1] = iy[2]
            niy[2] = iy[2] + dx
            niy[3] = iy[3] + dx

    @njit(cache=True, nogil=True)
    def follow_line(ix0, iy0, mx, my, x, y, z, cval, ilog, wx, wy, n):
        """
        Follow a contour from a starting cell through the grid (VOLGLIJN) and add its points to the buffer.

        Parameters
        ----------
        ix0, iy0 : np.ndarray
            The x and y indices of the four corners of the starting cell
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
        wx, wy : np.ndarray
            The buffer of the x and y coordinates of the points, filled in place
        n : int
            The number of points in the buffer

        Returns
        -------
        int
            The new number of points in the buffer, or -1 if the buffer is full
        """
        cap = wx.shape[0]
        ix = ix0.copy()
        iy = iy0.copy()
        nix = np.empty(4, np.int64)
        niy = np.empty(4, np.int64)

        zxb = interpol(z[ix[0], iy[0]], z[ix[1], iy[1]], x[ix[0]], x[ix[1]], cval)
        zyb = interpol(z[ix[0], iy[0]], z[ix[1], iy[1]], y[iy[0]], y[iy[1]], cval)
        if n >= cap:
            return -1
        wx[n] = zxb
        wy[n] = zyb
        n += 1

        zxnew = zxb
        zynew = zyb
        cell_number = 1
        proceed = True
        while proceed:
            zxold = zxnew
            zyold = zynew
            next_cell(ix, iy, nix, niy, z, cval, ilog)
            for i in range(4):
                ix[i] = nix[i]
                iy[i] = niy[i]
            zxnew = interpol(z[ix[0], iy[0]], z[ix[1], iy[1]], x[ix[0]], x[ix[1]], cval)
            zynew = interpol(z[ix[0], iy[0]], z[ix[1], iy[1]], y[iy[0]], y[iy[1]], cval)
            if not (zxnew == zxold and zynew == zyold):
                if n >= cap:
                    return -1
                wx[n] = zxnew
                wy[n] = zynew
                n += 1
            if cell_number > 1:
                proceed = not (zxb == zxnew and zyb == zynew)
            proceed = proceed and (0 <= ix[2] < mx) and (0 <= iy[2] < my)
            cell_number = 2

        if n >= cap:
            return -1
        wx[n] = SENTINEL
        wy[n] = SENTINEL
        n += 1
        return n

    @njit(cache=True, nogil=True)
    def set_cell(ix, iy, a, b, c, d, e, f, g, h):
        """
        Set the corner indices of a cell.

        Parameters
        ----------
        ix, iy : np.ndarray
            The x and y indices of the four corners, filled in place
        a, b, c, d : int
            The x indices of the four corners
        e, f, g, h : int
            The y indices of the four corners
        """
        ix[0] = a
        ix[1] = b
        ix[2] = c
        ix[3] = d
        iy[0] = e
        iy[1] = f
        iy[2] = g
        iy[3] = h

    @njit(cache=True, nogil=True)
    def isoline_kernel(x, y, z, cval, wx, wy):
        """
        Determine the contour of a value on a grid (ISOLIJN): first from the edges in the order bottom, right, top,
        left, then the closed contours.

        Parameters
        ----------
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid, with the shape (x, y)
        cval : float
            The value of the contour
        wx, wy : np.ndarray
            The buffer of the x and y coordinates of the points, filled in place

        Returns
        -------
        int
            The number of points in the buffer, or -1 if the buffer is full
        """
        mx = x.shape[0]
        my = y.shape[0]
        ilog = np.zeros((mx - 1, my), np.bool_)
        for iyi in range(my):
            for ixi in range(mx - 1):
                ilog[ixi, iyi] = z[ixi, iyi] < cval <= z[ixi + 1, iyi]

        ix = np.empty(4, np.int64)
        iy = np.empty(4, np.int64)
        n = 0

        # Bottom edge
        j = 0
        for i in range(mx - 1):
            if z[i, j] < cval <= z[i + 1, j]:
                ilog[i, j] = False
                set_cell(ix, iy, i, i + 1, i + 1, i, j, j, j + 1, j + 1)
                n = follow_line(ix, iy, mx, my, x, y, z, cval, ilog, wx, wy, n)
                if n < 0:
                    return -1

        # Right edge
        i = mx - 1
        for j in range(my - 1):
            if z[i, j] < cval <= z[i, j + 1]:
                set_cell(ix, iy, i, i, i - 1, i - 1, j, j + 1, j + 1, j)
                n = follow_line(ix, iy, mx, my, x, y, z, cval, ilog, wx, wy, n)
                if n < 0:
                    return -1

        # Top edge
        j = my - 1
        for i in range(mx - 1, 0, -1):
            if z[i, j] < cval <= z[i - 1, j]:
                set_cell(ix, iy, i, i - 1, i - 1, i, j, j, j - 1, j - 1)
                n = follow_line(ix, iy, mx, my, x, y, z, cval, ilog, wx, wy, n)
                if n < 0:
                    return -1

        # Left edge
        i = 0
        for j in range(my - 1, 0, -1):
            if z[i, j] < cval <= z[i, j - 1]:
                set_cell(ix, iy, i, i, i + 1, i + 1, j, j - 1, j - 1, j)
                n = follow_line(ix, iy, mx, my, x, y, z, cval, ilog, wx, wy, n)
                if n < 0:
                    return -1

        # Closed contours
        for j in range(1, my - 1):
            for i in range(mx - 1):
                if ilog[i, j]:
                    ilog[i, j] = False
                    set_cell(ix, iy, i, i + 1, i + 1, i, j, j, j + 1, j + 1)
                    n = follow_line(ix, iy, mx, my, x, y, z, cval, ilog, wx, wy, n)
                    if n < 0:
                        return -1
        return n

    def isoline(x: np.ndarray, y: np.ndarray, z: np.ndarray, cval: float):
        """
        Determine the contour of a value on a grid (ISOLIJN), with the same signature and output as
        'Contour.isoline_python', but compiled.

        Parameters
        ----------
        x, y : np.ndarray
            The coordinates of the grid
        z : np.ndarray
            The values on the grid, with the shape (x, y)
        cval : float
            The value of the contour

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            The x and y coordinates of the contour, with SENTINEL between the parts; None if the buffer turned out to
            be too small, so the caller uses the Python route
        """
        mx, my = len(x), len(y)
        cap = SPACE_PER_POINT * mx * my + 16
        wx = np.empty(cap)
        wy = np.empty(cap)
        n = isoline_kernel(
            np.ascontiguousarray(x, float),
            np.ascontiguousarray(y, float),
            np.ascontiguousarray(z, float),
            float(cval),
            wx,
            wy,
        )
        if n < 0:
            return None
        if n == 0:
            return np.empty(0), np.empty(0)

        # Remove the last sentinel and consecutive duplicate points
        wx = wx[: n - 1]
        wy = wy[: n - 1]
        if wx.shape[0] == 0:
            return np.empty(0), np.empty(0)
        keep = np.ones(wx.shape[0], np.bool_)
        keep[1:] = ~((wx[1:] == wx[:-1]) & (wy[1:] == wy[:-1]))
        return np.ascontiguousarray(wx[keep]), np.ascontiguousarray(wy[keep])
