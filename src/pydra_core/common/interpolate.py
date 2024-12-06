import numpy as np


class Interpolate:
    """
    A class with common functions used for interpolation.
    """

    @staticmethod
    def inextrp1d(x, xp, fp):
        """
        Interpolate an array along the given axis.
        Similar to np.interp, but with extrapolation outside range.

        Parameters
        ----------
        x : np.array
            Array with positions to interpolate at
        xp : np.array
            Array with positions of known values
        fp : np.array
            Array with values as known positions to interpolate between

        Returns
        -------
        np.array
            interpolated array
        """
        # Determine lower bounds
        intidx = np.minimum(np.maximum(0, np.searchsorted(xp, x) - 1), len(xp) - 2)

        # Determine interpolation fractions
        fracs = (x - xp[intidx]) / (xp[intidx + 1] - xp[intidx])

        # Interpolate (1-frac) * f_low + frac * f_up
        f = (1 - fracs) * fp[intidx] + fp[intidx + 1] * fracs

        return f

    @staticmethod
    def inextrp1d_log_probability(
        x: np.ndarray, xp: np.ndarray, fp: np.ndarray
    ) -> np.ndarray:
        """
        Perform 1D log-interpolation of log-probability values using piecewise
        linear interpolation.

        This function interpolates log-probability values for input points 'x'
        using a given set of log-probabilities 'fp' and corresponding
        probabilities 'xp'. The interpolation is performed in a piecewise
        manner, taking into account the logarithmic nature of the data.

        Parameters
        ----------
        x : np.ndarray
            Input points at which log-probabilities are to be interpolated.
        xp : np.ndarray
            Known data points representing the probabilities.
        fp : np.ndarray
            Log-probabilities corresponding to the known data points 'xp'.

        Returns
        -------
        np.ndarray
            Interpolated log-probability values for the input points 'x'.

        Raises:
            ValueError: If the provided xp values (probabilities) are not
            monotonically increasing.

        Notes
        -----
        -   The input array 'xp' should be strictly increasing; otherwise, a
            ValueError is raised.
        """
        # Check whether the probabilities are monotonically decreasing
        if not (np.diff(xp) >= 0.0).all():
            raise ValueError(
                "[ERROR] xp-values are not monotonically increasing or decreasing"
            )

        # Determine lower bounds
        intidx = np.minimum(np.maximum(0, np.searchsorted(xp, x) - 1), len(xp) - 2)

        # Split in log and linear part
        iszero = xp[intidx] == 0.0
        intidx_log = intidx[~iszero]
        intidx_lin = intidx[iszero]

        # Determine interpolation fractions
        fracs = np.zeros(len(x), dtype=float)
        fracs[~iszero] = (np.log(x[~iszero]) - np.log(xp[intidx_log])) / (
            np.log(xp[intidx_log + 1]) - np.log(xp[intidx_log])
        )
        fracs[iszero] = (x[iszero] - xp[intidx_lin]) / (
            xp[intidx_lin + 1] - xp[intidx_lin]
        )

        # Interpolate (1-frac) * f_low + frac * f_up
        f = (1 - fracs) * fp[intidx] + fp[intidx + 1] * fracs

        return f

    @staticmethod
    def triangular_interpolation(
        x1: float,
        y1: float,
        h1: float,
        x2: float,
        y2: float,
        h2: float,
        x3: float,
        y3: float,
        h3: float,
        x: float,
        y: float,
    ) -> float:
        """
        Performs triangular interpolation to calculate the parameter value
        (water level) at a given point within a triangle.

        Parameters
        ----------
        x1 : float
            X-coordinate of the first input location.
        y1 : float
            Y-coordinate of the first input location.
        h1 : float
            Parameter value (water level) at the first input location.
        x2 : float
            X-coordinate of the second input location.
        y2 : float
            Y-coordinate of the second input location.
        h2 : float
            Parameter value (water level) at the second input location.
        x3 : float
            X-coordinate of the third input location.
        y3 : float
            Y-coordinate of the third input location.
        h3 : float
            Parameter value (water level) at the third input location.
        x : float
            X-coordinate of the point to interpolate.
        y : float
            Y-coordinate of the point to interpolate.

        Returns
        -------
        float
            The interpolated parameter value (water level) at the given point.
        """
        # Check if the input locations are invalid
        if (
            (x1 == x2 and y1 == y2)
            or (x1 == x3 and y1 == y3)
            or (x2 == x3 and y2 == y3)
        ):
            raise ValueError(
                "[ERROR] Two input locations in triangle interpolation are exactly the same. This is not allowed."
            )
        if (x1 == x2 == x3) or (y1 == y2 == y3):
            raise ValueError(
                "[ERROR] The three input locations in triangle interpolation are collinear. This is not allowed."
            )

        # Check if the locations are on the same line
        a_1 = (y1 - y2) / (x1 - x2)
        b_1 = y1 - a_1 * x1
        a_2 = (y3 - y2) / (x3 - x2)
        b_2 = y3 - a_1 * x3
        if (a_1 == a_2) and (b_1 == b_2):
            raise ValueError(
                "[ERROR] The three input locations in triangle interpolation are collinear. This is not allowed."
            )

        # Calculate auxiliary parameter a
        a = x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2)

        # Parameter a should not be equal to 0
        if a == 0.0:
            raise ValueError(
                "[ERROR] In triangle interpolation, there is an auxiliary parameter (a) that should not be equal to 0, but it is currently set to 0."
            )

        # Calculate parameters B1, B2 en B3
        b1 = h1 * (y2 - y3) - h2 * (y1 - y3) + h3 * (y1 - y2)
        b2 = x1 * (h2 - h3) - x2 * (h1 - h3) + x3 * (h1 - h2)
        b3 = (
            x1 * (y2 * h3 - y3 * h2)
            - x2 * (y1 * h3 - y3 * h1)
            + x3 * (y1 * h2 - y2 * h1)
        )

        # Calculate the value at (x, y) using the triangular interpolation
        triangular_interp = (b1 / a) * x + (b2 / a) * y + b3 / a
        return triangular_interp


class InterpStruct:
    """
    Interpolation helper class.

    This class provides functionality for 1D interpolation of arrays along a
    specified axis. It is designed to interpolate arrays using piecewise linear
    interpolation based on provided data points.
    """

    def __init__(self, x: np.ndarray, xp: np.ndarray):
        """
        Initialize the InterpStruct object and calculate interpolation factors.

        This method is used internally to calculate interpolation factors based
        on given data points 'x' and 'xp'. It sets the 'xp', 'x', 'intidx', and
        'fracs' attributes of the InterpStruct object.

        Parameters
        ----------
        x : np.ndarray
            Input data points used for interpolation.
        xp : np.ndarray
            Known data points representing the interpolation axis.
        """
        self.xp = np.asarray(xp)
        self.x = np.asarray(x)
        self.intidx = np.minimum(
            np.maximum(0, np.searchsorted(self.xp, self.x) - 1), len(self.xp) - 2
        )
        self.fracs = (self.x - self.xp[self.intidx]) / (
            self.xp[self.intidx + 1] - self.xp[self.intidx]
        )

    def interp(
        self,
        fp: np.ndarray,
        axis: int = 0,
        extrapolate: bool = True,
        left: bool = None,
        right: bool = None,
    ):
        """
        Interpolate a (multidimensional) array along the given axis using
        piecewise linear interpolation.

        Parameters
        ----------
        fp : np.ndarray
            Array with values to interpolate.
        axis : int, optional
            Axis along which the interpolation is performed (default is 0).
        extrapolate : bool, optional
            If True, allow extrapolation of values outside the range of 'xp'.
            If False, set the values outside the interpolation range to NaN
            (default is True).
        left : float or None, optional
            Value to use for extrapolation on the left side. If None, the left
            extrapolation will result in NaN (default is None).
        right : float or None, optional:
            Value to use for extrapolation on the right side. If None, the
            right extrapolation will result in NaN (default is None).

        Returns
        -------
        ndarray
            Interpolated multidimensional array.

        Raises
        ------
            ValueError: If the given axis is higher than the dimensions of the
            'fp' array.
        """
        # Convert to array if needed
        if isinstance(fp, list):
            fp = np.asarray(fp)

        # Check given axis and fp shape
        if axis > fp.ndim - 1:
            raise ValueError(
                f"Given axis ({axis}) is higher than dimensions of fp array ({fp.ndim})."
            )

        # Create shape with all ones, except for the array which is used for
        # interpolation
        shape = [1] * fp.ndim
        shape[axis] = len(self.fracs)

        # Interpolate (1-frac) * f_low + frac * f_up
        f = (1 - self.fracs.reshape(shape)) * np.take(
            fp, self.intidx, axis=axis
        ) + np.take(fp, self.intidx + 1, axis=axis) * self.fracs.reshape(shape)

        # If 'extrapolate' is False, set values outside the interpolation range
        # to NaN or the specified values
        if not extrapolate:
            if left is not None:
                f[(self.x < self.xp[0])] = left
            else:
                f[(self.x < self.xp[0])] = np.nan
            if right is not None:
                f[(self.x > self.xp[-1])] = right
            else:
                f[(self.x > self.xp[-1])] = np.nan

        return f

    def interp_angle(
        self, fp: np.ndarray, axis: int = 0, extrapolate: bool = True
    ) -> np.ndarray:
        """
        Interpolate angles (in degrees) using piecewise linear interpolation.

        This function performs interpolation of angles represented by 'fp'
        using piecewise linear interpolation. The angles are given in degrees
        and can be provided as a 1D or 2D array.

        Parameters
        ----------
        fp : np.ndarray
            Input angles (in degrees) to be interpolated.
        axis : int, optional
            The axis along which the interpolation is performed (default is 0).
        extrapolate : bool, optional
            If True, allow extrapolation of values outside the range of
            'self.xp'. If False, set the angles outside the interpolation
            range to NaN (default is True).

        Returns
        -------
        np.ndarray
            Interpolated angles in degrees.
        """
        # Convert angles from degrees to cosine and sine values and perform
        # interpolation along the specified 'axis'
        xint = self.interp(np.cos(np.radians(fp)), axis=axis)
        yint = self.interp(np.sin(np.radians(fp)), axis=axis)

        # Compute the interpolated angle in degrees using arctan2 and wrap it
        # to the range [0, 360)
        f = np.degrees(np.arctan2(yint, xint)) % 360

        # If 'extrapolate' is False, set angles outside the interpolation range
        # to NaN
        if not extrapolate:
            f[(self.x < self.xp[0]) | (self.x > self.xp[-1])] = np.nan

        return f
