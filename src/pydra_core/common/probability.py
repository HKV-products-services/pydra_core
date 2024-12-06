import numpy as np

from collections import namedtuple

pdstruct = namedtuple("pdstruct", ["delta", "probability", "density", "edges"])


class ProbabilityFunctions:
    """
    A class with common functions used within statistics.
    """

    @staticmethod
    def probability_density(
        values: np.ndarray,
        exceedance_probability: np.ndarray,
        bounded: bool = True,
        check: bool = True,
        axis: int = None,
    ) -> pdstruct:
        """
        Function to convert the exceedance probability into a probability
        density function.

        Parameters
        ----------
        values : np.ndarray
            Values, for example the wind speed discretisation
        exceedance_probability : np.ndarray
            Exceedance probability of the values
        bounded : bool, optional
            If bounded, add the first and last element based on the min and max
            in the exceedance probability. Otherwise between 1.0 and 0.0
            (default : True)
        check : bool, optional
            Check whether the values and exceedance probabilities are
            monotonously increasing (default : True)
        axis : int, optional
            Axis (default : None)

        Raises
        ------
        ValueError
            If the values or exceedance probabilities are not monotonously
            increasing

        Returns
        -------
        pdstruct
            Probability density structure
        """
        # For multiple dimensions, use the _nd function
        if exceedance_probability.ndim > 1:
            return ProbabilityFunctions.probability_density_nd(
                values, exceedance_probability, bounded, axis=axis
            )

        # Check whether the values and exceedance probabilities are monotonously increasing
        if check:
            for arr, tag in zip(
                [values, exceedance_probability], ["Values", "Exceedance probabilities"]
            ):
                diff = arr[1:] - arr[:-1]
                if not (all(diff >= 0) or all(diff <= 0)):
                    raise ValueError(
                        tag + "are not monotonously increasing or decreasing.",
                        arr,
                        diff,
                    )

        # Determine the exceedance probability bins
        bins_edges = (exceedance_probability[1:] + exceedance_probability[:-1]) / 2.0

        # If bounded, add the first and last element based on the min and max in the exceedance probability
        if bounded:
            bins_edges = np.concatenate(
                [[exceedance_probability[0]], bins_edges, [exceedance_probability[-1]]]
            )

        # Else, determine the bins between the 0 and 1
        else:
            if exceedance_probability[0] < exceedance_probability[-1]:
                bins_edges = np.concatenate([[0.0], bins_edges, [1.0]])
            else:
                bins_edges = np.concatenate([[1.0], bins_edges, [0.0]])

        # The difference between the bin_edges are the bins_probabilities
        bins_probability = np.absolute(bins_edges[1:] - bins_edges[:-1])

        # Determine the delta of the values
        bins_values = np.concatenate(
            [[values[0]], (values[1:] + values[:-1]) / 2.0, [values[-1]]]
        )
        bins_deltas = np.absolute(np.diff(bins_values))

        # Probability density is the bins_probability divided by the delta
        probability_density = np.absolute(bins_probability / bins_deltas)

        # Return as a structure
        return pdstruct(bins_deltas, bins_probability, probability_density, bins_edges)

    @staticmethod
    def probability_density_nd(
        values: np.ndarray,
        exceedance_probability: np.ndarray,
        bounded: bool = True,
        axis: int = None,
    ) -> pdstruct:
        """
        Convert the exceedance probability into a probability density function
        for multidimensional arrays.

        This function converts the exceedance probabilities
        (exceedance_probability) into a probability density function (PDF) for
        multidimensional arrays represented by 'values' and
        'exceedance_probability'.

        Parameters
        ----------
        values : np.ndarray
            Values, e.g., wind speed discretization, for which the PDF is
            calculated.
        exceedance_probability : np.ndarray
            Exceedance probability of the values.
        bounded (bool):
            If True, add the first and last element based on the min and max in
            the exceedance probability. Otherwise, create the PDF between 1.0
            and 0.0.
        axis : int, optional
            The axis along which the PDF is calculated. By default, axis 0 is
            used.

        Returns
        -------
        pdstruct
            Probability density structure containing the calculated PDF.
        """
        # If bounded, add the first and last element based on the min and max
        # in the exceedance probability
        if bounded:
            bins_edges = np.concatenate(
                [
                    [exceedance_probability[0]],
                    (exceedance_probability[1:] + exceedance_probability[:-1]) / 2.0,
                    [exceedance_probability[-1]],
                ]
            )

        # If not
        else:
            bins_edges = np.pad(
                (exceedance_probability[1:] + exceedance_probability[:-1]) / 2,
                pad_width=(1, 1),
                mode="constant",
                constant_values=(0, 1)
                if (exceedance_probability[0] < exceedance_probability[-1]).all()
                else (1, 0),
            )

        # The difference between the bin_edges are the bins_probabilities
        bins_probability = np.absolute(bins_edges[1:, ...] - bins_edges[:-1, ...])

        # Edges between consecutive values, the difference gives the bin_deltas
        bins_deltas = np.absolute(
            np.diff(
                np.concatenate(
                    [[values[0]], (values[1:] + values[:-1]) / 2.0, [values[-1]]]
                )
            )
        )

        # Probability density is the bins_probability divided by the delta
        shp = [1] * bins_probability.ndim
        if axis is None:
            axis = 0
        shp[axis] = -1
        probability_density = np.absolute(
            bins_probability / bins_deltas.reshape(tuple(shp))
        )

        # Return as a structure
        return pdstruct(bins_deltas, bins_probability, probability_density, bins_edges)

    @staticmethod
    def get_hnl_disc_array(vmin: float, vmax: float, step: float) -> np.ndarray:
        """
        Get a discretized array of values between vmin and vmax with the given
        step size.

        This function generates a discretized array of values in the specified
        range [vmin, vmax] with a given step size. The array includes the vmin
        and vmax values and is uniformly spaced with steps of the specified
        size.

        Parameters
        ----------
        vmin : float
            Minimum value of the range.
        vmax : float
            Maximum value of the range.
        step : float
            Step size between values.

        Returns
        -------
        np.array
            Discretized array of values.
        """
        # Calculate the number of steps between vmin and vmax and create the
        # discretized array
        n = round((vmax - vmin) / step)
        levels = np.arange(vmin, vmin + (n + 0.1) * step, step)

        # Ensure that the last element of the array is exactly equal to vmax
        levels[-1] = vmax
        return levels

    @staticmethod
    def conditional_probability(probability: np.ndarray, axis: int) -> np.ndarray:
        """
        Calculate the conditional probability along an axis, taking into account
        dividing by zero.

        This function calculates the conditional probability along the
        specified 'axis' of the input 'probability' array. The conditional
        probability is the probability of an event occurring given that another
        event has occurred. If the denominator is zero along the specified
        'axis', the result is set to zero to avoid division by zero errors.

        Parameters
        ----------
        probability : np.ndarray
            Array with probabilities
        axis : int
            Axis for which the conditional probability has to be calculated

        Returns
        -------
        np.ndarray
            Conditional probability
        """
        # Calculate the denominator as the sum of probabilities along the
        # specified 'axis'
        denominator = np.sum(probability, axis=axis)

        # Create a shape with all ones, except for the axis that will be used
        # for broadcasting
        shape = list(denominator.shape)
        shape.insert(axis, -1)

        # Calculate the conditional probability using element-wise division,
        # handling division by zero
        cond = np.divide(
            probability,
            denominator.reshape(shape),
            out=np.zeros_like(probability),
            where=(denominator != 0).reshape(shape),
        )

        return cond

    @staticmethod
    def calculate_boundaries(levels: np.ndarray):
        """
        Calculate the boundaries for the input array.

        This function calculates the boundaries between adjacent elements in
        the input 'arr' array. The boundaries are computed as the midpoints
        between consecutive elements, with additional values added at the start
        and end based on the step differences.

        Parameters
        ----------
        levels : np.ndarray
            Input array for which boundaries are to be calculated.

        Returns
        -------
        ndarray
            Array containing the calculated boundaries.
        """
        # Calculate the step differences between consecutive elements
        lower_step = levels[2] - levels[1]
        upper_step = levels[-1] - levels[-2]

        # Calculate midpoints between consecutive elements
        mid = (levels[1:] + levels[:-1]) / 2

        # Calculate the boundaries array by concatenating the midpoints with
        # additional boundary values
        bounds = np.concatenate([[mid[0] - lower_step], mid, [mid[-1] + upper_step]])

        return bounds
