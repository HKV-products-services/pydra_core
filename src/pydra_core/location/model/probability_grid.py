import numpy as np


def freeze(array: np.ndarray) -> None:
    """
    Make an array read-only, and the array it is a view of as well, so its values cannot change through either.

    Parameters
    ----------
    array : np.ndarray
        The array
    """
    array.flags.writeable = False
    if isinstance(array.base, np.ndarray):
        array.base.flags.writeable = False


class DirectionProbabilityGrid:
    """
    A probability grid with the wind direction as the last axis, kept as one dense slice per wind direction. Filling
    a dense grid per wind direction and copying it with the wind direction as the first axis afterwards both stride
    through the whole grid; this keeps the slices as they are calculated.

    It supports what its users need: 'shape', a contiguous slice '[a:b]' on the first axis (for 'p_hur_tr[1:-1]' in
    the HBN calculation) and 'get_slice(ir)'. Anything else raises an error instead of silently doing something
    different from a dense array.
    """

    def __init__(self, shape):
        """
        Create an empty grid.

        Parameters
        ----------
        shape : tuple
            The shape of the dense grid, with the wind direction as the last axis
        """
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.float64
        self.slice_shape = self.shape[:-1]
        self._slices = {}

    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions of the dense grid.

        Returns
        -------
        int
            The number of dimensions
        """
        return len(self.shape)

    def add_slice(self, ir: int, dense_slice: np.ndarray) -> None:
        """
        Add a slice to a wind direction. The first one is kept itself: in a dense grid of zeros it would be 0.0 plus
        the probability, which is the same probability (never -0.0, probabilities are not negative).

        Parameters
        ----------
        ir : int
            The index of the wind direction
        dense_slice : np.ndarray
            The probabilities of the wind direction, with the shape 'slice_shape'
        """
        if dense_slice.shape != self.slice_shape or dense_slice.dtype != np.float64:
            raise ValueError(f"[ERROR] Slice {dense_slice.shape} {dense_slice.dtype} does not fit {self.slice_shape} float64.")
        if ir in self._slices:
            self._slices[ir] += dense_slice
        else:
            self._slices[ir] = dense_slice

    def get_slice(self, ir: int) -> np.ndarray:
        """
        Return the slice of a wind direction, read-only and always the same array.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The probabilities of the wind direction, with the shape 'slice_shape'
        """
        if ir not in self._slices:
            self._slices[ir] = np.zeros(self.slice_shape)
        dense_slice = self._slices[ir]
        freeze(dense_slice)
        return dense_slice

    def __getitem__(self, key):
        """
        Take a contiguous slice on the first axis, as a read-only view per wind direction. The slices of this grid
        become read-only as well, so the views cannot change.

        Parameters
        ----------
        key : slice
            A slice '[a:b]' without a step

        Returns
        -------
        DirectionProbabilityGrid
            The grid of the slice
        """
        if not isinstance(key, slice) or key.indices(self.shape[0])[2] != 1:
            raise TypeError(f"[ERROR] DirectionProbabilityGrid only supports '[a:b]' on the first axis, not {key!r}.")
        start, stop, _ = key.indices(self.shape[0])
        grid = DirectionProbabilityGrid((max(stop, start) - start,) + self.shape[1:])
        for ir, dense_slice in self._slices.items():
            freeze(dense_slice)
            grid._slices[ir] = dense_slice[key]
        return grid


class SparseProbabilityGrid:
    """
    A probability grid with the wind direction as the last axis, that keeps only the nonzero values per wind
    direction. For the large grids of the HBN calculation (up to 30 GiB dense) almost all values are zero.

    It supports the same as DirectionProbabilityGrid: 'shape', a contiguous slice '[a:b]' on the first axis and
    'get_slice(ir)'.
    """

    def __init__(self, shape):
        """
        Create an empty grid.

        Parameters
        ----------
        shape : tuple
            The shape of the dense grid, with the wind direction as the last axis
        """
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.float64
        self.slice_shape = self.shape[:-1]
        self._indices = {}
        self._values = {}

    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions of the dense grid.

        Returns
        -------
        int
            The number of dimensions
        """
        return len(self.shape)

    def set_slice(self, ir: int, dense_slice: np.ndarray) -> None:
        """
        Store the nonzero values of a dense slice for a wind direction.

        Parameters
        ----------
        ir : int
            The index of the wind direction
        dense_slice : np.ndarray
            The probabilities of the wind direction, with the shape 'slice_shape'
        """
        nz = np.flatnonzero(dense_slice)
        self._indices[ir] = nz
        self._values[ir] = dense_slice.reshape(-1)[nz].copy()

    def fill(self, ir: int, buffer: np.ndarray) -> None:
        """
        Put the stored values of a wind direction back into a buffer of zeros.

        Parameters
        ----------
        ir : int
            The index of the wind direction
        buffer : np.ndarray
            A buffer of zeros with the shape 'slice_shape', filled in place
        """
        if ir in self._indices:
            buffer.reshape(-1)[self._indices[ir]] = self._values[ir]

    def get_slice(self, ir: int) -> np.ndarray:
        """
        Return the dense slice of a wind direction, as np.take(..., ir, axis=-1) on the dense grid.

        Parameters
        ----------
        ir : int
            The index of the wind direction

        Returns
        -------
        np.ndarray
            The probabilities of the wind direction, with the shape 'slice_shape'
        """
        dense_slice = np.zeros(self.slice_shape)
        self.fill(ir, dense_slice)
        return dense_slice

    def __getitem__(self, key):
        """
        Take a contiguous slice on the first axis.

        Parameters
        ----------
        key : slice
            A slice '[a:b]' without a step

        Returns
        -------
        SparseProbabilityGrid
            The grid of the slice
        """
        if not isinstance(key, slice) or key.indices(self.shape[0])[2] != 1:
            raise TypeError(f"[ERROR] SparseProbabilityGrid only supports '[a:b]' on the first axis, not {key!r}.")
        start, stop, _ = key.indices(self.shape[0])
        stop = max(stop, start)
        grid = SparseProbabilityGrid((stop - start,) + self.shape[1:])
        per_level = int(np.prod(self.slice_shape[1:])) if len(self.slice_shape) > 1 else 1
        for ir, flat in self._indices.items():
            level = flat // per_level
            keep = (level >= start) & (level < stop)
            grid._indices[ir] = flat[keep] - start * per_level
            grid._values[ir] = self._values[ir][keep]
        return grid
