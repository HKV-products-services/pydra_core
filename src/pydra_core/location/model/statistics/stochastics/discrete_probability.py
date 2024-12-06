import numpy as np

from .....io.file_hydranl import FileHydraNL


class DiscreteProbability:
    """
    Class to describe the discrete statistics.
    """

    def __init__(self, statistics_file_path: str):
        """
        Constructor class for the discrete statistics.

        Parameters
        ----------
        statistics_file_path : str
            Path to the statistics file
        """
        self.discretisation, self.probability = FileHydraNL.read_file_2columns(
            statistics_file_path
        )

    def __len__(self):
        """
        Return the number discretisations.

        Returns
        -------
        int
            Number of discretisations
        """
        return len(self.discretisation)

    def get_discretisation(self) -> np.ndarray:
        """
        Return the discretisation.

        Returns
        -------
        np.ndarray
            1D array with discretisation
        """
        return self.discretisation

    def get_probability(self) -> np.ndarray:
        """
        Return the probability of each discretisation.

        Returns
        -------
        np.ndarray
            1D array with probabilities
        """
        return self.probability
