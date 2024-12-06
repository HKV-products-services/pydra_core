from ....settings.settings import Settings
from .....io.file_hydranl import FileHydraNL


class SigmaFunction:
    """
    Class for the sigma function, which describes the correlation between the
    sea level and wind speed.
    """

    def __init__(self, settings: Settings):
        """
        Constructor class for the SigmaFunction.

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Read the statistics file
        self.sigma_sea_level, self.sigma = FileHydraNL.read_file_ncolumns(
            settings.sigma_function
        )
        self.correlation = self.sigma.min(axis=0) > 0.0
