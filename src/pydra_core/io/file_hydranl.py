import numpy as np

from pathlib import Path


class FileHydraNL:
    """
    Class with staticsmethods to read Hydra-NL input files
    """

    # Path to the package
    PACKAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "statistics"

    @staticmethod
    def read_file_2columns(path):
        """
        Read a Hydra-NL statistics file with two columns

        Parameters
        ----------
        path : str
            Path to the statistics file
        """
        # Open file
        with open(FileHydraNL.PACKAGE_PATH / Path(path), "r", encoding="cp1252") as f:
            lines = f.readlines()

        # Read non commented values from file, and split per line
        vals = [line.strip().split() for line in lines if not line.startswith("*")]

        # Convert to float and split in columns
        kol1, kol2 = np.vstack(vals).astype(float).T

        return kol1, kol2

    @staticmethod
    def read_file_ncolumns(path):
        """
        Read a Hydra-NL statistics file with more than two columns

        Parameters
        ----------
        path : str
            Path to the statistics file
        """
        # Open file
        with open(FileHydraNL.PACKAGE_PATH / Path(path), "r", encoding="cp1252") as f:
            lines = f.readlines()

        # Read non commented values from file, and split per line
        vals = [line.strip().split() for line in lines if not line.startswith("*")]

        # Convert to float and split in columns
        floatvals = np.vstack(vals).astype(float)

        return floatvals[:, 0], floatvals[:, 1:]

    @staticmethod
    def read_file_ncolumns_loc(path):
        """
        Read a Hydra-NL statistics location file with more than two columns

        Parameters
        ----------
        path : str
            Path to the statistics file
        """
        # Open file
        with open(FileHydraNL.PACKAGE_PATH / Path(path), "r", encoding="cp1252") as f:
            lines = f.readlines()

        # Process data
        vals_tot = [line.strip().split() for line in lines]

        # Convert to float and split in columns
        x = int(vals_tot[2][1][:-1])
        y = int(vals_tot[2][2])

        return x, y
