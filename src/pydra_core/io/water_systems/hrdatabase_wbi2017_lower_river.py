import pandas as pd

from .hrdatabase_wbi2017_generic import HRDReaderWBI2017
from ..file_hydranl import FileHydraNL


class HRDReaderWBI2017LowerRiver(HRDReaderWBI2017):
    """
    HR database reader for WBI2017 lower river databases (Rhine tidal, Meuse tidal).
    Extends HRDReaderWBI2017 with Europoort barrier closing levels.
    """

    def get_closing_levels_table_europoort(self, closing_function: str | None) -> pd.DataFrame:
        """
        Read the closing levels for the Europoort barrier: the table in the database if there is one, otherwise the
        closing function of the scenario.

        Parameters
        ----------
        closing_function : str | None
            The closing function of the scenario, relative to 'data/statistics' ('europoort_barrier_closing_function');
            None if the scenario has none for this water system

        Returns
        -------
        pd.DataFrame
            A Dataframe with the closing level at sea (m) given r, u, q
        """
        # If there is a table called 'Sluitfunctie Europoortkering', use it
        try:
            # Read the table
            table = pd.read_sql("SELECT * FROM [Sluitfunctie Europoortkering]", con=self.con)

        # Otherwise use the closing function of the scenario
        except Exception as e:
            print(f"{e}: Using default functions")
            if closing_function is None:
                raise ValueError(f"[ERROR] No closing levels for water system '{self.get_water_system().name}'.") from e
            table = pd.read_csv(FileHydraNL.PACKAGE_PATH / closing_function, delimiter=";")

        # All columns to lower
        table.columns = table.columns.str.lower()

        # Rename
        table.rename(
            columns={
                "windrichting": "r",
                "afvoer": "q",
                "windsnelheid": "u",
                "zeewaterstand": "m",
            },
            inplace=True,
        )

        # Return table
        return table
