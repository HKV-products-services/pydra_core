import pandas as pd

from pathlib import Path

from .hrdatabase_wbi2017_generic import HRDReaderWBI2017
from ...common.enum import WaterSystem


class HRDReaderWBI2017LowerRiver(HRDReaderWBI2017):
    """
    HR database reader for WBI2017 lower river databases (Rhine tidal, Meuse tidal).
    Extends HRDReaderWBI2017 with Europoort barrier closing levels.
    """

    def get_closing_levels_table_europoort(self) -> pd.DataFrame:
        """
        Read the closing levels for the Europoort barrier.

        Returns
        -------
        pd.DataFrame
            A Dataframe with the closing level at sea (m) given r, u, q
        """
        # If there is a table called 'Sluitfunctie Europoortkering', use it
        try:
            # Read the table
            table = pd.read_sql("SELECT * FROM [Sluitfunctie Europoortkering]", con=self.con)

        # Otherwise use the default functions
        except Exception as e:
            print(f"{e}: Using default functions")
            PATH = Path(__file__).resolve().parent.parent / ".." / "data" / "statistics" / "Sluitpeilen"
            if self.get_water_system() in [
                WaterSystem.RHINE_TIDAL,
                WaterSystem.EUROPOORT,
            ]:
                table = pd.read_csv(PATH / "Sluitfunctie Europoortkering Rijn 2017.csv", delimiter=";")
            elif self.get_water_system() == WaterSystem.MEUSE_TIDAL:
                table = pd.read_csv(PATH / "Sluitfunctie Europoortkering Maas 2017.csv", delimiter=";")
            else:
                raise (f"[ERROR] No closing levels for water system '{self.get_water_system().name}'.")

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
