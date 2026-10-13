import sqlite3

from pathlib import Path
from typing import List, Union

from ..common.enum import WaterSystem
from ..location.settings.settings import Settings


class HRDReader:
    """
    Base class for HR database sqlite readers.
    """

    def __init__(self, path_to_database: str) -> None:
        # Check if the path is valid
        if not Path(path_to_database).exists():
            raise OSError(path_to_database)

        # Save the path
        self.path_to_database = path_to_database
        self.con = None

    def __enter__(self) -> "HRDReader":
        # Init the connection
        self.con = sqlite3.connect(self.path_to_database)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Close the connection
        self.con.close()

    def get_water_system(self) -> WaterSystem:
        """
        Obtain the water system from the .sqlite database

        Returns
        -------
        WaterSystem
            Corresponding water system
        """
        # Obtain the water system ID from the sqlite
        wsid = self.con.execute("SELECT GeneralId FROM General").fetchone()[0]

        # Return the WaterSystem
        return WaterSystem(wsid)

    def get_hrdlocations_names(self) -> List[str]:
        """
        Obtain a list with all names of the hrdlocations

        Returns
        -------
        list[str]
            A list with all names of hrdlocations
        """
        # Obtain the names from the sqlite
        hrdlocations = self.con.execute("SELECT Name FROM HRDLocations").fetchall()

        # Convert the result to a list of strings
        names = [row[0] for row in hrdlocations]

        # Return names
        return names

    def get_hrdlocation_id(self, hrdlocation: Union[str, Settings]) -> int:
        """
        Returns the HRDLocationID

        Parameters
        ----------
        hrdlocation : Union[str, Settings]
            HRDLocation

        Returns
        -------
        int
            HRDLocationId
        """
        # Obtain the HRDLocationName from a Settings object
        if isinstance(hrdlocation, Settings):
            hrdlocation = hrdlocation.location

        # Obtain the HRDLocationId from the sqlite
        hrdlocationid = self.con.execute(f"SELECT HRDLocationId FROM HRDLocations WHERE Name = '{hrdlocation}'").fetchone()[0]

        # Return HRDLocationId
        return hrdlocationid

    def get_hrdlocation_xy(self, hrdlocation: Union[str, Settings]) -> Union[float, float]:
        """
        Returns the X and Y coordinate of the HRDLocation

        Parameters
        ----------
        hrdlocation : Union[str, Settings]
            HRDLocation

        Returns
        -------
        Union[float, float]
            X and Y coordinate
        """
        # Obtain the HRDLocationName from a Settings object
        if isinstance(hrdlocation, Settings):
            hrdlocation = hrdlocation.location

        # Obtain the coordinates from the sqlite
        hrdlocationxy = self.con.execute(f"SELECT XCoordinate, YCoordinate FROM HRDLocations WHERE Name = '{hrdlocation}'").fetchone()

        # Return coordinates
        return hrdlocationxy

    @staticmethod
    def from_settings(settings: "Settings") -> "HRDReader":
        """
        Returns the appropriate HRDReader subclass for the given settings.

        Parameters
        ----------
        settings : Settings
            A Settings object used to determine the water system and database path.

        Returns
        -------
        HRDReader
            An HRDReader instance for the specified water system.
        """
        # Lazy import to avoid circular dependency with the factory
        from .hrdatabase_factory import HRDReaderFactory

        return HRDReaderFactory.get_hrdreader(settings)

    def get_wind_directions(self) -> dict:
        """
        Obtain a dictionary with HRDWindDirectionIds and Directions

        Returns
        -------
        dict
            A dictionary {HRDWindDirectionId : Direction}
        """
        # Wind directions
        results = self.con.execute("SELECT * FROM HRDWindDirections").fetchall()

        # Process wind directions such that {wind_id : wind_direction}
        wind_direction = {wid: wr for wid, wr in results}

        # Return wind direction dictionary
        return wind_direction
