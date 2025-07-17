import os
import warnings

from typing import List, Union

from ..common.enum import WaterSystem
from ..io.database_hr import DatabaseHR
from ..location.location import Location
from ..location.settings.settings import Settings


class HRDatabase:
    """
    HRDatabase class

    Attributes
    ----------
    watersystem, WaterSystem
        Water system of the dike trajectory
    locations, list
        List with all locations in the database
    """

    def __init__(self, database_path: str) -> None:
        # Connect to the database
        self.database_path = database_path

        # Check if path exists
        if not os.path.exists(self.database_path):
            raise FileNotFoundError()

        # Obtain water system and locations
        with DatabaseHR(database_path) as database:
            self.watersystem = database.get_water_system()
            self.locationnames = database.get_hrdlocations_names()

        # Empty locations
        self.locations = {}

    def __len__(self) -> int:
        """
        Return the number of locations

        Returns
        -------
        int
            Number of parts
        """
        return len(self.locationnames)

    def __iter__(self):
        """
        Dunder to allow iterating through the locations
        """
        self.__iterindex = -1
        return self

    def __next__(self):
        """
        Dunder to allow iterating through the locations
        """
        self.__iterindex += 1
        if self.__iterindex < len(self.locationnames):
            return self.locationnames[self.__iterindex]
        else:
            raise StopIteration

    def get_settings(self, hrdlocation: str) -> Settings:
        """
        Returns the Settings class of a hrdlocation
        Useful for when you manually want to adjust the settings
        To create a location using the settings use the function 'create_location(settings)'

        Parameters
        ----------
        hrdlocation : str
            The hrdlocation (uitvoerpunt)

        Returns
        -------
        Settings
            The Settings object for the hrdlocation
        """
        # Check if the hrdlocation exists
        if self.check_location(hrdlocation):
            # Return the settings
            return Settings(hrdlocation, self.database_path)

        # Otherwise, raise an exception
        else:
            raise ValueError(
                f"[ERROR] HRDLocation '{hrdlocation}' not found in the database."
            )

    def create_location(self, settings_or_str: Union[Settings, str]) -> Location:
        """
        Creates a location based upon the Settings object

        Parameters
        ----------
        settings : Settings
            The Settings object of a location or a string of the hrdlocation (uitvoerpunt)

        Returns
        -------
        Location
            The Location object for the given settings
        """
        # If the object is Settings
        if isinstance(settings_or_str, Settings):
            # Create the location and save it in the cache
            self.locations[settings_or_str.location] = Location(settings_or_str)

            # Return location
            return self.locations[settings_or_str.location]
        
        # If the object is a string
        elif isinstance(settings_or_str, str):
            # Check if the hrdlocation exists
            if self.check_location(settings_or_str):
                # If the location is not cached, create the location
                if settings_or_str not in self.locations:
                    # Default settings and create location
                    settings = self.get_settings(settings_or_str)
                    return self.create_location(settings)

                # Return the cached location
                else:
                    return self.locations[settings_or_str]

            # Otherwise, raise an exception
            else:
                raise ValueError(f"[ERROR] HRDLocation '{settings_or_str}' not found in the database.")

        else:
            raise ValueError("Use Settings object of a string of the hrdlocation.")

    def get_location(self, hrdlocation: str) -> Location:
        """
        Returns the Location class of a hrdlocation.
        Uses the default set of parameters.

        Parameters
        ----------
        hrdlocation : str
            The hrdlocation (uitvoerpunt)

        Returns
        -------
        Location
            The Location object for the hrdlocation
        """
        warnings.warn(
            "get_location() is deprecated and will be removed in a future version. "
            "Use create_location() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.create_location(hrdlocation)

    def check_location(self, hrdlocation: str) -> bool:
        """
        Check if a hrdlocation exists within the database

        Parameters
        ----------
        hrdlocation : str
            The hrdlocation (uitvoerpunt)

        Returns
        -------
        bool
            Whether the hrdlocation is in the database
        """
        return hrdlocation in self.locationnames

    def get_location_names(self) -> List[str]:
        """
        Return a list of all hrdlocations in the database

        Returns
        -------
        list[str]
            List with all hrdlocation names
        """
        return self.locationnames

    def get_water_system(self) -> WaterSystem:
        """
        A method that returns the water system corresponding to the trajectory.

        Returns
        -------
        WaterSystem
            Water system of the trajectory
        """
        return self.watersystem
