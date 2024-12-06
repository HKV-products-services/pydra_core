from abc import ABC, abstractmethod
from typing import Union

from ..hrdatabase.hrdatabase import HRDatabase
from ..location.location import Location


class Calculation(ABC):
    """
    Base class for calculation modules
    """

    @abstractmethod
    def __init__(self):
        """
        Init class, differs per calculation
        """
        pass

    def calculate(self, input: Union[Location, HRDatabase]):
        """
        Execute a calculation

        Parameters
        ----------
        input : Union[Location, HRDatabase]
            The input
        """
        # Depending on the type of input, call CalculateLocation
        if isinstance(input, Location):
            return self.calculate_location(input)

        elif isinstance(input, HRDatabase):
            return {
                loc: self.calculate_location(input.get_location(loc)) for loc in input
            }

        else:
            raise NotImplementedError("[ERROR] Input type not implemented")

    @abstractmethod
    def calculate_location(self, location: Location):
        """
        Executes a calculation for a location

        Parameter
        ---------
        location : Location
            The Location object
        """
        pass
