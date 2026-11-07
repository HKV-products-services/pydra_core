from typing import Union

import numpy as np

from .loading_lower_rivier import LoadingLowerRiver


class LoadingHollandscheIJssel(LoadingLowerRiver):
    """
    Loading class for the Hollandsche IJssel
    The loading of the Rhine-Meuse delta, with the order of operations of Hydra-NL for the Hollandsche IJssel
    (database_uitlezen_HIJ.f90): the water level is repaired on the grid of the database before it is extended,
    and the wave height and wave period are not repaired. In the closed states the water level is not monotonic
    in the discharge; extending first would interpolate across that jump and the repair would carry it over the
    grid.
    """

    # Result variables that are never repaired
    NOT_REPAIRED = ("hs", "tp", "tspec")

    def _extend_loadingmodels(self) -> None:
        """
        Repair the water level on the grid of the database, then extend the LoadingModels.
        """
        self.repair_loadingmodels("h")
        super()._extend_loadingmodels()

    def repair_loadingmodels(self, result_variables: Union[list, str], epsilon: float = 0.0) -> None:
        """
        Repair the result variables for all LoadingModels, except the wave height and wave period.

        Parameters
        ----------
        result_variables : Union[list, str]
            Result variables
        epsilon : float
            The minimum increase when repairing (default : 0.0, a running maximum as in Hydra-NL)
        """
        if isinstance(result_variables, str):
            if result_variables in self.NOT_REPAIRED:
                return
        else:
            result_variables = [rv for rv in np.atleast_1d(result_variables) if rv not in self.NOT_REPAIRED]
        super().repair_loadingmodels(result_variables, epsilon)
