import numpy as np

from typing import Union

from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....settings.settings import Settings
from .....io.hrdatabase_reader import HRDReader


class LoadingCoast(Loading):
    """
    Loading class for the Coast
    Water systems: Coast (North, Central, South), Waddensea (West, East) and Western scheldt
    """

    def __init__(self, settings: Settings):
        """
        Init the Loading object for the Coast

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit the from parent
        super().__init__(settings)

        # Read and process the loading
        self.read_loading()

    def read_loading(self) -> None:
        """
        Read the HR result table and create LoadingModels
        """
        # Read table
        with HRDReader.from_settings(self.settings) as database:
            table = database.get_result_table(self.settings)
            ivids = database.get_input_variables()
            rvids = database.get_result_variables()
            correction = database.get_water_level_correction(self.settings)

        # For the coast, the sea level (m) plus the water level correction is equal to the local water level (h)
        table["h"] = table["m"] + correction

        # Init LoadingModels for each combination of wind direction (r) and closing situation (k)
        for comb, deeltabel in table.groupby(["r", "k"]):
            direction, closing_situation = comb

            # Create a LoadingModel
            model = LoadingModel(direction, closing_situation, ivids, rvids)
            model.initialise(deeltabel.copy())

            # Add model to the models dictionary
            self.model[comb] = model

        # Extend and repair loadingmodels
        self._extend_loadingmodels()
        self.repair_loadingmodels(rvids)

    def repair_loadingmodels(self, result_variables: Union[list, str], epsilon: float = 0.0) -> None:
        """
        For the coast, the loading is not repaired on the grid of the database. As in Hydra-NL, the loading is repaired
        after refining it onto the grid of the statistics (see LoadingModel.refine).

        Parameters
        ----------
        result_variables : Union[list, str]
            Result variables
        epsilon : float
            The minimum increase when repairing (not used)
        """
        for _, model in self.iter_models():
            model.repair_after_refine = [inpvar for inpvar in ["u", "m"] if (inpvar in model.input_variables) and getattr(self.settings, f"{inpvar}_repair")]

    def extend_lowest_sea_level(self, m_min: float) -> None:
        """
        Extend the loading to the lowest sea level of the statistics, as in Hydra-NL (UITBREIDING_GOLVEN_IN_H).
        The wave conditions at the lowest sea level are linearly extrapolated from the two lowest sea levels in the
        database, but at least 0 and at most the value at the lowest sea level in the database. The wave direction is
        equal to the wave direction at the lowest sea level in the database.

        Parameters
        ----------
        m_min : float
            The lowest sea level of the statistics
        """
        for _, model in self.iter_models():
            # Only if the lowest sea level in the database is higher than the lowest sea level of the statistics
            if (len(model.m) < 2) or (model.m[0] <= m_min):
                continue

            # Move the sea level axis to the front
            axis = model.input_variables.index("m")
            m1, m2 = model.m[0], model.m[1]
            for var in model.result_variables:
                arr = np.moveaxis(getattr(model, var), axis, 0)

                # Extrapolate the wave conditions (at least 0, at most the value at the lowest sea level)
                if var in ["hs", "tp", "tspec"]:
                    new = np.clip(arr[0] + (arr[1] - arr[0]) / (m2 - m1) * (m_min - m1), 0.0, arr[0])

                # Wave direction equal to the wave direction at the lowest sea level
                elif var == "dir":
                    new = arr[0]

                # Extrapolate other values (e.g. the water level, which is linear in the sea level)
                else:
                    new = arr[0] + (arr[1] - arr[0]) / (m2 - m1) * (m_min - m1)

                # Add the extended result variable array back to the object
                setattr(model, var, np.moveaxis(np.concatenate([new[None], arr]), 0, axis))

            # Add the extended sea level back to the object
            model.m = np.concatenate([[m_min], model.m])
