from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....settings.settings import Settings
from .....io.database_hr import DatabaseHR


class LoadingLowerRiver(Loading):
    """
    Loading class for Lower Rivers
    Water systems: Tidal River Rhine, Tidal River Meuse
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
        with DatabaseHR(self.settings.database_path) as database:
            table = database.get_result_table(self.settings)
            rvids = database.get_result_variables()
            ivids_west = database.get_input_variables()

        # For lower tidal rivers, there is a different loading model for sector east and west
        ivids_east = ivids_west.copy()
        ivids_east.remove("m")

        # Init LoadingModels for each combination of wind direction (r) and closing situation (k)
        for comb, deeltabel in table.groupby(["r", "k"]):
            wind_direction, closing_situation = comb

            # Sector west
            if ((wind_direction >= 0.0) and (wind_direction <= 11.25)) or (
                (wind_direction > 212.75) and (wind_direction <= 360.0)
            ):
                model = LoadingModel(
                    wind_direction, closing_situation, ivids_west, rvids
                )

            # Sector east
            else:
                model = LoadingModel(
                    wind_direction, closing_situation, ivids_east, rvids
                )

            # Init loading model
            model.initialise(deeltabel.copy())

            # Add model to the models dictionary
            self.model[comb] = model

        # Extend and repair loadingmodels
        self._extend_loadingmodels()
        self.repair_loadingmodels(rvids)
