from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....settings.settings import Settings
from .....io.database_hr import DatabaseHR


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
        with DatabaseHR(self.settings.database_path) as database:
            table = database.get_result_table(self.settings)
            ivids = database.get_input_variables()
            rvids = database.get_result_variables()

        # For the coast, the sea level (m) is equal to the local water level (h)
        table["h"] = table["m"]

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
