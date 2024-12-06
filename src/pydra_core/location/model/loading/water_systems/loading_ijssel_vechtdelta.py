from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....settings.settings import Settings
from .....io.database_hr import DatabaseHR


class LoadingIJsselVechtdelta(Loading):
    """
    Loading class for the IJssel-Vechtdelta
    Water systems: IJssel Delta, Vecht Delta
    """

    def __init__(self, settings: Settings):
        """
        Init the Loading object for the Upper Rivers

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

        # Check if there are wave conditions present or whether they should be derived with Bretschneider
        if "hs" not in table.columns:
            raise NotImplementedError("[ERROR] Bretschneider")

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
