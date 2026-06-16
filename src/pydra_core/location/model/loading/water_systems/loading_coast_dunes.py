from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....settings.settings import Settings
from .....io.database_hr import DatabaseHR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LoadingCoastDunes(Loading):
    """
    Loading class for the Coast Dunes
    Water systems: Coast Dunes
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
            table = database.get_result_table_dunes(self.settings)
            ivids = database.get_input_variables_dunes()
            rvids = database.get_result_variables_dunes()

        # Init LoadingModels for each combination of wind direction (r) and closing situation (k)
        for comb, deeltabel in table.groupby(["HRDWindDirectionId", "ClosingSituationId"]):
            direction, closing_situation = comb

            # Create a LoadingModel
            model = LoadingModel(direction, closing_situation, ivids, rvids)
            model.initialise(deeltabel.copy())

            # Add model to the models dictionary
            self.model[comb] = model

        # Extend and repair loadingmodels
        # self._extend_loadingmodels()
        # self.repair_loadingmodels(rvids)


    def plot_h_grid(table: pd.DataFrame,
                    wave_height_u_fixed: float = 0.0,
                    wave_period_u_fixed: float = 0.0):
        """
        Plot een 2D heatmap van de lokale waterstand als functie van
        'Sea water level (u)' en 'Uncertainty water level (u)',
        voor vaste waarden van 'Wave height (u)' en 'Wave period (u)'.

        Parameters
        ----------
        table : pd.DataFrame
            Output van get_result_table_dunes()
        wave_height_u_fixed : float
            Vaste waarde voor 'Wave height (u)', standaard 0.0
        wave_period_u_fixed : float
            Vaste waarde voor 'Wave period (u)', standaard 0.0
        """
        # Selecteer de gewenste slice
        mask = (
            np.isclose(table["Wave height (u)"], wave_height_u_fixed) &
            np.isclose(table["Wave period (u)"], wave_period_u_fixed)
        )
        subset = table[mask]

        # Pivot naar 2D grid: rijen = Sea water level (u), kolommen = Uncertainty water level (u)
        grid = subset.pivot(
            index="Sea water level (u)",
            columns="Uncertainty water level (u)",
            values="h"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(
            grid.columns,   # Uncertainty water level (u)
            grid.index,     # Sea water level (u)
            grid.values,
            cmap="viridis",
            shading="auto"
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Lokale waterstand h (m+NAP)")

        ax.set_xlabel("Uncertainty water level (u)")
        ax.set_ylabel("Sea water level (u)")
        ax.set_title(
            f"Lokale waterstand h\n"
            f"Wave height (u) = {wave_height_u_fixed},  "
            f"Wave period (u) = {wave_period_u_fixed}"
        )

        plt.tight_layout()
        plt.show()

    # Gebruik:
    # plot_h_grid(table)
    # Of met andere vaste waarden:
    # plot_h_grid(table, wave_height_u_fixed=4.0, wave_period_u_fixed=0.0)