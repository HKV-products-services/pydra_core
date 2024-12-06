import numpy as np
import pandas as pd

from itertools import product
from scipy.interpolate import interp1d

from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....settings.settings import Settings
from .....common.interpolate import InterpStruct
from .....io.database_hr import DatabaseHR


class LoadingEasternScheldt(Loading):
    """
    Loading class for the EasternScheldt
    Water systems: Eastern Scheldt
    """

    def __init__(self, settings: Settings):
        """
        Init the Loading object for the Eastern Scheldt

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
            waterlevels, waveconditions = database.get_result_table_eastern_scheldt(
                self.settings
            )
            ivids = database.get_input_variables()
            rvids = database.get_result_variables()

        # Remove the 'm_os' from the ivids (two times m/m_os because m is used in the wave conditions)
        ivids.remove("m_os")

        # The water level and wave conditions are based on different grids
        # Repair water level
        for richting, snelheid in product(
            np.unique(waveconditions["r"]), np.unique(waveconditions["u"])
        ):
            # Kijk of deze combi bestaat
            if (
                waterlevels["r"].eq(richting) & waterlevels["u"].eq(snelheid)
            ).sum() == 0:
                _format = waterlevels[
                    waterlevels["r"].eq(richting) & waterlevels["u"].eq(0)
                ].copy()
                _format.loc[:, "u"] = snelheid
                _format.loc[:, "h"] = np.NaN
                waterlevels = pd.concat([waterlevels, _format], ignore_index=True)

        # Repareer Hs
        for richting, snelheid in product(
            np.unique(waterlevels["r"]), np.unique(waterlevels["u"])
        ):
            # Kijk of deze combi bestaat
            if (
                waveconditions["r"].eq(richting) & waveconditions["u"].eq(snelheid)
            ).sum() == 0:
                _format = waveconditions[
                    waveconditions["r"].eq(richting) & waveconditions["u"].eq(0)
                ].copy()
                _format.loc[:, "u"] = snelheid
                _format.loc[:, "hs"] = np.NaN
                _format.loc[:, "tp"] = np.NaN
                _format.loc[:, "tspec"] = np.NaN
                _format.loc[:, "dir"] = np.NaN
                waveconditions = pd.concat([waveconditions, _format], ignore_index=True)

        # Sorteer
        waterlevels = waterlevels.sort_values(by=["k", "r", "m", "d", "p", "u"])
        waterlevels = waterlevels.reset_index(drop=True)
        waveconditions = waveconditions.sort_values(by=["r", "h", "u"])
        waveconditions = waveconditions.reset_index(drop=True)

        # Fix water levels
        missing_h_index = waterlevels[waterlevels["h"].isna()].index
        waterlevels.loc[missing_h_index, "h"] = np.around(
            waterlevels.loc[missing_h_index - 1, "h"].to_numpy()
            + (
                waterlevels.loc[missing_h_index, "u"].to_numpy()
                - waterlevels.loc[missing_h_index - 1, "u"].to_numpy()
            )
            * (
                (
                    waterlevels.loc[missing_h_index + 1, "h"].to_numpy()
                    - waterlevels.loc[missing_h_index - 1, "h"].to_numpy()
                )
                / (
                    waterlevels.loc[missing_h_index + 1, "u"].to_numpy()
                    - waterlevels.loc[missing_h_index - 1, "u"].to_numpy()
                )
            ),
            4,
        )

        # Vul de NaNs golfcondities
        for n, row in waveconditions[waveconditions["hs"].isna()].iterrows():
            wd = row["r"]
            wl = row["h"]
            ws = row["u"]
            grid = waveconditions[
                waveconditions["r"].eq(wd) & waveconditions["h"].eq(wl)
            ].dropna()
            waveconditions.loc[n, "hs"] = round(
                float(interp1d(grid["u"], grid["hs"], fill_value="extrapolate")(ws)), 4
            )
            waveconditions.loc[n, "tp"] = round(
                float(interp1d(grid["u"], grid["tp"], fill_value="extrapolate")(ws)), 4
            )
            waveconditions.loc[n, "tspec"] = round(
                float(interp1d(grid["u"], grid["tspec"], fill_value="extrapolate")(ws)),
                4,
            )

            # Niet de wave direction extrapoleren
            if np.min(grid["u"]) <= ws and ws <= np.max(grid["u"]):
                waveconditions.loc[n, "dir"] = round(
                    float(interp1d(grid["u"], grid["dir"])(ws)), 4
                )
            elif ws > np.max(grid["u"]):
                waveconditions.loc[n, "dir"] = grid["dir"].to_numpy()[-1]
            elif ws < np.min(grid["u"]):
                waveconditions.loc[n, "dir"] = grid["dir"].to_numpy()[0]

        # Interpoleer de golfcondities op de waterstanden, alvorens ze aan de belastingmodellen toe te voegen
        for (richting, snelheid), wavecond_ur in waveconditions.groupby(["r", "u"]):
            # Maak een index voor de selectie van windrichting en windsnelheid
            idx = waterlevels["r"].eq(richting) & waterlevels["u"].eq(snelheid)

            # Maak een interpolatiestructuur voor de windrichting en windsnelheid
            intstr = InterpStruct(
                x=waterlevels.loc[idx, "h"].to_numpy(), xp=wavecond_ur["h"].to_numpy()
            )

            # Interpoleer voor elk van de golfparameters
            for param in ["hs", "tp", "tspec", "dir"]:
                waterlevels.loc[idx, param] = intstr.interp(
                    fp=wavecond_ur[param].to_numpy()
                )

        # Init LoadingModels for each combination of wind direction (r) and closing situation (k)
        for comb, deeltabel in waterlevels.groupby(["r", "k"]):
            direction, closing_situation = comb

            # Create a LoadingModel
            model = LoadingModel(direction, closing_situation, ivids, rvids)
            model.initialise(deeltabel.copy())

            # Add model to the models dictionary
            self.model[comb] = model

        # Breidt het belastingmodel uit op basis van de invoergegevens
        self._extend_loadingmodels()
        self.repair_loadingmodels(rvids)
