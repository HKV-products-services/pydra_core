import numpy as np
import pandas as pd

from itertools import product

from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....settings.settings import Settings
from .....common.interpolate import InterpStruct
from .....io.hrdatabase_reader import HRDReader


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
        with HRDReader.from_settings(self.settings) as database:
            waterlevels, waveconditions = database.get_result_table(self.settings)
            ivids = database.get_input_variables()
            rvids = database.get_result_variables()

        # Remove the 'm_os' from the ivids (two times m/m_os because m is used in the wave conditions)
        ivids.remove("m_os")

        # The water level and wave conditions are based on different grids
        # Repair water level: add the missing combinations of wind direction and wind speed (filled below)
        existing = set(zip(waterlevels["r"], waterlevels["u"]))
        missing = []
        for richting, snelheid in product(np.unique(waveconditions["r"]), np.unique(waveconditions["u"])):
            if (richting, snelheid) not in existing:
                _format = waterlevels[(waterlevels["r"] == richting) & (waterlevels["u"] == 0)].copy()
                _format["u"] = snelheid
                _format["h"] = np.nan
                missing.append(_format)
        if missing:
            waterlevels = pd.concat([waterlevels] + missing, ignore_index=True)

        # Repareer Hs: add the missing combinations of wind direction and wind speed (filled below)
        existing = set(zip(waveconditions["r"], waveconditions["u"]))
        missing = []
        for richting, snelheid in product(np.unique(waterlevels["r"]), np.unique(waterlevels["u"])):
            if (richting, snelheid) not in existing:
                _format = waveconditions[(waveconditions["r"] == richting) & (waveconditions["u"] == 0)].copy()
                _format["u"] = snelheid
                _format[["hs", "tp", "tspec", "dir"]] = np.nan
                missing.append(_format)
        if missing:
            waveconditions = pd.concat([waveconditions] + missing, ignore_index=True)

        # Sorteer
        waterlevels = waterlevels.sort_values(by=["k", "r", "m", "d", "p", "u"])
        waterlevels = waterlevels.reset_index(drop=True)
        waveconditions = waveconditions.sort_values(by=["r", "h", "u"])
        waveconditions = waveconditions.reset_index(drop=True)

        # Fix water levels
        missing_h_index = waterlevels[waterlevels["h"].isna()].index
        waterlevels.loc[missing_h_index, "h"] = np.around(
            waterlevels.loc[missing_h_index - 1, "h"].to_numpy() + (waterlevels.loc[missing_h_index, "u"].to_numpy() - waterlevels.loc[missing_h_index - 1, "u"].to_numpy()) * ((waterlevels.loc[missing_h_index + 1, "h"].to_numpy() - waterlevels.loc[missing_h_index - 1, "h"].to_numpy()) / (waterlevels.loc[missing_h_index + 1, "u"].to_numpy() - waterlevels.loc[missing_h_index - 1, "u"].to_numpy())),
            4,
        )

        # Vul de NaNs golfcondities, per windrichting en waterstand in oplopende windsnelheid (eerder gevulde rijen tellen mee)
        columns = list(waveconditions.columns)
        icol = {c: columns.index(c) for c in columns}
        data = waveconditions.to_numpy(dtype=float)
        for rows in waveconditions.groupby(["r", "h"], sort=False).indices.values():
            sub = data[rows]
            for j in np.where(np.isnan(sub[:, icol["hs"]]))[0]:
                known = ~np.isnan(sub).any(axis=1)
                u = sub[known, icol["u"]]
                ws = sub[j, icol["u"]]
                for param in ["hs", "tp", "tspec"]:
                    sub[j, icol[param]] = round(self.__interp_extrapolate(ws, u, sub[known, icol[param]]), 4)

                # Niet de wave direction extrapoleren
                if u.min() <= ws <= u.max():
                    sub[j, icol["dir"]] = round(float(np.interp(ws, u, sub[known, icol["dir"]])), 4)
                elif ws > u.max():
                    sub[j, icol["dir"]] = sub[known, icol["dir"]][-1]
                else:
                    sub[j, icol["dir"]] = sub[known, icol["dir"]][0]
            data[rows] = sub
        for param in ["hs", "tp", "tspec", "dir"]:
            waveconditions[param] = data[:, icol[param]]

        # Interpoleer de golfcondities op de waterstanden, alvorens ze aan de belastingmodellen toe te voegen
        waterlevel_rows = waterlevels.groupby(["r", "u"]).indices
        h = waterlevels["h"].to_numpy()
        results = {param: np.full(len(waterlevels), np.nan) for param in ["hs", "tp", "tspec", "dir"]}
        for comb, rows in waveconditions.groupby(["r", "u"]).indices.items():
            idx = waterlevel_rows.get(comb)
            if idx is None:
                continue

            # Maak een interpolatiestructuur voor de windrichting en windsnelheid
            intstr = InterpStruct(x=h[idx], xp=data[rows, icol["h"]])

            # Interpoleer voor elk van de golfparameters (geen extrapolatie buiten het golfmodel grid)
            for param in ["hs", "tp", "tspec", "dir"]:
                fp = data[rows, icol[param]]
                results[param][idx] = intstr.interp(fp=fp, extrapolate=False, left=fp[0], right=fp[-1])
        for param in ["hs", "tp", "tspec", "dir"]:
            waterlevels[param] = results[param]

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

    @staticmethod
    def __interp_extrapolate(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        """
        Linear interpolation, with linear extrapolation of the first or last segment (as scipy interp1d with fill_value='extrapolate').
        """
        if x < xp[0]:
            i = 0
        elif x > xp[-1]:
            i = len(xp) - 2
        else:
            return float(np.interp(x, xp, fp))
        return float(fp[i] + (x - xp[i]) * (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]))
