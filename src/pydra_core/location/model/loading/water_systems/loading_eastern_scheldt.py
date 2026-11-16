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

        # The wave table (u, h) per wind direction as in the database, for the crest levels (see 'get_wave_table_conditions')
        self.wave_table = {}
        for direction, table in waveconditions.groupby("r"):
            u, h = np.unique(table["u"]), np.unique(table["h"])
            self.wave_table[direction] = (u, h, {param: table.pivot_table(index="u", columns="h", values=param).reindex(index=u, columns=h).to_numpy() for param in ["hs", "tp", "tspec", "dir"]})

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

        # Interpolate the wave direction to the (not yet repaired) water levels. Hs, Tp and Tspec are read after the
        # water level is repaired, see below.
        waterlevel_rows = waterlevels.groupby(["r", "u"]).indices
        h = waterlevels["h"].to_numpy()
        direction_at_h = np.full(len(waterlevels), np.nan)
        for comb, rows in waveconditions.groupby(["r", "u"]).indices.items():
            idx = waterlevel_rows.get(comb)
            if idx is None:
                continue

            # No extrapolation outside the wave model grid
            intstr = InterpStruct(x=h[idx], xp=data[rows, icol["h"]])
            fp = data[rows, icol["dir"]]
            direction_at_h[idx] = intstr.interp(fp=fp, extrapolate=False, left=fp[0], right=fp[-1])
        waterlevels["dir"] = direction_at_h
        for param in ["hs", "tp", "tspec"]:
            waterlevels[param] = np.nan

        # Init LoadingModels for each combination of wind direction (r) and closing situation (k)
        for comb, deeltabel in waterlevels.groupby(["r", "k"]):
            direction, closing_situation = comb

            # Create a LoadingModel
            model = LoadingModel(direction, closing_situation, ivids, rvids)
            model.initialise(deeltabel.copy())

            # Add model to the models dictionary
            self.model[comb] = model

        # Extend the loading models and repair the water level
        self._extend_loadingmodels()
        self.repair_loadingmodels(["h"])

        # Read Hs, Tp and Tspec from the wave table at the repaired water level and then repair the waves, as Hydra-NL
        # Zout ('PLAATS_GOLFPARAMETER', then 'REPAREREN_ZOUT'). The waves then belong to the water level that is
        # calculated with.
        for (direction, _), model in self.iter_models():
            self.__read_waves_at_water_level(direction, model)
        self.repair_loadingmodels([rv for rv in rvids if rv != "h"])

    def get_wave_table_conditions(self, direction: float) -> tuple[np.ndarray, dict, np.ndarray]:
        """
        Return the wave conditions on the water levels of the wave table in the database, for the crest levels.

        As in Hydra-NL Zout ('DIJKBELASTING_ZOUT' and 'PLAATS_BELASTINGEN'), the crest level is calculated on the wave table
        and then interpolated along the water level; the waves are never interpolated between water levels. At dry
        points of the wave table (Hs = 0) the database has wave direction 360. Interpolating the waves between a dry
        and a wet water level gives waves with a direction that belongs to neither point, and the loading models
        contain repaired water levels whose waves were read at the unrepaired water level.

        Parameters
        ----------
        direction : float
            The wind direction

        Returns
        -------
        Tuple[np.ndarray, dict{str : np.ndarray}, np.ndarray]
            1.) The water levels of the wave table (1d-array)
            2.) A dictionary with the loading variables (h, hs, tp, tspec, dir), 2D arrays with size [water level, u]
            3.) The wind speed discretisation of the loading models (1d-array)
        """
        table_u, waterlevels, table = self.wave_table[direction]
        u = next(model.u for key, model in self.model.items() if direction in key)

        # Linear along u, with linear extrapolation of the first or last segment
        i = np.clip(np.searchsorted(table_u, u, side="right") - 1, 0, len(table_u) - 2)
        w = ((u - table_u[i]) / (table_u[i + 1] - table_u[i]))[:, None]
        wave_conditions = {param: np.maximum(0.0, table[param][i] * (1 - w) + table[param][i + 1] * w).T for param in ["hs", "tp", "tspec"]}

        # The wave direction along u on the unwrapped angle, so it does not run the long way between 360 and 0
        angle = np.unwrap(np.radians(table["dir"]), axis=0)
        wave_conditions["dir"] = np.degrees(angle[i] * (1 - w) + angle[i + 1] * w).T % 360.0
        wave_conditions["h"] = np.repeat(waterlevels[:, None], len(u), axis=1)
        return waterlevels, {param: values.astype(np.float32) for param, values in wave_conditions.items()}, u

    def __read_waves_at_water_level(self, direction: float, model: LoadingModel) -> None:
        """Hs, Tp and Tspec of the loading model from the wave table at its water levels, bilinear with linear
        extrapolation along u and h ('INEXTRP2D')."""
        table_u, table_h, table = self.wave_table[direction]
        axis_u = model.input_variables.index("u")
        h = np.moveaxis(model.h, axis_u, 0)

        ih = np.clip(np.searchsorted(table_h, h, side="right") - 1, 0, len(table_h) - 2)
        wh = (h - table_h[ih]) / (table_h[ih + 1] - table_h[ih])
        shape_u = (-1,) + (1,) * (h.ndim - 1)
        iu = np.clip(np.searchsorted(table_u, model.u, side="right") - 1, 0, len(table_u) - 2)
        wu = ((model.u - table_u[iu]) / (table_u[iu + 1] - table_u[iu])).reshape(shape_u)
        iu = iu.reshape(shape_u)

        for param in ["hs", "tp", "tspec"]:
            if param not in model.result_variables:
                continue
            fp = table[param]
            at_lower_u = fp[iu, ih] * (1 - wh) + fp[iu, ih + 1] * wh
            at_upper_u = fp[iu + 1, ih] * (1 - wh) + fp[iu + 1, ih + 1] * wh
            values = np.maximum(0.0, at_lower_u * (1 - wu) + at_upper_u * wu)
            setattr(model, param, np.moveaxis(values, 0, axis_u).astype(getattr(model, param).dtype))

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
