import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Union

from ...settings.settings import Settings
from ....common.interpolate import InterpStruct
# from ....io.database_hr import DatabaseHR


class Loading(ABC):
    """
    Loading Abstract Base Class. This class contains all LoadingModels.

    Attributes
    ----------
    settings : Settings
        The Settings object
    model : dict
        Dictionary with all LoadingModels (wind direction, closing situation)
    database : DatabaseHR
        Connection with HR Database
    """

    def __init__(self, settings: Settings):
        """
        Init the Loading object

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Save settings and init dictionary for all models
        self.settings = settings
        self.model = {}

    @abstractmethod
    def read_loading(self) -> None:
        """
        Read the HR and create Loading Models
        """
        pass

    def iter_models(self):
        """
        Iterate through each of the LoadingModels.
        LoadingModels are ordered by WindDirection, ClosingSituation
        """
        for discreet, model in self.model.items():
            yield discreet, model

    def _extend_loadingmodels(self) -> None:
        """
        Preprocess the LoadingModels.
        Makes sure the discretisations in every LoadingModel are the same.
        """
        # Obtain the wind directions and closing situations
        self.r = sorted(set([r[0] for r in list(self.model.keys())]))
        self.k = sorted(set([k[1] for k in list(self.model.keys())]))

        # Collect all unique wind speeds from all loadingmodels
        utot = sorted(set([u for _, model in self.iter_models() for u in model.u]))

        # Extend the stochasts in each loadingmodel
        for _, model in self.iter_models():
            # Extend the wind speed
            model.extend("u", utot)

            # Extend for discharge
            if "q" in model.input_variables:
                if len(model.q) > 1:
                    model.extend(
                        "q",
                        list(filter(None, [self.settings.q_min, self.settings.q_max])),
                    )

            # Extend for sea level
            if "m" in model.input_variables:
                if len(model.m) > 1:
                    model.extend(
                        "m",
                        list(filter(None, [self.settings.m_min, self.settings.m_max])),
                    )

            # Extend for lake level
            if "a" in model.input_variables:
                if len(model.a) > 1:
                    model.extend(
                        "a",
                        list(filter(None, [self.settings.a_min, self.settings.a_max])),
                    )

    def repair_loadingmodels(
        self, result_variables: Union[list, str], epsilon: float = 1e-6
    ) -> None:
        """
        Repair the result variables for all LoadingModels
        Repairs depending on the '{input_variable}_repair' flag in the Settings

        Parameters
        ----------
        result_variables : Union[list, str]
            Result variables
        epsilon : float
            The minimum increase when repairing (default : 1e-6)
        """
        # For each LoadingModel (wind direction and closing situation)
        for _, model in self.iter_models():
            # Loop over the inputvariables
            for inpvar in model.input_variables:
                # Only repair base stochastics
                if inpvar not in ["u", "m", "q", "a"]:
                    continue

                # Only if the flag '{input_variable}_repair' is True
                if not getattr(self.settings, f"{inpvar}_repair"):
                    continue

                # For each result variable
                for rv in np.atleast_1d(result_variables):
                    # If the result variable from the argument is not in the LoadingModel or if it the wave direction, skip
                    if (rv not in model.result_variables) or (rv == "dir"):
                        continue

                    # Repair the LoadingModel
                    model.repair(inpvar, rv, epsilon)

    def get_result_variable_statistic(self, result_variable: str, stat: str, args=()):
        """
        Obtain statistics for a result variable. (e.g. min, max)

        Parameters
        ----------
        result_variable : str
            Result variable (h, hs, tp, tspec, hbn, qov, etc.)
        statistic : str
            NumPy function, e.g. 'max', 'min'
        args : tuple
            Optional if required by NumPy function

        Returns
        -------
        statistic : float
            The statistic
        """
        # Obtain raveled result variable array
        arr = self.get_result_variable_raveled(result_variable)

        # Obtain the statistics using the raveled result variable array
        result = getattr(np, stat)(arr, *args)

        # Return statistic
        return result

    def get_result_variable_raveled(self, result_variable: str) -> np.ndarray:
        """
        Obtain all values for a result variable in one array

        Parameters
        ----------
        result_variable : str
            Result variable (h, hs, tp, tspec, hbn, qov, etc.)

        Returns
        -------
        np.ndarray
            1D array with all values of a certain result variable
        """
        # Loop over every discrete model and add all result variables values to an array
        arr = np.concatenate(
            [
                getattr(model, result_variable).ravel()
                for _, model in self.iter_models()
            ],
            axis=-1,
        )

        # Return the array
        return arr

    def get_quantile_range(
        self,
        result_variable: str,
        lower_quantile: float,
        upper_quantile: float,
        round_digits: int = None,
    ) -> Tuple[float, float]:
        """Geef kwantielen van een gekozen variabele. De uitkomst kan worden
        afgerond wanneer 'rounddigits' is gegeven.

        Parameters
        ----------
        result_variable : str
            Result variable (h, hs, tp, tspec, hbn, qov, etc.)
        lower_quantile : float
            Lower quantile, between 0.0 en 1.0
        upper_quantile : float
            Upper quantile, between 0.0 en 1.0
        round_digits : int, optional
            Aantal digits waarop afgerond wordt, standaard None

        Returns
        -------
        tuple
            Parameterwaarden voor het onder en bovenkwantiel.
        """
        # Obtain all values of the result variable in the discrete models in one array
        arr = self.get_result_variable_raveled(result_variable)

        # Determine lower and upper quantile
        lower = np.quantile(arr, lower_quantile)
        upper = np.quantile(arr, upper_quantile)

        # If required, apply rounding
        if round_digits is not None:
            lower = np.round(lower, round_digits)
            upper = np.round(upper, round_digits)

        # Return the quantiles
        return lower, upper

    def get_wave_conditions(
        self,
        direction: float,
        waterlevel: Union[int, float, list, np.ndarray],
        extrapolate: bool = True,
    ) -> Tuple[dict, np.ndarray]:
        """
        Return the wave conditions given a wind direction and wind speed.

        The function assumes the wave conditions are a function of:
        - The local water level
        - The wind direction
        - The wind speed

        Parameters
        ----------
        direction : float
            The wind direction
        waterlevel : Union[float, list, np.ndarray]
            Water level(s)
        extrapolate : bool, optional
            Whether or not to extrapolate (default : True)

        Returns
        -------
        Tuple[dict{str : np.ndarray}, np.ndarray]
            1.) A dictionary with
                key : The loading variable (e.g. h, hs, tspec, tp, dir)
                np.ndarray : 2D array with size [u, waterlevel]
            The wave conditions for different wind speed (nd-array)
            2.) The wind speed discretistation (1d-array)
        """
        # Convert the water level to a ndarray
        if isinstance(waterlevel, (float, int)):
            waterlevel = np.array([waterlevel])
        elif isinstance(waterlevel, list):
            waterlevel = np.array(waterlevel)

        # Result variables
        resvars = ["h", "hs", "tspec", "tp", "dir"]

        # Obtain all loading models for one wind direction
        loading_r = [model for key, model in self.model.items() if direction in key]
        first_model = loading_r[0]

        # Create an empty dictionary for the wave conditions
        wave_conditions = {
            resvar: np.zeros((len(waterlevel), len(first_model.u)), dtype=np.float32)
            for resvar in resvars
        }

        # Change the loading shape such that all wave conditions are available per wind direction and wind speed
        loading_reshaped = {}

        # For each loading parameter (resvar)
        for resvar in resvars:
            # Obtain the loading array for each loading model, stack over the last dimension such
            # that the input order of the other stochastics are unchanged: e.g. parts[model, u, m, d, p]
            parts = []
            for model in loading_r:
                if hasattr(model, resvar):
                    parts.append(getattr(model, resvar))
                else:
                    if (resvar == "tp") and hasattr(model, "tspec"):
                        parts.append(getattr(model, "tspec") * 1.1)
                    elif (resvar == "tspec") and hasattr(model, "tp"):
                        parts.append(getattr(model, "tp") / 1.1)
                    else:
                        raise KeyError(f'"{resvar}" not present.')

            # e.g. arr[u, m, d, p, model]
            arr = np.stack(parts, axis=-1)

            # Reshape
            upos = first_model.input_variables.index("u")

            # If the axis of the wind speed is not the first dimension, swap axes
            if upos != 0:
                # Swap to axis with the wind speed to the first dimension
                tmparr = np.swapaxes(arr, 0, upos)

                # Reshaped [u, comb van (m, d, p en de sluitsituatie (model))]
                reshaped = tmparr.reshape((tmparr.shape[0], np.prod(tmparr.shape[1:])))

            # Otherwise, only reshape the array
            else:
                reshaped = arr.reshape((arr.shape[0], np.prod(arr.shape[1:])))

            # Determine the order of the water levels
            if resvar == "h":
                order = np.argsort(reshaped, axis=1)

            # Sort
            i = np.arange(reshaped.shape[0])[:, None]
            loading_reshaped[resvar] = reshaped[i, order]

        # Obtain the waterlevels, over which will be interpolated
        hbelast = loading_reshaped["h"]

        # Interpolate over the water level, for the wind speed
        for iu in range(len(first_model.u)):
            # Prepare interpolation
            index = np.unique(hbelast[iu].round(3), return_index=True)[1]
            h_unique = hbelast[iu][index]

            # If there is only one water level given, we dont need to interpolate
            if len(h_unique) == 1:
                for resvar in resvars:
                    wave_conditions[resvar][:, iu] = loading_reshaped[resvar][iu][index]

            # For more water levels, we do need to interpolate
            else:
                intstr = InterpStruct(x=waterlevel, xp=h_unique)
                for resvar in resvars:
                    # Dont interpolate the water level, we will add this at the end
                    if resvar == "h":
                        continue
                    fp = loading_reshaped[resvar][iu][index]

                    # If all result variables are 0, skip
                    if (fp == 0.0).all():
                        continue

                    # If there is only one result variable, apply this one
                    if (fp == fp[0]).all():
                        intbelast = fp[0]

                    # Otherwise, interpolate the wave loading
                    else:
                        if resvar == "dir":
                            intbelast = intstr.interp_angle(
                                fp=fp, extrapolate=extrapolate
                            )
                        else:
                            intbelast = np.maximum(
                                0, intstr.interp(fp=fp, extrapolate=extrapolate)
                            )

                    # Add the resvar to the result 'wave_conditions' dictionary
                    wave_conditions[resvar][:, iu] = intbelast

        # Add the water level
        wave_conditions["h"][:, :] = waterlevel[:, None]

        # Return wave conditions and the wind speed discretisation
        return wave_conditions, first_model.u
