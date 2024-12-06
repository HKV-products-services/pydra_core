import numpy as np
import pandas as pd

from typing import Dict, Union

from ....profile.profile import Profile
from .....common.interpolate import InterpStruct


class LoadingModel:
    """
    A LoadingModel is a model for one combination of wind direction and closing situation

    The LoadingModel allows to process the data (e.g. extend, refine or repair)
    """

    def __init__(
        self,
        direction: float,
        closing_situation: int,
        input_variables: list,
        result_variables: list,
    ):
        """
        Init the LoadingModel.

        Parameters
        ----------
        direction : float
            Wind direction
        closing_situation: int
            Closing situation id
        input_variables : list
            The input variable symbols (e.q. [u, q])
        result_variables : list
            The result variable symbols (e.q. [h, hs, tp, tspec, dir])
        """
        # Save the arguments into the object
        self.direction = direction
        self.closing_situation = closing_situation
        self.input_variables = input_variables
        self.result_variables = result_variables

    def initialise(self, table: pd.DataFrame) -> None:
        """
        Create a LoadingModel from a pandas DataFrame.

        Parameters
        ----------
        table : pd.DataFrame
            DataFrame with all input and output variables.
        """
        # Rename columns
        if any(ivar not in table.columns for ivar in self.input_variables):
            raise KeyError(
                f"Not all input variables are present. Expected a column for each of: {', '.join(self.input_variables)}, got: {', '.join(table.columns.tolist())}."
            )

        # Add the discretisation of each input variable to the object
        for key in self.input_variables:
            setattr(self, key, np.sort(table[key].unique()))

        # Init an empty grid for each result variable
        shape = tuple([len(getattr(self, key)) for key in self.input_variables])
        for var in self.result_variables:
            setattr(self, var, np.full(shape, np.nan))

        # Determine per input variabele where to put it into the results array
        idxs = []
        for key in self.input_variables:
            idxlist = getattr(self, key).tolist()
            idxs.append([idxlist.index(i) for i in table[key].array])

        # Add the results to the result arrays
        for rvid in self.result_variables:
            arr = getattr(self, rvid)

            # If the result variable is in the dataframe
            if rvid in table.columns:
                arr[tuple(idxs)] = table[rvid].to_numpy()

            # Otherwise, translate tspec to tp, or tp to tspec
            else:
                if rvid == "tspec" and ("tp" in table.columns):
                    arr[tuple(idxs)] = table["tp"] / 1.1
                elif rvid == "tp" and ("tspec" in table.columns):
                    arr[tuple(idxs)] = table["tspec"] * 1.1
                else:
                    raise KeyError(rvid)

    def extend(
        self,
        input_variable: str,
        grid: Union[list, np.ndarray],
        include_bounds: bool = False,
        merge_grid: bool = True,
    ) -> None:
        """
        Extend the grid of the input variable and therefore also the output variables.

        Parameters
        ----------
        input_variable : str
            The name of the input variable.
        grid : Union[list, np.ndarray]
            The 1D grid values to which the input variable should be extended.
        include_bounds : bool, optional
            Whether or not to include values within the upper and lower bound of the input variable (default is False).
        merge_grid : bool, optional
            Whether or not to merge the new grid with the existing grid of the input variable (default is True).
        """
        # Haal de huidige discretisatie van de variabele op
        xp = getattr(self, input_variable)
        axis = self.input_variables.index(input_variable)

        # If include_bounds = True, add values between the min and max, otherwise add all
        if not include_bounds:
            x = np.array(
                [
                    val
                    for val in np.atleast_1d(grid)
                    if not (xp.min() <= val <= xp.max())
                ]
            )

        # If merge_grid, add the grid to the existing values
        if merge_grid:
            x = np.array(sorted(set(np.atleast_1d(grid)).union(xp)))

        # If x is empty or the requested values are equal to those already present, continue
        if not any(x) or np.array_equal(x, xp):
            return None

        # Extend all result variables
        intstr = InterpStruct(x=x, xp=xp)
        for resvar in self.result_variables:
            # Obtain the result variable
            arr = getattr(self, resvar)
            if np.isnan(arr).any():
                raise ValueError(
                    f'[ERROR] NaN values ({np.isnan(arr).sum()}) in array "{resvar}" to interpolate.'
                )

            # Interpolate wave conditions
            if resvar in ["hs", "tp", "tspec"]:
                arr = np.maximum(0.0, intstr.interp(fp=arr, axis=axis))

            # Interpoleer wave angle
            elif resvar == "dir":
                arr = intstr.interp_angle(fp=arr, axis=axis)

            # Use of VZM, (TODO find out: interpolate between 0 and 1?)
            elif resvar == "vzm":
                arr = intstr.interp(fp=arr, axis=axis)

            # Interpolate other values
            else:
                arr = intstr.interp(fp=arr, axis=axis)

            # Add the extended result variable array back to the object
            setattr(self, resvar, arr)

        # Add the extended input variable back to the object
        setattr(self, input_variable, x)

    def refine(
        self, result_variable: str, grid: Dict[str, Union[list, np.ndarray]]
    ) -> np.ndarray:
        """
        Extend the grid of the input variable and therefore also the output variables.

        Parameters
        ----------
        input_variable : str
            The name of the input variable.
        result_variable : str
            The name of the output variable.
        grid : Dict[str, Union[list, np.ndarray]]
            Input variables with their corresponding grid to extend the result_variable to

        Returns
        -------
        np.ndarray
            The adjusted grid.
        """
        # Controleer of variabele aanwezig is
        if result_variable not in self.result_variables:
            raise KeyError(
                f"[ERROR] Result variable '{result_variable}' not in loading model."
            )

        # Obtain the array from the result variable
        belasting_int = getattr(self, result_variable)

        # Loop over the different grid items
        for inpvar, x in grid.items():
            if inpvar not in self.input_variables:
                raise KeyError(
                    f"[ERROR] Input variable '{inpvar}' not in loading model ({', '.join(self.input_variables)})"
                )

            # Obtain the current grid and axis
            xp = getattr(self, inpvar)
            axis = self.input_variables.index(inpvar)

            # If x and xp are equal, no interpolation needed
            if np.array_equal(x, xp):
                continue

            # If all x in xp, just select the requested values
            if np.isin(x, xp).all():
                idx = np.isin(xp, x)
                belasting_int = np.take(
                    belasting_int, indices=np.where(idx)[0], axis=axis
                )
                continue

            # If xp is just one value, duplicate
            if len(xp) == 1:
                belasting_int = np.tile(belasting_int, (1, 1, len(x)))
                continue

            # Interpolate
            intstr = InterpStruct(x=x, xp=xp)

            # Interpolate wave conditions
            if result_variable in ["hs", "tp", "tspec"]:
                belasting_int = np.maximum(
                    0.0, intstr.interp(fp=belasting_int, axis=axis)
                )

            # Interpolate wave direction
            elif result_variable == "dir":
                belasting_int = intstr.interp_angle(fp=belasting_int, axis=axis)

            # Use of VZM, (TODO find out: interpolate between 0 and 1?)
            elif result_variable == "vzm":
                belasting_int = intstr.interp(fp=belasting_int, axis=axis)

            # Interpolate other values
            else:
                belasting_int = intstr.interp(fp=belasting_int, axis=axis)

        # Return the interpolated array for the result variable
        return belasting_int

    def repair(
        self,
        input_variable: str,
        result_variables: Union[str, list] = None,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Make the result values of the given output variable monotonically increasing along the axis of a given input variable.

        Parameters
        ----------
        input_variable : str
            The name of the input variable along which the result values should be made monotonically increasing.
        result_variables : Union[str, list]
            The name of the output variable to be made monotonically increasing.
        epsilon : float, optional
            The minimum difference between the values of the repaired output variable (default is 1e-6).
        """
        # Obtain the relevant axis
        axis = self.input_variables.index(input_variable)

        # If no result variables are given, take all result variables in the model
        if result_variables is None:
            result_variables = self.result_variables

        # Loop over the result variables
        for var in np.atleast_1d(result_variables):
            # Obtain the result variable grid
            arr = getattr(self, var)
            rows = [np.take(arr, indices=0, axis=axis)]

            # Make monotonous increasing over the axis of the input_variable
            for i in range(1, arr.shape[axis]):
                last = rows[-1]
                nxtt = np.take(arr, indices=i, axis=axis)
                rows.append(np.maximum(last + epsilon, nxtt))

            # Add the adjusted grid to the loadingmodel
            setattr(self, var, np.stack(rows, axis=axis))

    def calculate_hbn(
        self, profile: Profile, qcrit: float, factor_hs: float, factor_tspec: float
    ):
        """
        Add hbn result variables to each of the LoadingModels.
        If 'hbn' is already defined, it will overwrite the old result variable.

        Parameters
        ----------
        profile : Profile
            The profile
        qcrit : float
            The critical discharge
        factor_hs : float
            Factor for the significant wave height, used for model uncertainty
        factor_tspec : float
            Factor for the spectral wave period, used for model uncertainty
        """
        # Controleer of de juiste resultaatwaarden aanwezig zijn
        for resvar in ["h", "hs", "tspec", "dir"]:
            if not hasattr(self, resvar):
                raise KeyError(
                    f"Resultaatvariabele '{resvar}' is nodig voor kruinhoogteberekening, maar niet aanwezig."
                )

        # Prepare wave conditions
        _h = self.h.ravel()
        _hs = np.array(self.hs * factor_hs).ravel()
        _tspec = np.array(self.tspec * factor_tspec).ravel()
        _dir = np.array(self.dir).ravel()

        # Calculate HBN
        self.hbn = np.reshape(
            [profile.calculate_crest_level(qcrit, _h, _hs, _tspec, _dir)], self.h.shape
        )

        # Add the result variable
        self.result_variables.append("hbn")
