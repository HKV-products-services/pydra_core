import ast
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Union

from .foreshore import Foreshore
from .profile_loading import ProfileLoading
from ...common.enum import Breakwater
from ...common.interpolate import Interpolate


class Profile:
    """
    Profile class

    TODO: Implement support for sheet piles
    """

    # Profile name
    profile_name = None

    # Breakwater
    breakwater_type = Breakwater.NO_BREAKWATER
    breakwater_level = 0.0

    # Foreshore
    foreshore_x: None | list = None
    foreshore_y: None | list = None

    # Dike schematisation
    dike_orientation: None | float = None
    dike_crest_level: None | float = None
    dike_x: None | list = None
    dike_y: None | list = None
    dike_roughness: None | list = None

    def __init__(self, profile_name: str = "Profile", crest_level: float = None, orientation: float = None, cota_slope: float = None):
        """
        Create a new profile
        To generate a fast profile, it is possible to include the crest level,
        orientation, and slope (cota) as arguments. The toe of this preset will
        be at -10.

        Parameters
        ----------
        profile_name: str
            Name of the profile (default: 'Profile')
        crest_level: float, optional
            Crest level of the dike
        orientation: float, optional
            Orientation of the dike
        cota_slope: float, optional
            Slopes of the dike (cot a)
        """
        # Save the profile name
        self.profile_name = profile_name

        # Arguments
        if crest_level is not None:
            self.set_dike_crest_level(crest_level)
        if orientation is not None:
            self.set_dike_orientation(orientation)
        if cota_slope is not None:
            self.set_dike_geometry([-10 * cota_slope, crest_level * cota_slope], [-10, crest_level])

    def validate_profile(self) -> bool:
        """
        Function to validate the profile.

        Returns
        -------
        bool
            True if the profile is OK
        """
        # TODO: Improve validation
        # Check necessary components
        # Orientation
        if self.dike_orientation is None:
            return False

        # Crest height
        if self.dike_crest_level is None:
            return False

        # Geometry
        if (self.dike_x is None) or (self.dike_y is None) or (self.dike_roughness is None):
            return False

        return True

    def set_dike_orientation(self, dike_orientation: float):
        """
        Change the dike orientation

        Parameters
        ----------
        dike_orientation : float
            The dike orientation
        """
        self.dike_orientation = dike_orientation

    def set_dike_crest_level(self, dike_crest_level: float):
        """
        Change the crest level

        Parameters
        ----------
        dike_crest_level : float
            The crest level of the dike
        """
        self.dike_crest_level = dike_crest_level

    def set_dike_geometry(self, dike_x: list, dike_y: list, dike_roughness: list = None):
        """
        Change the geometry of the outer slope of the dike

        Parameters
        ----------
        dike_x : list
            A list with the x coordinates of the profile
        dike_y : list
            A list with the y coordinates of the profile
        dike_roughness : list, optional
            A list with the roughness of the profile (default : all parts 1.0)
        """
        # Make sure the dike geometry starts at (x = 0)
        min_x = np.min(dike_x)

        # If dike roughness is None, the roughness of every part is 1.0
        if dike_roughness is None:
            dike_roughness = [1.0] * len(dike_x)

        # Save the dike geometry coordinates
        self.dike_x = dike_x - min_x
        self.dike_y = dike_y
        self.dike_roughness = dike_roughness

    def has_foreshore(self) -> bool:
        """
        Returns whether the profile has a foreshore or breakwater

        Returns
        -------
        bool
            True if the profile has a foreshore or breakwater
        """
        # Check for a breakwater
        if self.breakwater_type != Breakwater.NO_BREAKWATER:
            return True

        # Check for a foreshore
        if self.foreshore_x is not None:
            return True

        # No foreshore or breakwater
        return False

    def set_foreshore_geometry(self, foreshore_x: list = None, foreshore_y: list = None):
        """
        Change the geometry of the foreshore
        Setting the foreshore x and y coordinates to None will remove the foreshore.

        Parameters
        ----------
        foreshore_x : list
            A list with the x coordinates of the foreshore (default: None)
        foreshore_y : list
            A list with the y coordinates of the foreshore (default: None)
        """
        # Add foreshore
        if foreshore_x is not None:
            # Make sure the foreshore ends at (x = 0)
            max_x = np.max(foreshore_x)

            # Save the foreshore coordinates
            self.foreshore_x = foreshore_x - max_x
            self.foreshore_y = foreshore_y

        # Delete foreshore
        else:
            self.foreshore_x = None
            self.foreshore_y = None

    def remove_foreshore(self) -> None:
        """
        Remove the foreshore
        Wrapper for set_foreshore_geometry()
        """
        self.set_foreshore_geometry(foreshore_x=None, foreshore_y=None)

    def set_breakwater(self, breakwater_type: Breakwater = None, breakwater_level: float = 0.0):
        """
        Change the breakwater
        """
        # None equals no breakwater
        if breakwater_type is None:
            breakwater_type = Breakwater.NO_BREAKWATER

        # If there is no breakwater, the height will be set to None
        if breakwater_type == Breakwater.NO_BREAKWATER:
            breakwater_level = 0.0

        # Apply settings
        self.breakwater_type = breakwater_type
        self.breakwater_level = breakwater_level

    def remove_breakwater(self) -> None:
        """
        Remove the breakwater
        Wrapper for set_breakwater()
        """
        self.set_breakwater(breakwater_type=Breakwater.NO_BREAKWATER, breakwater_level=0.0)

    def calculate_overtopping(
        self,
        water_level: Union[float, list],
        significant_wave_height: Union[float, list],
        spectral_wave_period: Union[float, list],
        wave_direction: Union[float, list],
        tp_tspec: float = 1.1,
        dll_settings: dict = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the overtopping discharge

        Parameters
        ----------
        water_level : Union[float, list]
            List or float with the water level
        significant_wave_height : Union[float, list]
            List or float with the significant wave height
        spectral_wave_period : Union[float, list]
            List or float with the spectral wave period
        wave_direction : Union[float, list]
            List or float with the wave direction
        tp_tspec : float, optional
            Ratio between Tp and Tspec, only used for dam and foreshore (default : 1.1)

        Returns
        -------
        Union[float, np.ndarray]
            List or float with the overtopping discharge
        """
        # Transform wave conditions
        water_level, significant_wave_height, spectral_wave_period, wave_direction = self.transform_wave_conditions(
            water_level,
            significant_wave_height,
            spectral_wave_period,
            wave_direction,
            tp_tspec,
            force_array=True,
        )

        # Create profile loading
        profile_loading = ProfileLoading(self, dll_settings)

        # Calculate
        qov = []
        for _h, _hs, _tspec, _dir in zip(water_level, significant_wave_height, spectral_wave_period, wave_direction):
            qov.append(profile_loading.calculate_discharge(_h, _hs, _tspec, _dir))

        # Return
        return qov[0] if len(qov) == 1 else np.array(qov)

    def calculate_runup(
        self,
        water_level: Union[float, list],
        significant_wave_height: Union[float, list],
        spectral_wave_period: Union[float, list],
        wave_direction: Union[float, list],
        tp_tspec: float = 1.1,
        dll_settings: dict = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the runup height

        Parameters
        ----------
        water_level : Union[float, list]
            List or float with the water level
        significant_wave_height : Union[float, list]
            List or float with the significant wave height
        spectral_wave_period : Union[float, list]
            List or float with the spectral wave period
        wave_direction : Union[float, list]
            List or float with the wave direction
        tp_tspec : float, optional
            Ratio between Tp and Tspec, only used for dam and foreshore (default : 1.1)

        Returns
        -------
        Union[float, list]
            List or float with the runup height
        """
        # Transform wave conditions
        water_level, significant_wave_height, spectral_wave_period, wave_direction = self.transform_wave_conditions(
            water_level,
            significant_wave_height,
            spectral_wave_period,
            wave_direction,
            tp_tspec,
            force_array=True,
        )

        # Create profile loading
        profile_loading = ProfileLoading(self, dll_settings)

        # Calculate
        ru2p = []
        for _h, _hs, _tspec, _dir in zip(water_level, significant_wave_height, spectral_wave_period, wave_direction):
            ru2p.append(profile_loading.calculate_runup(_h, _hs, _tspec, _dir))

        # Return
        return ru2p[0] if len(ru2p) == 1 else np.array(ru2p)

    def calculate_crest_level(
        self,
        q_overtopping: float,
        water_level: Union[float, list],
        significant_wave_height: Union[float, list],
        spectral_wave_period: Union[float, list],
        wave_direction: Union[float, list],
        tp_tspec: float = 1.1,
        dll_settings: dict = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the crest level for a given overtopping discharge

        Parameters
        ----------
        q_overtopping : float
            Critical overtopping discharge
        water_level : Union[float, list]
            List or float with the water level
        significant_wave_height : Union[float, list]
            List or float with the significant wave height
        spectral_wave_period : Union[float, list]
            List or float with the spectral wave period
        wave_direction : Union[float, list]
            List or float with the wave direction
        tp_tspec : float, optional
            Ratio between Tp and Tspec, only used for dam and foreshore (default : 1.1)

        Returns
        -------
        Union[float, list]
            List or float with the crest level
        """
        # Transform wave conditions
        water_level, significant_wave_height, spectral_wave_period, wave_direction = self.transform_wave_conditions(
            water_level,
            significant_wave_height,
            spectral_wave_period,
            wave_direction,
            tp_tspec,
            force_array=True,
        )

        # Length of the list
        n_q = 1 if isinstance(q_overtopping, (float, int)) else len(q_overtopping)

        # If q_overtopping is an int or float
        if isinstance(q_overtopping, (float, int)):
            q_overtopping = [q_overtopping] * len(water_level)

        # If q_overtopping is a list and water_level has length 1
        elif n_q > 1 and len(water_level) == 1:
            water_level = [water_level] * n_q
            significant_wave_height = [significant_wave_height] * n_q
            spectral_wave_period = [spectral_wave_period] * n_q
            wave_direction = [wave_direction] * n_q

        # Uneven length of arrays
        else:
            raise ValueError("[ERROR] Uneven length of arrays")

        # To numpy
        q_overtopping = np.array(q_overtopping)
        water_level = np.array(water_level)
        significant_wave_height = np.array(significant_wave_height)
        spectral_wave_period = np.array(spectral_wave_period)
        wave_direction = np.array(wave_direction)

        # Create profile loading
        profile_loading = ProfileLoading(self, dll_settings)

        # Calculate crest level
        hbn = []
        for _q, _h, _hs, _tspec, _dir in zip(
            q_overtopping,
            water_level,
            significant_wave_height,
            spectral_wave_period,
            wave_direction,
        ):
            hbn.append(profile_loading.calculate_crest_level(_q, _h, _hs, _tspec, _dir))

        # Return
        return hbn[0] if len(hbn) == 1 else np.array(hbn)

    def transform_wave_conditions(
        self,
        water_level: Union[float, list],
        significant_wave_height: Union[float, list],
        spectral_wave_period: Union[float, list],
        wave_direction: Union[float, list],
        tp_tspec: float = 1.1,
        force_array: bool = False,
    ) -> np.ndarray:
        """
        Transform the wave conditions for the schematized foreshore

        Parameters
        ----------
        water_level : Union[float, list]
            Water level
        significant_wave_height : Union[float, list]
            Significant wave height
        spectral_wave_period : Union[float, list]
            Spectral wave period
        wave_direction : Union[float, list]
            Wave direction
        tp_tspec : float, optional
            Ratio between Tp and Tspec, only used for dam and foreshore (default : 1.1)
        force_array : bool, optional
            Always force to return an array

        Returns
        -------
        np.ndarray
            Water level and transformed wave conditions (h, hs, tp, dir)
        """
        # Length of the lists
        n_h = 1 if isinstance(water_level, (float, int)) else len(water_level)
        n_hs = 1 if isinstance(significant_wave_height, (float, int)) else len(significant_wave_height)
        n_tspec = 1 if isinstance(spectral_wave_period, (float, int)) else len(spectral_wave_period)
        n_dir = 1 if isinstance(wave_direction, (float, int)) else len(wave_direction)
        n_max = np.max([n_h, n_hs, n_tspec, n_dir])

        # Make lists
        if isinstance(water_level, (float, int)):
            water_level = [water_level] * n_max
        elif len(water_level) != n_max:
            raise ValueError("[ERROR] Uneven length of arrays")

        if isinstance(significant_wave_height, (float, int)):
            significant_wave_height = [significant_wave_height] * n_max
        elif len(significant_wave_height) != n_max:
            raise ValueError("[ERROR] Uneven length of arrays")

        if isinstance(spectral_wave_period, (float, int)):
            spectral_wave_period = [spectral_wave_period] * n_max
        elif len(spectral_wave_period) != n_max:
            raise ValueError("[ERROR] Uneven length of arrays")

        if isinstance(wave_direction, (float, int)):
            wave_direction = [wave_direction] * n_max
        elif len(wave_direction) != n_max:
            raise ValueError("[ERROR] Uneven length of arrays")

        # To numpy
        water_level = np.array(water_level)
        significant_wave_height = np.array(significant_wave_height)
        spectral_wave_period = np.array(spectral_wave_period)
        wave_direction = np.array(wave_direction)

        # Correct for foreshore
        if self.has_foreshore():
            fl = Foreshore(self)
            water_level, significant_wave_height, peak_wave_period, wave_direction = fl.transform_wave_conditions(
                water_level,
                significant_wave_height,
                spectral_wave_period * tp_tspec,
                wave_direction,
            )
            spectral_wave_period = peak_wave_period / tp_tspec

        # Return array or floats?
        if len(water_level.ravel()) == 1 and not force_array:
            return np.array(
                [
                    water_level.ravel()[0],
                    significant_wave_height.ravel()[0],
                    spectral_wave_period.ravel()[0],
                    wave_direction.ravel()[0],
                ]
            )
        else:
            return np.array(
                [
                    water_level,
                    significant_wave_height,
                    spectral_wave_period,
                    wave_direction,
                ]
            )

    def to_prfl(self, export_path: str, id: str = "Onbekend000", memo: str = "") -> None:
        """
        Export to a prfl file

        export_path : str
            Path to where the profile has to be exported
        id : str
            Id used in RisKeer
        memo : str
            Memo to be added to the prfl file
        """
        # Check required info
        if (self.dike_x is None) or (self.dike_crest_level is None) or (self.dike_orientation is None):
            raise ValueError(f"[ERROR] Cannot generate .prfl for profile '{self.profile_name}', geometry, crest level or orientation is missing.")

        # Version
        breakwater_level = self.breakwater_level if self.breakwater_level is not None else 0.0
        n_foreshore = len(self.foreshore_x) if self.foreshore_x is not None else 0
        n_dike = len(self.dike_x) if self.dike_x is not None else 0
        export = f"VERSIE 4.0\nID {id}\n\nRICHTING {self.dike_orientation}\n\nDAM {int(self.breakwater_type.value)}\nDAMHOOGTE {breakwater_level}\n\nVOORLAND {n_foreshore}\n[FORELAND]\nDAMWAND 0\nKRUINHOOGTE {self.dike_crest_level}\nDIJK {n_dike}\n[DIKE]\nMEMO\nGenerated with Pydra for profile '{self.profile_name}'\n{memo}\n"

        # Foreshore
        foreshore = ""
        if n_foreshore > 0:
            for foreshore_x, foreshore_y in zip(self.foreshore_x, self.foreshore_y):
                foreshore = foreshore + f"{round(foreshore_x,3):.3f}\t{round(foreshore_y,3):.3f}\t1.000\n"
        export = export.replace("[FORELAND]", foreshore)

        # Dike
        dike = ""
        if n_dike > 0:
            for dike_x, dike_y, dike_r in zip(self.dike_x, self.dike_y, self.dike_roughness):
                dike = dike + f"{round(dike_x,3):.3f}\t{round(dike_y,3):.3f}\t{round(dike_r,3):.3f}\n"
        export = export.replace("[DIKE]", dike)

        # Export .prfl
        if not export_path.lower().endswith(".prfl"):
            export_path = export_path + ".prfl"
        prfl_file = open(export_path, "w")
        prfl_file.write(export)
        prfl_file.close()

    def to_dict(self) -> dict:
        """
        Export Profile to dictionary

        Returns
        -------
        dictionary
            Dictionary with all profile settings
        """
        # Create an empty dictionary
        profile = {}

        # Add all settings to the dictionary
        for setting in dir(self):
            if not str(setting).startswith("__") and not callable(getattr(self, setting)):
                profile[setting] = getattr(self, setting)

        return profile

    def draw_profile(self) -> None:
        """
        Draw a cross-section of the profile
        """
        # Init plot
        plt.figure(figsize=[8, 5])

        # Foreshore
        if self.foreshore_x is not None:
            plt.plot(
                self.foreshore_x,
                self.foreshore_y,
                color="orange",
                label="Foreshore",
                zorder=1,
            )

        # Geometry
        for i in range(len(self.dike_x) - 1):
            r = self.dike_roughness[i]
            plt.plot(
                self.dike_x[i : i + 2],
                self.dike_y[i : i + 2],
                color="black" if r != 1.0 else "green",
                label=f"Roughness: {r}",
                zorder=1,
            )

        # Breakwater
        if isinstance(self.breakwater_type, Breakwater):
            # Where should we draw the breakwater?
            if self.foreshore_x is not None:
                x_orig, y_orig = (
                    self.foreshore_x[0],
                    self.foreshore_y[0],
                )
            else:
                x_orig, y_orig = self.dike_x[0], self.dike_y[0]

            # Draw
            bw_height = self.breakwater_level
            if self.breakwater_type == Breakwater.CAISSON:
                plt.fill_between(
                    [x_orig - 10, x_orig],
                    [bw_height, bw_height],
                    [y_orig, y_orig],
                    color="dimgrey",
                    label="Caisson",
                    zorder=2,
                )
            elif self.breakwater_type == Breakwater.VERTICAL_WALL:
                plt.plot(
                    [x_orig, x_orig],
                    [y_orig, bw_height],
                    color="black",
                    lw=3,
                    label="Vertical wall",
                    zorder=2,
                )
            elif self.breakwater_type == Breakwater.RUBBLE_MOUND:
                diff = bw_height - y_orig
                plt.fill_between(
                    [
                        x_orig - (3 * diff) - 2,
                        x_orig - (1.5 * diff) - 2,
                        x_orig - (1.5 * diff),
                        x_orig,
                    ],
                    [y_orig, bw_height, bw_height, y_orig],
                    [y_orig, y_orig, y_orig, y_orig],
                    color="grey",
                    label="Rubble mound",
                    zorder=2,
                )

        # Crest height
        plt.axhline(
            self.dike_crest_level,
            linestyle=":",
            color="grey",
            label="Crest height",
            zorder=1,
        )

        # Legend, labels, etc
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper left")
        plt.title(f"{self.profile_name} ({self.dike_orientation}°)", fontweight="bold")
        plt.xlabel("Distance [m]")
        plt.ylabel("Level [NAP+m]")
        plt.show()

    @classmethod
    def from_prfl(cls, prfl_path: str, profile_name: str = "Profile"):
        """
        Import a profile from a .prfl file

        Parameters
        ----------
        prfl_path : str
            Path to the '.prfl' file
        profile_name : str
            Name of the profile (default: 'Profile')

        Returns
        -------
        Profile
            Profile object
        """
        # Check if the file extension if .prfl
        if not prfl_path.lower().endswith(".prfl"):
            raise FileNotFoundError(f"[ERROR] Input file: {prfl_path} should be a .prfl file.")

        # Check if the provided path exists
        if not Path(prfl_path).exists():
            raise FileNotFoundError(f"[ERROR] Input file: {prfl_path} not found.")

        # Read the file
        with open(prfl_path, "r") as file:
            prfl = file.read()
            prfl = prfl.replace("\t", " ")

        # Split the SQL commands by a line break (\n) and remove empty lines, stop before 'memo'
        prfl = [entry.lower().strip() for entry in prfl.split("\n") if entry.strip()]
        prfl = prfl[: prfl.index("memo")]

        # Create a new profile
        profile = cls(profile_name)

        # Version
        version = float([entry for entry in prfl if "versie" in entry][0].split(" ")[1])
        if version != 4.0:
            raise NotImplementedError(f"[ERROR] Prfl version {version} is not supported.")

        # Sheet pile
        sheetpile = True if float([entry for entry in prfl if "damwand" in entry][0].split(" ")[1]) == 1.0 else False
        if sheetpile:
            raise NotImplementedError("[ERROR] Sheet piles are not implemented.")

        # Breakwater
        breakwater = Breakwater(int([entry for entry in prfl if "dam" in entry][0].split(" ")[1]))
        breakwater_level = float([entry for entry in prfl if "damhoogte" in entry][0].split(" ")[1])
        profile.set_breakwater(breakwater, breakwater_level)

        # Foreshore
        n_foreshore = int([entry for entry in prfl if "voorland" in entry][0].split(" ")[1])
        idx_foreshore = prfl.index(f"voorland {n_foreshore}") + 1
        foreshore_x = []
        foreshore_y = []
        for row in range(idx_foreshore, idx_foreshore + n_foreshore):
            foreshore_x.extend([float(prfl[row].split(" ")[0])])
            foreshore_y.extend([float(prfl[row].split(" ")[1])])
        if n_foreshore > 0:
            profile.set_foreshore_geometry(foreshore_x, foreshore_y)

        # Dike
        n_dike = int([entry for entry in prfl if "dijk" in entry][0].split(" ")[1])
        idx_dike = prfl.index(f"dijk {n_dike}") + 1
        dike_x = []
        dike_y = []
        dike_r = []
        for row in range(idx_dike, idx_dike + n_dike):
            dike_x.extend([float(prfl[row].split(" ")[0])])
            dike_y.extend([float(prfl[row].split(" ")[1])])
            dike_r.extend([float(prfl[row].split(" ")[2])])
        profile.set_dike_geometry(dike_x, dike_y, dike_r)

        # Crest height
        dike_crest_level = float([entry for entry in prfl if "kruinhoogte" in entry][0].split(" ")[1])
        profile.set_dike_crest_level(dike_crest_level)

        # Dike orientation
        dike_orientation = float([entry for entry in prfl if "richting" in entry][0].split(" ")[1])
        profile.set_dike_orientation(dike_orientation)

        # Return the class
        return profile

    @classmethod
    def from_dictionary(cls, dictionary: dict):
        """
        Create a profile from a dictionary

        Parameters
        ----------
        dictionary: dict
            Dictionary with all profile settings
        """
        # Create a new profile
        profile = cls(dictionary["profile_name"])

        # Add settings from dictionary to profile
        for setting in dictionary.keys():
            setattr(profile, setting, dictionary[setting])

        # Validate
        profile.validate_profile()

        return profile

    @classmethod
    def from_gebugekb_tool(cls, sql_path: str, profile_name: str = "Profile", berm_slope: float = 1 / 100):
        """
        Import a profile from the GEBUGEKB plugin.

        Parameters
        ----------
        sql_path : str
            Path to the '1.sql' file
        profile_name : str
            Name of the profile (default: 'Profile')
        berm_slope : float
            If applicable: slope of the berm (default : 1/100)

        Returns
        -------
        Profile
            Profile object
        """
        # Check if the file extension if .prfl
        if not sql_path.lower().endswith(".sql"):
            raise FileNotFoundError(f"[ERROR] Input file: {sql_path} should be a .sql file.")

        # Check if the provided path exists
        if not Path(sql_path).exists():
            raise FileNotFoundError(f"[ERROR] Input file: {sql_path} not found.")

        # Check if the berm slope is not too steep or shallow
        if berm_slope < 1 / 100 or berm_slope > 1 / 15:
            raise ValueError(f"[ERROR] The slope of the berm cannot be steeper than 1/15 or shallower than 1/100 (Given: 1/{1/berm_slope}).")

        # Read the file
        with open(sql_path, "r") as file:
            sql_commands = file.read()

        # Split the SQL commands by a line break (\n)
        commands = sql_commands.split("\n")

        # Remove all lines starting with 'DELETE FROM', '--' or is empty. Remove comments (by splitting at the semicolumn)
        commands = [entry.split(";")[0] for entry in commands if not entry.startswith("DELETE FROM") and not entry.startswith("--") and not entry == ""]

        # Create a new profile
        profile = cls(profile_name)

        # Breakwater
        breakwater = [entry for entry in commands if "Breakwaters" in entry]
        breakwater = np.array([ast.literal_eval(entry.split("VALUES ")[-1]) for entry in breakwater])
        if len(breakwater) > 1:
            raise ValueError("[ERROR] Multiple breakwaters are defined. Only one can be defined at a time.")
        elif len(breakwater) == 1:
            profile.set_breakwater(Breakwater(int(breakwater[0][1])), breakwater[0][2])

        # Foreshore
        foreshore = [entry for entry in commands if "Foreshores" in entry]
        foreshore = np.array([ast.literal_eval(entry.split("VALUES ")[-1]) for entry in foreshore])
        if len(foreshore) > 0:
            profile.set_foreshore_geometry(foreshore[:, 2], foreshore[:, 3])

        # Dike geometry
        dike = [entry.replace("NULL", "-999") for entry in commands if "VariableDatas" in entry]
        dike = np.array([ast.literal_eval(entry.split("VALUES ")[-1]) for entry in dike])[:, 4:]

        # Dike geometry
        slope_lower = dike[dike[:, 0] == 10][0][1]
        slope_upper = dike[dike[:, 0] == 11][0][1]
        toe_level = dike[dike[:, 0] == 12][0][1]
        berm_level = dike[dike[:, 0] == 13][0][1]
        berm_length = dike[dike[:, 0] == 14][0][1]
        crest_level = dike[dike[:, 0] == 15][0][1]
        dike_orientation = dike[dike[:, 0] == 16][0][1]
        roughness = dike[dike[:, 0] == 17][0][1]
        zone_roughness = dike[dike[:, 0] == 50][0][1]
        zone_ymin = dike[dike[:, 0] == 51][0][1]
        zone_ymax = dike[dike[:, 0] == 52][0][1]

        # Calculate coordinates
        dike_x = [0]
        dike_y = [toe_level]

        # If there is no berm
        if berm_length <= 0:
            dike_x.extend([(berm_level - toe_level) / slope_lower])
            dike_y.extend([berm_level])
        else:
            berm_difference = berm_slope * berm_length
            dike_x.extend([(berm_level - 0.5 * berm_difference - toe_level) / slope_lower])
            dike_y.extend([berm_level - 0.5 * berm_difference])
            dike_x.extend([dike_x[-1] + berm_length])
            dike_y.extend([berm_level + 0.5 * berm_difference])

        # Crest
        dike_x.extend([dike_x[-1] + (crest_level - dike_y[-1]) / slope_upper])
        dike_y.extend([crest_level])

        # Roughness
        dike_r = np.ones([len(dike_x)]) * roughness
        dike_r[(dike_y >= zone_ymin) & (dike_y < zone_ymax)] = zone_roughness

        # Do we need to add an extra point for zone_ymin?
        if (zone_ymin >= np.min(dike_y)) and (zone_ymin not in dike_y):
            x_int = Interpolate.inextrp1d(zone_ymin, dike_y, dike_x)
            idx = np.searchsorted(dike_x, x_int)
            dike_x = np.insert(dike_x, idx, x_int)
            dike_y = np.insert(dike_y, idx, zone_ymin)
            dike_r = np.insert(dike_r, idx, zone_roughness)

        # Do we need to add an extra point for zone_ymax?
        if (zone_ymax <= np.max(dike_y)) and (zone_ymax not in dike_y):
            x_int = Interpolate.inextrp1d(zone_ymax, dike_y, dike_x)
            idx = np.searchsorted(dike_x, x_int)
            dike_x = np.insert(dike_x, idx, x_int)
            dike_y = np.insert(dike_y, idx, zone_ymax)
            dike_r = np.insert(dike_r, idx, roughness)

        # Dike orientation, crest and geometry
        profile.set_dike_orientation(dike_orientation)
        profile.set_dike_geometry(dike_x, dike_y, dike_r)
        profile.set_dike_crest_level(crest_level)

        # Return the class
        return profile
