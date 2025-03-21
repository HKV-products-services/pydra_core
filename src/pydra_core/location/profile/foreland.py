import numpy as np

from ctypes import CDLL, POINTER, c_char_p, c_double, c_int, c_long, byref
from numpy.ctypeslib import ndpointer
from pathlib import Path
from typing import Tuple


class Foreland:
    """
    This module will use the Dam and Foreland module (DaF) to transform wave conditions
    based on the schematized foreshore. The DaF module can be used to transform wave conditions
    over a breakwater and/or foreshore.
    """

    def __init__(self, profile, log: bool = False):
        # Path to the library
        lib_path = Path(__file__).resolve().parent / "lib"

        # Load the DaF library v20.1.1.692 (differ from Hydra-NL, there are some slight changes)
        self.daf_library = CDLL(str(lib_path / "DynamicLib-DaF.dll"))

        # Default settings
        self.alpha_c = c_double(1.0)
        self.fc_c = c_double(0.021)
        self.ratiodepth_c = c_double(0.5)
        self.minstepsize_c = c_double(1.0)
        self.invalid_c = c_double(-999.99)
        self.logging_c = c_int(int(log))
        self.loggingfilename_c = c_char_p("dlldaf_log.txt".encode("utf-8"))
        self.loggingfilenamelength_c = c_int(len(self.loggingfilename_c.value))
        self.g_c = c_double(9.81)
        self.rho_c = c_double(1000.0)

        # Intialize dll entries
        self.__initialize_dll_entries()

        # Set profile
        self.profile = profile

    def add_profile(self, profile) -> None:
        """
        Set profile

        Parameters
        ----------
        Profile : profile
            A Profile object
        """
        self.profile = profile

    def transform_wave_conditions(
        self,
        water_level: np.ndarray,
        significant_wave_height: np.ndarray,
        peak_wave_period: np.ndarray,
        wave_direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the wave conditions for the schematized foreland

        Parameters
        ----------
        water_level : np.ndarray
            Water level
        significant_wave_height : np.ndarray
            Significant wave height
        peak_wave_period : np.ndarray
            Peak wave period
        wave_direction : np.ndarray
            Wave direction

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Water level and transformed wave conditions (h, hs, tp, dir)
        """
        # Get size and shape
        shp = significant_wave_height.shape

        # Determine part where wave height is larger than zero
        mask = (significant_wave_height > 0.0) & (peak_wave_period > 0.0)
        N = mask.sum()

        # Allocate output arrays as single dimension
        hm0dike = np.zeros(N, order="F")
        tpdike = np.zeros(N, order="F")
        refractedwaveangledike = np.zeros(N, order="F")
        message = "".ljust(1000).encode("utf-8")
        messagelength = c_int(1000)
        n_vl = (
            len(self.profile.foreland_x_coordinates)
            if self.profile.foreland_x_coordinates is not None
            else 1
        )
        x_vl = (
            np.array(self.profile.foreland_x_coordinates).astype(np.float64)
            if self.profile.foreland_x_coordinates is not None
            else np.array([0]).astype(np.float64)
        )
        y_vl = (
            np.array(self.profile.foreland_y_coordinates).astype(np.float64)
            if self.profile.foreland_x_coordinates is not None
            else np.array([-999]).astype(np.float64)
        )

        res = self.daf_library.C_FORTRANENTRY_RollerModel5(
            byref(c_int(self.profile.breakwater_type.value)),
            byref(c_double(self.profile.breakwater_level)),
            byref(self.alpha_c),
            byref(self.fc_c),
            byref(self.invalid_c),
            byref(c_int(N)),
            significant_wave_height[mask].astype(np.float64),
            peak_wave_period[mask].astype(np.float64),
            water_level[mask].astype(np.float64),
            wave_direction[mask].astype(np.float64),
            byref(c_double(self.profile.dike_orientation)),
            byref(c_int(n_vl)),
            x_vl,
            byref(self.minstepsize_c),
            y_vl,
            byref(self.ratiodepth_c),
            byref(self.logging_c),
            self.loggingfilename_c,
            self.loggingfilenamelength_c,
            byref(self.g_c),
            byref(self.rho_c),
            hm0dike,
            tpdike,
            refractedwaveangledike,
            message,
            messagelength,
        )
        message = message.decode("utf-8")
        message = message.rstrip()
        messagelength = len(message)

        if res != 0:
            print(message + " - Using uncorrected wave parameters.")
            print(self.profile.foreland_x_coordinates)
            print(self.profile.foreland_y_coordinates)
            hm0dike[:] = significant_wave_height[mask].ravel()[:]
            tpdike[:] = peak_wave_period[mask].ravel()[:]
            refractedwaveangledike[:] = wave_direction[mask].ravel()[:]

        # If not all input conditions were non-zero, put the calculated conditions on the original grid again.
        if not mask.all():
            # Copy original values
            hm0_tmp, tp_tmp, wdir_tmp = (
                significant_wave_height.copy(),
                peak_wave_period.copy(),
                wave_direction.copy(),
            )

            # Insert calculated values
            hm0_tmp[mask] = hm0dike
            tp_tmp[mask] = tpdike
            wdir_tmp[mask] = refractedwaveangledike
            conditions = (water_level, hm0_tmp, tp_tmp, wdir_tmp % 360.0)

        # Else, all values, where calculated. Only reshape to input shape
        else:
            conditions = (
                water_level,
                hm0dike.reshape(shp),
                tpdike.reshape(shp),
                refractedwaveangledike.reshape(shp) % 360.0,
            )

        # Return the transformed conditions
        return conditions

    def __initialize_dll_entries(self) -> None:
        """
        Initializes the arguments types of various functions in the dynamic library.
        """
        # get entry point in dll, the function to use
        arraypointer = ndpointer(dtype="double", ndim=1, flags="F_CONTIGUOUS")

        # Define all the argument types
        argtypes = {
            "DamType": POINTER(c_int),
            "DamHeight": POINTER(c_double),
            "Alpha": POINTER(c_double),
            "Fc": POINTER(c_double),
            "Invalid": POINTER(c_double),
            "DimHm0": POINTER(c_int),
            "Hm0": arraypointer,
            "Tp": arraypointer,
            "Wlev": arraypointer,
            "IncomingWaveAngle": arraypointer,
            "DikeNormal": POINTER(c_double),
            "DimX": POINTER(c_int),
            "X": arraypointer,
            "MinStepSize": POINTER(c_double),
            "BottomLevel": arraypointer,
            "RatioDepth": POINTER(c_double),
            "Logging": POINTER(c_int),
            "LoggingFileName": c_char_p,
            "LoggingFileNameLength": c_int,
            "Ag": POINTER(c_double),
            "Rho": POINTER(c_double),
            "Hm0Dike": arraypointer,
            "TpDike": arraypointer,
            "RefractedWaveAngleDike": arraypointer,
            "Message": c_char_p,
            "MessageLength": c_int,
        }

        # Note function definition for DAF module ROLLERMODEL5
        self.daf_library.C_FORTRANENTRY_RollerModel5.restype = c_long
        self.daf_library.C_FORTRANENTRY_RollerModel5.argtypes = [
            argtypes[name]
            for name in [
                "DamType",
                "DamHeight",
                "Alpha",
                "Fc",
                "Invalid",
                "DimHm0",
                "Hm0",
                "Tp",
                "Wlev",
                "IncomingWaveAngle",
                "DikeNormal",
                "DimX",
                "X",
                "MinStepSize",
                "BottomLevel",
                "RatioDepth",
                "Logging",
                "LoggingFileName",
                "LoggingFileNameLength",
                "Ag",
                "Rho",
                "Hm0Dike",
                "TpDike",
                "RefractedWaveAngleDike",
                "Message",
                "MessageLength",
            ]
        ]

        # Note function definition for DAF function to obtain version number.
        self.daf_library.GetVersionInfo.argtypes = [c_char_p, c_int]
