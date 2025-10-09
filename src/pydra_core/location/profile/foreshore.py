import numpy as np
import platform
from ctypes import (
    CDLL,
    POINTER,
    byref,
    c_char,
    c_double,
    c_int,
    c_long,
    create_string_buffer,
)
from numpy.ctypeslib import ndpointer
from pathlib import Path
from typing import Tuple


class Foreshore:
    """
    This module will use the Dam and Foreshore module (DaF) to transform wave conditions
    based on the schematized foreshore. The DaF module can be used to transform wave conditions
    over a breakwater and/or foreshore.
    """

    def __init__(self, profile, log: bool = False):
        # Path to the library
        lib_path = Path(__file__).resolve().parent / "lib"

        # Load the DaF library (differ from Hydra-NL, there are some slight changes)
        sys_pltfrm = platform.system()
        lib_path = Path(__file__).resolve().parent / "lib" / "DaF_25.1.1"
        if sys_pltfrm == "Windows":
            self._requires_string_lengths = True
            self.daf_library = CDLL(str(lib_path / "win64" / "DynamicLib-DaF.dll"))
            self.rm5 = getattr(self.daf_library, "C_FORTRANENTRY_RollerModel5")
        elif sys_pltfrm == "Linux":
            self._requires_string_lengths = False
            self.daf_library = CDLL(str(lib_path / "linux64" / "libDamAndForeshore.so"))
            self.rm5 = getattr(self.daf_library, "c_fortranentry_rollermodel5_")
        else:
            raise NotImplementedError(f"'{sys_pltfrm}' is not supported for DaF.")

        # Default settings
        self.alpha_c = c_double(1.0)
        self.fc_c = c_double(0.021)
        self.ratiodepth_c = c_double(0.5)
        self.minstepsize_c = c_double(1.0)
        self.invalid_c = c_double(-999.99)
        self.logging_c = c_int(int(log))
        self.loggingfilename_buffer = create_string_buffer(b"dlldaf_log.txt", 256)
        self.loggingfilename_c = self.loggingfilename_buffer
        self.loggingfilename_length = c_int(len(self.loggingfilename_buffer.value))
        self._message_buffer_size = 1000
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
        Transform the wave conditions for the schematized foreshore

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
        message_buffer = create_string_buffer(self._message_buffer_size)
        message_length = c_int(self._message_buffer_size)
        n_vl = len(self.profile.foreland_x_coordinates) if self.profile.foreland_x_coordinates is not None else 1
        x_vl = np.asfortranarray(self.profile.foreland_x_coordinates, dtype=np.float64) if self.profile.foreland_x_coordinates is not None else np.asfortranarray([0.0], dtype=np.float64)
        y_vl = np.asfortranarray(self.profile.foreland_y_coordinates, dtype=np.float64) if self.profile.foreland_x_coordinates is not None else np.asfortranarray([-999.0], dtype=np.float64)

        hm0_input = np.asfortranarray(significant_wave_height[mask], dtype=np.float64)
        tp_input = np.asfortranarray(peak_wave_period[mask], dtype=np.float64)
        water_level_input = np.asfortranarray(water_level[mask], dtype=np.float64)
        wave_direction_input = np.asfortranarray(wave_direction[mask], dtype=np.float64)
        breakwater_type = c_int(self.profile.breakwater_type.value)
        breakwater_level = c_double(self.profile.breakwater_level)
        dike_orientation = c_double(self.profile.dike_orientation)
        n_values = c_int(N)
        n_vl_c = c_int(n_vl)

        rm5_args = [
            byref(breakwater_type),
            byref(breakwater_level),
            byref(self.alpha_c),
            byref(self.fc_c),
            byref(self.invalid_c),
            byref(n_values),
            hm0_input,
            tp_input,
            water_level_input,
            wave_direction_input,
            byref(dike_orientation),
            byref(n_vl_c),
            x_vl,
            byref(self.minstepsize_c),
            y_vl,
            byref(self.ratiodepth_c),
            byref(self.logging_c),
            self.loggingfilename_c,
        ]
        if self._requires_string_lengths:
            rm5_args.append(self.loggingfilename_length)
        rm5_args.extend(
            [
                byref(self.g_c),
                byref(self.rho_c),
                hm0dike,
                tpdike,
                refractedwaveangledike,
                message_buffer,
            ]
        )
        if self._requires_string_lengths:
            rm5_args.append(message_length)

        res = self.rm5(*rm5_args)
        message = message_buffer.value.decode("utf-8", errors="ignore")
        message = message.rstrip()

        if res != 0:
            print(message + " - Using uncorrected wave parameters.")
            print(self.profile.foreshore_x)
            print(self.profile.foreshore_y)
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
            "LoggingFileName": POINTER(c_char),
            "Ag": POINTER(c_double),
            "Rho": POINTER(c_double),
            "Hm0Dike": arraypointer,
            "TpDike": arraypointer,
            "RefractedWaveAngleDike": arraypointer,
            "Message": POINTER(c_char),
        }
        if self._requires_string_lengths:
            argtypes["LoggingFileNameLength"] = c_int
            argtypes["MessageLength"] = c_int

        # Note function definition for DAF module ROLLERMODEL5
        self.rm5.restype = c_long
        argtype_order = [
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
        ]
        if self._requires_string_lengths:
            argtype_order.append("LoggingFileNameLength")
        argtype_order.extend(
            [
                "Ag",
                "Rho",
                "Hm0Dike",
                "TpDike",
                "RefractedWaveAngleDike",
                "Message",
            ]
        )
        if self._requires_string_lengths:
            argtype_order.append("MessageLength")

        self.rm5.argtypes = [argtypes[name] for name in argtype_order]
