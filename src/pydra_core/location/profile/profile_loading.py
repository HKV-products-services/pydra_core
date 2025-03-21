import numpy as np

from ctypes import (
    ARRAY,
    CDLL,
    POINTER,
    Structure,
    c_bool,
    c_char,
    c_double,
    c_int,
    byref,
    create_string_buffer,
)
from pathlib import Path


class ProfileLoading:
    """
    Profile loading object to calculate hbn, runup, overtopping and overflow
    discharge.
    """

    # Model factors
    MODEL_FACTORS = {
        "FactorDeterminationQbFn": 2.3,
        "FactorDeterminationQbFb": 4.3,
        "M_z2": 1.07,
        "Fshallow": 0.67778,
        "ComputedOvertopping": 1.0,
        "CriticalOvertopping": 1.0,
        "RelaxationFactor": 1.0,
        "ReductionFactorForeshore": 0.5,
    }

    def __init__(self, profile, settings: dict = None):
        """
        Constructor of the ProfileLoading object

        Parameters
        ----------
        profile : Profile
            The Profile object
        settings : dict
            Model uncertainty settings, optional (default = None)
            Parameters: FactorDeterminationQbFn, FactorDeterminationQbFb, M_z2,
            Fshallow, ComputedOvertopping, CriticalOvertopping,
            RelaxationFactor, ReductionFactorForeshore
        """
        # Update model factors if defined
        if isinstance(settings, dict):
            if all(k in list(self.MODEL_FACTORS.keys()) for k in list(settings.keys())):
                self.MODEL_FACTORS.update(settings)
            else:
                raise ValueError(
                    "[ERROR] Settings dictionary contains unknown keys (check uppercase?)."
                )

        # Path to the library
        lib_path = Path(__file__).resolve().parent / "lib"

        # Load RTO-libary (VTV v17.1.1.0) (used for runup / overtopping when the water level is below crest level)
        self.rto_library = CDLL(str(lib_path / "dllDikesOvertopping.dll"))

        # Load COO-library (used for overflow) (Note: DiKErnel does not include this .dll)
        self.coo_library = CDLL(str(lib_path / "CombOverloopOverslag64.dll"))

        # Modelfactors
        factors = np.array(list(self.MODEL_FACTORS.items()))[:, 1].astype(float)
        self.modelfactors_rto = (c_double * 8)(*factors)
        self.modelfactors_coo = c_double(factors[3])
        self.output = (c_double * 2)(0.0, 0.0)
        self.qo = c_double()
        self.niveau = c_double()
        self.load_rto = c_double * 4
        self.load_coo = OvertoppingLoad
        self.succes = c_bool()
        self.message = create_string_buffer(b"", 512)

        # Set profile
        self.profile = profile

    def set_profile(self, profile):
        """
        Set a profile

        Parameters
        ----------
        Profile : profile
            A Profile object
        """
        self.profile = profile

    def calculate_discharge(
        self,
        water_level: float,
        significant_wave_height: float,
        spectral_wave_period: float,
        wave_direction: float,
    ) -> float:
        """
        Calculate overtopping for loaded profile and given load combination.

        Parameters
        ----------
        water_level : float
            Water level
        significant_wave_height : float
            Significant wave height
        spectral_wave_period : float
            Spectral wave period
        wave_direction : float
            Wave direction

        Returns
        -------
        float
            The overtopping discharge
        """
        # If the waterlevel is lower or equal to the crest level, use rto
        if water_level <= self.profile.dike_crest_level:
            # Create instance of load structure and calculate the overtopping discharge
            load = self.load_rto(
                water_level,
                significant_wave_height,
                spectral_wave_period,
                wave_direction,
            )
            self.__calculate_discharge_rto(load)
            qov = self.output[0]

        # If the water is above the crestlevel use coo
        else:
            # Calculate overtopping at h = crest level
            load = self.load_rto(
                self.profile.dike_crest_level,
                significant_wave_height,
                spectral_wave_period,
                wave_direction,
            )
            self.__calculate_discharge_rto(load)
            qov_ot = self.output[0]

            # Calculate overflow
            load = self.load_coo(
                water_level,
                significant_wave_height,
                spectral_wave_period,
                wave_direction,
            )
            self.__calculate_discharge_coo(load)
            qov_ov = self.qo.value

            # Take the value from overflow only if it is larger than overtopping (to create a smooth function)
            if qov_ov > qov_ot:
                qov = qov_ov
            else:
                qov = qov_ot

        # Catch errors
        if not self.succes:
            raise ValueError(
                self.message.value.decode().strip()
                + f" (Load: h={water_level}, Hs={significant_wave_height}, Tm-1,0={spectral_wave_period}, wdir={wave_direction})"
            )

        # Return the overtopping discharge
        return qov

    def calculate_runup(
        self,
        water_level: float,
        significant_wave_height: float,
        spectral_wave_period: float,
        wave_direction: float,
    ) -> float:
        """
        Calculate run up (z2%) for loaded profile and given load combination.

        Parameters
        ----------
        water_level : float
            Water level
        significant_wave_height : float
            Significant wave height
        spectral_wave_period : float
            Spectral wave period
        wave_direction : float
            Wave direction

        Returns
        -------
        float
            The run up height (z2%)
        """
        # If the waterlevel is lower or equal to the crest level
        if water_level <= self.profile.dike_crest_level:
            # Calculate the run up
            load = self.load_rto(
                water_level,
                significant_wave_height,
                spectral_wave_period,
                wave_direction,
            )
            self.__calculate_discharge_rto(load)
            ru2p = self.output[1] + water_level

        # Otherwise give an error
        else:
            raise ValueError("Water level exceeds crest level.")

        # Catch errors
        if not self.succes:
            raise ValueError(
                self.message.value.decode().strip()
                + f" (Load: h={water_level}, Hs={significant_wave_height}, Tm-1,0={spectral_wave_period}, wdir={wave_direction})"
            )

        # Return the runup discharge
        return ru2p

    def calculate_crest_level(
        self,
        q_overtopping: float,
        water_level: float,
        significant_wave_height: float,
        spectral_wave_period: float,
        wave_direction: float,
    ) -> float:
        """
        Calculate the crest level for a discharge q_overtopping for loaded profile and given load combination.

        Parameters
        ----------
        q_overtopping : float
            Critical overtopping discharge
        water_level : float
            Water level
        significant_wave_height : float
            Significant wave height
        spectral_wave_period : float
            Spectral wave period
        wave_direction : float
            Wave direction

        Returns
        -------
        float
            The crest level
        """
        # Create instance of load structure
        self.qcr = c_double(q_overtopping)

        # If the waterlevel is lower or equal to the crest level, use rto
        if wave_direction > 360:
            if (wave_direction - 360) < 10e-4:
                wave_direction = 0
        load = self.load_rto(
            water_level, significant_wave_height, spectral_wave_period, wave_direction
        )
        self.__calculate_crest_level_rto(load)

        # Catch errors
        if not self.succes:
            raise ValueError(self.message.value.decode().strip())

        # If the calculated crest level is equal to the water level
        if self.niveau.value == water_level:
            load = self.load_coo(
                water_level,
                significant_wave_height,
                spectral_wave_period,
                wave_direction,
            )
            self.__calculate_crest_level_coo(load)

        # Catch errors
        if not self.succes:
            raise ValueError(self.message.value.decode().strip())

        return self.niveau.value

    def __calculate_discharge_rto(self, load):
        """
        Function to communicate with the dllDikesOvertopping.dll
        """
        self.arr_ctype = c_double * len(self.profile.dike_x_coordinates)
        self.__set_argtypes()
        self.rto_library.calculateQoJ(
            load,
            byref(self.arr_ctype(*self.profile.dike_x_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_y_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_roughness)),
            byref(c_double(self.profile.dike_orientation)),
            byref(c_int(len(self.profile.dike_x_coordinates))),
            byref(c_double(self.profile.dike_crest_level)),
            byref(self.modelfactors_rto),
            byref(self.output),
            byref(self.succes),
            byref(self.message),
        )

    def __calculate_crest_level_rto(self, load):
        """
        Function to communicate with the dllDikesOvertopping.dll
        """
        self.arr_ctype = c_double * len(self.profile.dike_x_coordinates)
        self.__set_argtypes()
        self.rto_library.omkeerVariantJ(
            load,
            byref(self.arr_ctype(*self.profile.dike_x_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_y_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_roughness)),
            byref(c_double(self.profile.dike_orientation)),
            byref(c_int(len(self.profile.dike_x_coordinates))),
            byref(self.qcr),
            byref(self.niveau),
            byref(self.modelfactors_rto),
            byref(self.output),
            byref(self.succes),
            byref(self.message),
        )

    def __calculate_discharge_coo(self, load):
        """
        Function to communicate with the CombOverloopOverslag64.dll
        """
        self.arr_ctype = c_double * len(self.profile.dike_x_coordinates)
        self.__set_argtypes()
        self.coo_library.CalculateDischarge(
            byref(c_double(self.profile.dike_orientation)),
            byref(c_int(len(self.profile.dike_x_coordinates))),
            byref(self.arr_ctype(*self.profile.dike_x_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_y_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_roughness)),
            load,
            byref(self.modelfactors_coo),
            byref(c_double(self.profile.dike_crest_level)),
            byref(self.qo),
            byref(self.succes),
            byref(self.message),
        )

    def __calculate_crest_level_coo(self, load):
        """
        Function to communicate with the CombOverloopOverslag64.dll
        """
        self.arr_ctype = c_double * len(self.profile.dike_x_coordinates)
        self.__set_argtypes()
        self.coo_library.CalculateHeight(
            byref(c_double(self.profile.dike_orientation)),
            byref(c_int(len(self.profile.dike_x_coordinates))),
            byref(self.arr_ctype(*self.profile.dike_x_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_y_coordinates)),
            byref(self.arr_ctype(*self.profile.dike_roughness)),
            load,
            byref(self.modelfactors_coo),
            byref(self.qcr),
            byref(self.niveau),
            byref(self.succes),
            byref(self.message),
        )

    def __set_argtypes(self):
        """
        Set argtypes for dll functions

        This function is called after the profile is loaded
        """
        argtypes = {
            "load_rto": POINTER(c_double * 4),
            "load_coo": POINTER(OvertoppingLoad),
            "xp": POINTER(self.arr_ctype),
            "yp": POINTER(self.arr_ctype),
            "rp": POINTER(self.arr_ctype),
            "dike_orientation": POINTER(c_double),
            "debiet": POINTER(c_double),
            "npoints": POINTER(c_int),
            "crest_level": POINTER(c_double),
            "modelfactors_rto": POINTER(c_double * 8),
            "modelfactors_coo": POINTER(c_double),
            "output": POINTER(c_double * 2),
            "succes": POINTER(c_bool),
            "errormessage": POINTER(ARRAY(c_char, 512)),
        }

        # Overtopping and runup
        self.rto_library.calculateQoJ.argtypes = [
            argtypes[name]
            for name in [
                "load_rto",
                "xp",
                "yp",
                "rp",
                "dike_orientation",
                "npoints",
                "crest_level",
                "modelfactors_rto",
                "output",
                "succes",
                "errormessage",
            ]
        ]

        # HBN overtopping and runup
        self.rto_library.omkeerVariantJ.argtypes = [
            argtypes[name]
            for name in [
                "load_rto",
                "xp",
                "yp",
                "rp",
                "dike_orientation",
                "npoints",
                "debiet",
                "crest_level",
                "modelfactors_rto",
                "output",
                "succes",
                "errormessage",
            ]
        ]

        # Overflow
        self.coo_library.CalculateDischarge.argtypes = [
            argtypes[name]
            for name in [
                "dike_orientation",
                "npoints",
                "xp",
                "yp",
                "rp",
                "load_coo",
                "modelfactors_coo",
                "crest_level",
                "debiet",
                "succes",
                "errormessage",
            ]
        ]

        # HBN overflow
        self.coo_library.CalculateHeight.argtypes = [
            argtypes[name]
            for name in [
                "dike_orientation",
                "npoints",
                "xp",
                "yp",
                "rp",
                "load_coo",
                "modelfactors_coo",
                "debiet",
                "crest_level",
                "succes",
                "errormessage",
            ]
        ]


class OvertoppingLoad(Structure):
    """
    OvertoppingLoad class

    Contains crest level, wave height, wave period and wave direction.
    """

    _fields_ = [
        ("h", c_double),
        ("hm0", c_double),
        ("tm_10", c_double),
        ("phi", c_double),
    ]
