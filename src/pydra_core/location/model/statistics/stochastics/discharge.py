import numpy as np

from .wave_shape import WaveShape
from ....settings.settings import Settings
from .....common.enum import WaveShapeType
from .....common.interpolate import Interpolate
from .....io.file_hydranl import FileHydraNL


class Discharge:
    """
    Class to describe the discharge statistics.
    """

    def __init__(self, settings: Settings):
        """
        Constructor for discharge statistics

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Read exceedance probability peak discharge
        qpeak, epqeak = FileHydraNL.read_file_2columns(settings.discharge_probability)

        # Create grid for peak discharges
        self.qpeak = np.r_[
            np.arange(settings.q_min, settings.q_max, settings.q_step), settings.q_max
        ]
        self.nqpeak = len(self.qpeak)

        # Inter/extrapolate the exceedance probability of the discharge
        self.epqpeak = np.exp(
            Interpolate.inextrp1d(x=self.qpeak, xp=qpeak, fp=np.log(epqeak))
        )

        #  Transformatie van de tabel met OVERschrijdingskansen naar exponentiële ruimte
        #  Zie ook vgl. 4.5 van [Geerse, 2003], merk op dat het daar ONDERschrijdingskansen betreft#
        self.eqpeak_exp = -np.log(self.epqpeak)

        # Init the wave forms
        self.wave_shape = WaveShape(settings, type=WaveShapeType.DISCHARGE)
        self.wave_shape.initialise_wave_shapes(self.qpeak)

        # Create grid with blok discharges
        if settings.q_min < np.min(self.qpeak):
            step = self.qpeak[1] - self.qpeak[0]
            self.qblok = np.r_[
                np.arange(settings.q_min, np.min(self.qpeak), step), self.qpiek
            ]
        elif settings.q_min > np.min(self.qpeak):
            raise ValueError("[ERROR] Q_min is larger than the lowest Q_peak.")
        else:
            self.qblok = self.qpeak
        self.nqblok = len(self.qblok)

        # Calculate the instantaneous exceedance probability
        inst_epq = self.wave_shape.instantaneous_exceedance_probability(
            self.epqpeak, self.qblok
        )

        # Make sure the exceedance probability is not larger than 1 or smaller than 0
        self.upqblok = np.clip(1.0 - inst_epq, 0.0, 1.0)

    def __len__(self):
        """
        Return the number of discharge discretisations.

        Returns
        -------
        int
            Number of discretisations
        """
        return self.nqblok

    def get_discretisation(self) -> np.ndarray:
        """
        Return the discharge discretisation.

        Returns
        -------
        np.ndarray
            1D array with discretisation
        """
        return self.qblok

    def get_exceedance_probability(self) -> np.ndarray:
        """
        Return exceedance probility of the discharge

        Returns
        -------
        np.ndarray
            A 1D array with the discharge exceedance probability
        """
        return self.upqblok

    def get_wave_shape(self) -> WaveShape:
        """
        Return the wave form statistics object.

        Returns
        -------
        WaveShape
            The WaveShape statistics
        """
        return self.wave_shape
