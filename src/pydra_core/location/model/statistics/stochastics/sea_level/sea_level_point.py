import numpy as np

from .sea_level import SeaLevel
from .....settings.settings import Settings
from ......common.interpolate import Interpolate
from ......common.probability import ProbabilityFunctions
from ......io.file_hydranl import FileHydraNL


class SeaLevelPoint(SeaLevel):
    """
    Class to describe the sea (water) level statistics.
    Sea level statistics are conditional on the wind direction.
    """

    def __init__(self, settings: Settings, epsilon: float = 1.0e-15):
        """
        For Eastern Scheldt systems, we base the SeaLevel statistics on one
        point.

        Parameters
        ----------
        settings : Settings
            The Settings object
        epsilon : float
            Maximum exceedance probability to determine the transformation
            table, used to take correlation into account.
        """
        # Inherit
        super().__init__()

        #  Lees de conditionele overschrijdingskansen van de zeewaterstand gegeven de windrichting
        m, epm = FileHydraNL.read_file_ncolumns(settings.sea_level_probability)
        nr = epm.shape[1]

        # Equidistant vullen van vector met zeewaterstanden
        settings.m_min = settings.m_min if settings.m_min is not None else m[0]
        self.m = ProbabilityFunctions.get_hnl_disc_array(
            settings.m_min, settings.m_max, settings.m_step
        )

        # Alloceer matrices
        self.nm = len(self.m)

        #  Interpoleer de conditionele overschrijdingskansen van de zeewaterstand gegeven de windrichting naar het gewenste rooster
        self.epm = np.zeros((self.nm, nr))
        for ir in range(nr):
            self.epm[:, ir] = np.maximum(
                0, np.exp(Interpolate.inextrp1d(self.m, m, np.log(epm[:, ir])))
            )

        # Adjust for sea level rise
        self.m = self.m + settings.sea_level_rise

        #  Berekenen kansdichtheid zeewaterstand gegeven de windrichting
        self.pm = np.zeros((self.nm, nr))
        for ir in range(nr):
            self.pm[:, ir] = ProbabilityFunctions.probability_density(
                self.m, self.epm[:, ir]
            ).density

        # Bereken de transformatietabel van de OVERschrijdingskansen naar exponentiële ruimte
        # voor het correlatiemodel van de zeewaterstand en de windsnelheid.
        # Zie ook vgl. 4.7 van [Geerse, 2012], merk op dat het daar ONDERschrijdingskansen betreft#
        self.epm_exp = -np.log(np.maximum(self.epm, epsilon))
