import numpy as np

from .sea_level import SeaLevel
from .....settings.settings import Settings
from ......common.interpolate import Interpolate
from ......common.probability import ProbabilityFunctions
from ......io.file_hydranl import FileHydraNL


class SeaLevelLowerRiver(SeaLevel):
    """
    Class to describe the sea (water) level statistics.
    Sea level statistics are conditional on the wind direction.
    """

    def __init__(self, settings: Settings):
        """
        Initialise the sea level statistics for Rhine Tidal, Meuse Tidal,
        EuroPoort, Volkerak-Zoommeer and Hollandsche IJssel.

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit
        super().__init__()

        # Read the exceedance probability of the sea level conditional to the wind direction
        m, epm = FileHydraNL.read_file_ncolumns(settings.sea_level_probability)
        m += settings.sea_level_rise
        nr = epm.shape[1]

        # Correct for the distance between Hoek van Holland and Maasmond
        self.translation_m = 0.02 - settings.sea_level_rise

        # Equidistant filling vector with seawater levels
        self.m = ProbabilityFunctions.get_hnl_disc_array(
            settings.m_min, settings.m_max, settings.m_step
        )

        # Alloceer matrices
        self.nm = len(self.m)
        self.epm = np.zeros([self.nm, nr])

        # Interpolate the conditional exceedance probability of the sea level onto the given grid
        for ir in range(nr):
            iszero = epm[:, ir] == 0.0
            if iszero.any():
                zeroidx = np.where(iszero)[0][-1]

                # Interpoleer of extrapoleer logaritmisch tot de 0
                self.epm[:zeroidx, ir] = np.exp(
                    Interpolate.inextrp1d(
                        x=self.m[:zeroidx], xp=m[~iszero], fp=np.log(epm[~iszero, ir])
                    )
                )

                # Interpoleer of extrapoleer lineair vanaf 0
                self.epm[zeroidx:, ir] = np.maximum(
                    0.0, Interpolate.inextrp1d(self.m[zeroidx:], m, epm[:, ir])
                )

            else:
                self.epm[:, ir] = np.exp(
                    Interpolate.inextrp1d(x=self.m, xp=m, fp=np.log(epm[:, ir]))
                )

        # Limit the upper limit of the excedaance probability to 1.0
        self.epm = np.minimum(1.0, self.epm)

        # Scaling of the conditional exceedance probability to the used time period
        if (settings.block_duration != 12.0) and (settings.block_duration is not None):
            self.epm = 1.0 - (1.0 - self.epm) ** (settings.block_duration / 12.0)

        #  Allocate matrix
        self.pm = np.zeros((self.nm, nr))

        #  Berekenen kansdichtheid zeewaterstand gegeven de windrichting
        for ir in range(nr):
            self.pm[:, ir] = ProbabilityFunctions.probability_density(
                self.m, self.epm[:, ir]
            ).density

        #  Inlezen parameters kansverdeling gezamenlijke kansdichtheid zeewaterstand, windsnelheid
        #  en windrichting
        _, self.pwinds = FileHydraNL.read_file_ncolumns(
            settings.wind_sea_level_probability
        )
        self.pwinds = self.pwinds.T

        #  Geef twee instellingen van het gegevensblok mee in de structure self
        self.transitional_wind = settings.transitional_wind
