import numpy as np

from .wave_shape import WaveShape
from ....settings.settings import Settings
from .....common.enum import WaterSystem, WaveShapeType
from .....common.interpolate import Interpolate
from .....io.file_hydranl import FileHydraNL


class LakeLevel:
    """
    Class to describe the lake (water) level statistics.
    """

    def __init__(self, settings: Settings):
        """
        Constructor class for LakeLevel statistics.

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Read exceedance probability of peak lake level within one block (of base duration)
        apeak, epapeak = FileHydraNL.read_file_2columns(settings.lake_level_probability)

        # Adjust for lake level rise
        apeak += settings.lake_level_rise

        # Determine lowest lake level
        a_min_piek = max(apeak[0] - settings.lake_level_rise, settings.a_min)

        # Create grid for exceedance probability of peak lake level within one block (of base duration)
        self.apeak = np.r_[np.arange(a_min_piek, settings.a_max, settings.a_step), settings.a_max]
        self.napeak = len(self.apeak)

        # Inter/extrapolate the the exc. prob. of apeak on the new grid
        self.epapeak = np.exp(Interpolate.inextrp1d(x=self.apeak, xp=apeak, fp=np.log(epapeak)))
        self.epapeak[self.epapeak > 1] = 1.0

        # Create grid for the instateneous exceedancy probability P(Q > q)
        self.ablok = self.apeak.copy()
        self.nablok = len(self.ablok)

        # Initialise the wave shapes
        self.wave_shape = WaveShape(settings, type=WaveShapeType.LAKE_LEVEL)
        self.wave_shape.initialise_wave_shapes(self.apeak, climate_change=settings.lake_level_rise)

        # Calculate the instateneous exceedancy probability P(Q > q) (probability of Q exceeding q in one day)
        self.mom_ovkans_a = self.wave_shape.instantaneous_exceedance_probability(self.epapeak, self.ablok)

        # Clip exceedance probabilities
        self.epablok = np.clip(self.mom_ovkans_a, 0.0, 1.0)

        # Transition
        if (settings.transition_lake_wave_shape is not None) and (settings.transition_lake_wave_shape != 0.0):
            self.wave_shape.transition_wave(settings.transition_lake_wave_shape)

        # IJssel-Vechtdelta and VZM related
        if settings.watersystem in [WaterSystem.IJSSEL_DELTA, WaterSystem.VECHT_DELTA, WaterSystem.VOLKERAK_ZOOMMEER]:
            # Lokale parameters
            N = 376

            # Geef verwachtingswaarde en standaarddeviatie van de normale verdeling in de getransformeerde ruimte een waarde
            mu = 0.0
            sigma = settings.sigma_aq  # invoergeg.sigma_mq(ig)

            # Zet de gegevens voor de verfijnde interpolatietabel (de y-waarden, ofwel de K_sigma(s) uit vgl. 4.7 van [Geerse, 2003.129x]
            # Bereken voor elke y-waarde de integraal 4.3 [Geerse, 2003]
            f_y_k = np.linspace(-2 - 4 * sigma, 20 + 4 * sigma, N)

            # Verfijnde tabel klaarzetten van de cumulatieve verdelingsfunctie onder transformatie van s naar y
            # Zie ook vgl. 4.3 van [Geerse, 2003]

            # Maak de vector met y-waarden:
            # Bereken voor elke y-waarde de integraal 4.3 [Geerse, 2003]
            f_y_sigma = self.wave_shape.cum_norm_s_naar_y(f_y_k, mu, sigma)

            # Kansverdeling schalen of de hoogste ONDERschrijdingskans gelijk stellen aan 1
            if f_y_sigma[-1] > 1.0:
                f_y_sigma /= f_y_sigma[-1]
            else:
                f_y_sigma[-1] = 1.0

            for i in range(N):
                if f_y_sigma[i] > 0:
                    D = i
                    break

            # Een ONDERschrijdingskans gelijk aan 0.0 mag worden meegenomen.
            if D > 0:
                D -= 1

            self.k_apeak = Interpolate.inextrp1d(x=1 - self.epapeak, xp=f_y_sigma[D:], fp=f_y_k[D:])

    def __len__(self):
        """
        Return the number of discharge discretisations.

        Returns
        -------
        int
            Number of discretisations
        """
        return self.nablok

    def get_discretisation(self) -> np.ndarray:
        """
        Return the discharge discretisation.

        Returns
        -------
        np.ndarray
            1D array with discretisation
        """
        return self.ablok

    def get_exceedance_probability(self) -> np.ndarray:
        """
        Return exceedance probility of the discharge

        Returns
        -------
        np.ndarray
            A 1D array with the discharge exceedance probability
        """
        return self.epablok

    def get_wave_shape(self) -> WaveShape:
        """
        Return the wave form statistics object.

        Returns
        -------
        WaveShape
            The WaveShape statistics
        """
        return self.wave_shape
