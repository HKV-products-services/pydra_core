import numpy as np

from scipy.stats import norm

from .discrete_probability import DiscreteProbability
from .sigma_function import SigmaFunction
from ....settings.settings import Settings
from .....common.interpolate import Interpolate
from .....io.file_hydranl import FileHydraNL


class WindSpeed:
    """
    Class to describe the wind speed statistics.
    Wind speed statistics are conditional on the wind direction.
    """

    def __init__(self, settings: Settings):
        """
        Constructor class for the WindSpeed statistics.

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Default None, only initialized when correcting with sigma function
        self.k_u = None

        # Read wind speed
        self.__read_wind_speed(settings)

        # Scale for block_duration
        if (settings.block_duration != 12.0) and (settings.block_duration is not None):
            self.epu = 1.0 - (1.0 - self.epu) ** (settings.block_duration / 12.0)
        self.block_duration_wind = settings.block_duration

    def __len__(self):
        """
        Return the number of wind speed discretisations

        Returns
        -------
        int
            Number of discretisations
        """
        return self.nu

    def __read_wind_speed(self, settings: Settings) -> None:
        """
        Read and process the wind speed statistics

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Read the wind speed and exceedance probabilities
        lu, eplu = FileHydraNL.read_file_ncolumns(settings.wind_speed_probability)

        # If the u_max in the settings is larger than the maximum wind speed in the statistics
        if settings.u_max > max(lu):
            # Calculate the number of wind speeds to add
            nuhoger = int(round(max(1.0, settings.u_max - max(lu))))

            # Add the wind speeds with a step of 1
            u = np.concatenate(
                [lu, np.linspace(max(lu), settings.u_max, nuhoger + 1)[1:]]
            )
            self.epu = np.zeros((len(u), eplu.shape[1]))

            # Add exceedance probabilities (extrapolate)
            for i, col in enumerate(eplu.T):
                # If the last value is 0 or less, add zeroes
                if col[-1] <= 0.0:
                    self.epu[len(lu) :, i] = 0.0

                # Otherwise, extrapolate logarithmically
                else:
                    self.epu[:, i] = np.exp(
                        Interpolate.inextrp1d(x=u, xp=lu, fp=np.log(col))
                    )

        # Otherwise, do nothing
        else:
            u = lu[:]
            self.epu = eplu[:, :]

        # If a stepsize for the windspeed is defined
        if settings.u_step is not None:
            # Create a new grid
            ugrid = np.concatenate(
                [np.arange(0, settings.u_max, settings.u_step), [settings.u_max]]
            )

            # Interpolate the exceedance probabilities for this grid
            epu = []
            for col in self.epu.T:
                logep = col
                logep[logep > 0.0] = np.log(logep[logep > 0.0])
                epu.append(np.exp(Interpolate.inextrp1d(x=ugrid, xp=u, fp=logep)))

            # Vervang overschrijdingskansen en u
            self.epu = np.c_[epu].T
            u = ugrid

        # Add the results to the object
        self.u = u
        self.nu = len(u)

    def correct_with_sigma_function(
        self, sigma_function: SigmaFunction, wind_direction: DiscreteProbability
    ) -> None:
        """
        Correct with statistics with sigma function.

        Parameters
        ----------
        sigma_function : SigmaFunction
            The SigmaFunction statistics object
        """
        # Init a matrix
        self.k_u = np.ones((self.nu, len(wind_direction))) * -99.0

        # Per windrichting
        for ir in range(len(wind_direction)):
            # Als er correlatie is (sigma > 0)
            if sigma_function.correlation[ir]:
                lim1 = -4 * max(sigma_function.sigma[:, ir])
                lim2 = 20.0 + 4 * max(sigma_function.sigma[:, ir])
                f_y_m = np.linspace(lim1, lim2, 351)
                f_y_sigma = self.__wind_transformation(
                    f_y_m,
                    0.0,
                    sigma_function.sigma_sea_level,
                    sigma_function.sigma[:, ir],
                )

                # Kansverdeling schalen of de hoogste ONDERschrijdingskans gelijk stellen aan 1
                if f_y_sigma[-1] > 1.0:
                    f_y_sigma /= f_y_sigma[-1]
                else:
                    f_y_sigma[-1] = 1.0

                ondkansu = 1.0 - self.epu[:, ir]
                self.k_u[:, ir] = Interpolate.inextrp1d(
                    x=ondkansu, xp=f_y_sigma, fp=f_y_m
                )

    def __wind_transformation(self, y, mu, transzee, sigmafunctie):
        """
        Transform the wind speed statistics for correlation with the sea level.
        """
        if min(sigmafunctie) <= 0.0:
            raise ValueError()

        stap = 0.05
        x = np.arange(0, 20 + 0.1 * stap, stap)
        sigma = Interpolate.inextrp1d(x=x, xp=transzee, fp=sigmafunctie)

        if min(sigma) <= 0.0:
            raise ValueError()

        yn = (y[None, :] - x[:, None] - mu) / sigma[:, None]

        dx = np.diff(np.r_[x[0], (x[1:] + x[:-1]) / 2.0, x[-1]])

        f_y_sigma = (
            np.exp(-x[:, None]) * norm.cdf(x=yn, loc=0.0, scale=1.0) * dx[:, None]
        ).sum(0)

        return f_y_sigma

    def get_discretisation(self) -> np.ndarray:
        """
        Return the wind speed discretisation.

        Returns
        -------
        np.ndarray
            1D array with discretisation
        """
        return self.u

    def get_exceedance_probability(self) -> np.ndarray:
        """
        Return exceedance probility of the wind speed, conditional on the wind
        direction.

        Returns
        -------
        np.ndarray
            A 2D array with the wind speed exceedance probability conditional
            on the wind direction (r x u)
        """
        return self.epu
