import numpy as np

from .sea_level import SeaLevel
from .....settings.settings import Settings
from ......common.interpolate import Interpolate
from ......common.probability import ProbabilityFunctions
from ......io.file_hydranl import FileHydraNL


class SeaLevelTriangular(SeaLevel):
    """
    Class to describe the sea (water) level statistics.
    Sea level statistics are conditional on the wind direction.
    """

    def __init__(self, settings: Settings, epsilon: float = 1.0e-15):
        """
        For Coast systems, we use triangular interpolation of the statistics
        between three points.

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

        #  Read the exceedance probability of the sea level for three stations (triangular interpolation)
        m1, epm1 = FileHydraNL.read_file_ncolumns(settings.sea_level_probability_point1)
        m2, epm2 = FileHydraNL.read_file_ncolumns(settings.sea_level_probability_point2)
        m3, epm3 = FileHydraNL.read_file_ncolumns(settings.sea_level_probability_point3)

        # Check whether the amount of wind directions for each point are equal
        if epm1.shape[1] != epm2.shape[1] or epm2.shape[1] != epm3.shape[1]:
            raise ValueError(
                "aantal windrichtingen per hoekpunt zijn niet aan elkaar gelijk"
            )

        # Obtain the x and y positions of the three stations
        x1, y1 = FileHydraNL.read_file_ncolumns_loc(
            settings.sea_level_probability_point1
        )
        x2, y2 = FileHydraNL.read_file_ncolumns_loc(
            settings.sea_level_probability_point2
        )
        x3, y3 = FileHydraNL.read_file_ncolumns_loc(
            settings.sea_level_probability_point3
        )

        # Check if the exceedance probability of the lowest sea level are all equal to 1
        if any(epm1[0, :] != 1) or any(epm2[0, :] != 1) or any(epm3[0, :] != 1):
            raise ValueError(
                "[ERROR] Exceedance probability of the lowest sea level is not equal to 1"
            )

        # Calculate the lowest sea level for the hr location
        m_min = Interpolate.triangular_interpolation(
            x1,
            y1,
            m1[0],
            x2,
            y2,
            m2[0],
            x3,
            y3,
            m3[0],
            settings.x_coordinate,
            settings.y_coordinate,
        )

        # Equidistant filling vector with seawater levels
        self.m = ProbabilityFunctions.get_hnl_disc_array(
            m_min, settings.m_max, settings.m_step
        )
        self.nm = len(self.m)

        # Number of wind directions
        nr = epm1.shape[1]

        # Create matrix
        self.epm = np.zeros([self.nm, nr])

        # Iterate over each wind direction
        for ir in range(0, nr):
            # For high sea water levels, the exceedance probability may be 0
            # Find the location where the highest sea level has the greatest exceedance probability no equal to 0.
            largest_ep = 0.0
            i = 1
            while largest_ep <= 0.0:
                corners = [(m1, epm1), (m2, epm2), (m3, epm3)]
                for _m, _epm in corners:
                    nlm = len(_m) - i
                    if _epm[nlm, ir] > largest_ep:
                        largest_ep = _epm[nlm, ir]
                        ir_epm = _epm[:nlm, ir]
                i += 1

            # Determine the sea level discretisation for the three stations
            ir_m1 = Interpolate.inextrp1d_log_probability(
                ir_epm, epm1[:, ir][::-1], m1[::-1]
            )
            ir_m2 = Interpolate.inextrp1d_log_probability(
                ir_epm, epm2[:, ir][::-1], m2[::-1]
            )
            ir_m3 = Interpolate.inextrp1d_log_probability(
                ir_epm, epm3[:, ir][::-1], m3[::-1]
            )

            # Use triangular interpolation to calculate the sea level at the specified exceedance probability at the given location.
            ir_m = Interpolate.triangular_interpolation(
                x1,
                y1,
                ir_m1,
                x2,
                y2,
                ir_m2,
                x3,
                y3,
                ir_m3,
                settings.x_coordinate,
                settings.y_coordinate,
            )

            # Als de triangulaire interpolatie wordt toegepast in het geval van extrapolatie
            # kan de vector _m een verloop hebben dat niet langer monotoon is. Dit wordt
            # gecorrigeerd door die stukjes van de vector _m weg te gooien.
            idx = np.diff(ir_m) > 0.0
            if not idx.all():
                pos = np.where(idx)[0]
                pos = np.unique(np.concatenate([pos, pos + 1]))
                ir_m = ir_m[pos]
                ir_epm = ir_epm[pos]

            # Interpoleer logaritmisch de conditionele overschrijdingskansen van de zeewaterstand gegeven de windrichting naar het gewenste rooster
            ir_epm_int = np.exp(Interpolate.inextrp1d(self.m, ir_m, np.log(ir_epm)))

            # Door extrapolatie kunnen overschrijdingskansen groter dan 1 ontstaan zijn. Deze worden gelijkgesteld aan 1.
            # Door extrapolatie kunnen overschrijdingskansen kleiner dan 0 ontstaan zijn. Deze worden gelijkgesteld aan 0.
            # Door extrapolatie kan de mimimale zeewaterstand toch een overschrijdingskans van kleiner dan 1
            # krijgen. Dit betreft dan een onbelangrijke richting. Hiervoor wordt gecorrigeerd dat de
            # overschrijdingskans van de laagste zeewaterstand toch gelijk is aan 1.0
            ir_epm_int[0] = 1.0
            ir_epm_int = np.clip(ir_epm_int, 0.0, 1.0)

            # Add the results to the exceedance probability matrix
            self.epm[:, ir] = ir_epm_int

        # Verwerk de zeespiegelstijging
        self.m = self.m + settings.sea_level_rise

        # Bereken de transformatietabel van de OVERschrijdingskansen naar exponentiële ruimte
        # voor het correlatiemodel van de zeewaterstand en de windsnelheid.
        # Zie ook vgl. 4.7 van [Geerse, 2012], merk op dat het daar ONDERschrijdingskansen betreft#
        self.epm_exp = -np.log(np.maximum(self.epm, epsilon))
