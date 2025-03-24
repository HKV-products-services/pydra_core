import fiona as fn
import numpy as np
import sqlite3

from pathlib import Path
from shapely.geometry import Point, shape

from ...common.common import CommonFunctions
from ...common.enum import WaterSystem
from ...io.database_settings import DatabaseSettings


class Settings:
    # Non Hydra-NL related
    watersystem = None

    # Hydra-NL settings (Hydra-NL naming convention in comments)
    # Location-specific settings
    location = None  # LOCATIE
    x_coordinate = None  # XCOORDINAAT
    y_coordinate = None  # YCOORDINAAT
    region = None  # REGIO
    river = None  # RIVIER
    sea = None  # ZEE
    lake = None  # MEER

    # Block duration
    block_duration = None  # BLOKDUUR
    periods_block_duration = None  # NPERIODENWHJ

    # Wind speed
    u_repair = None  # REPAREER_U
    u_max = None  # UMAX
    u_step = None  # USTAP

    # Discharge
    q_repair = None  # REPAREER_Q
    q_min = None  # QMIN
    q_max = None  # QMAX
    q_step = None  # QSTAP (UNUSED IN PYDRA)
    q_limit = None  # QAFTOP

    # Sea level
    m_repair = None  # REPAREER_M
    m_min = None  # MMIN
    m_max = None  # MMAX
    m_step = None  # MSTAP
    sea_level_rise = None  # ZWS_STIJGING

    # Lake level
    a_repair = None  # REPAREER_A
    a_min = None  # MMIN
    a_max = None  # MMAX
    a_step = None  # MSTAP
    a_limit = None  # MAFTOP
    lake_level_rise = None  # MEERPEIL_STIJGING

    # Europoort
    europoort_barrier = None  # KERING
    europoort_barrier_distribution = None  # VERDELING
    europoort_barrier_mu = None  # MU
    europoort_barrier_sigma = None  # SIGMA
    europoort_barrier_alfa = None  # ALFA

    # Wave form (discharge (q) and lake level (a))
    base_duration = None  # BASISDUUR
    periods_base_duration = None  # NGEGEVENSBLOK
    waveshape_pw = None  # PW
    waveshape_time_step = None  # DISCRSTAPAFVMP
    top_duration_q = None  # TDQNM
    ifh_q = None  # IFHQ
    ifb_q = None  # IFBQ
    top_duration_a = None  # TDMNM
    ifh_a = None  # IFHM
    ifb_a = None  # IFBM

    # Lakes
    transition_lake_wave_shape = None  # VERSCHUIVING
    sigma_aq = None  # SIGMA_MQ

    # Lower riviers (and VZM)
    transitional_wind = None  # TRANSWIND
    fu = None  # FU

    # IJssel-Vechtdelta
    failure_probability_ramspol = None  # FAALKANSKERING

    # Model uncertainty
    model_uncertainty_water_level_steps = 7  # WS_ONZ_AANTAL
    model_uncertainty_wave_height_steps = 5  # GH_ONZ_AANTAL
    model_uncertainty_wave_period_steps = 5  # GP_ONZ_AANTAL

    # File paths
    # Database
    database_path = None  # DBRVW

    # Wind
    wind_direction_probability = None  # KANSRNM
    wind_speed_probability = None  # OVKANSU

    # Discharge
    discharge_probability = None  # OVKANSQ

    # Lake level
    lake_level_probability = None  # OVKANSM

    # Barrier Eastern Scheldt
    barrier_scenario_probability = None  # KANSENFAALSCENARIO
    barrier_closing_probability = None  # KANSSLUITINGEN

    # Phase difference between storm surge and tide
    phase_surge_tide_probability = None  # KANSENFASEVERSCHIL

    # Sea level
    sea_level_probability = None  # OVKANSM
    sea_level_probability_point1 = None  # OVKANSM_HOEKPUNT1
    sea_level_probability_point2 = None  # OVKANSM_HOEKPUNT2
    sea_level_probability_point3 = None  # OVKANSM_HOEKPUNT3

    # Sigma function
    sigma_function = None  # SIGMAFUNCTIE

    # Storm surge duration
    storm_surge_duration_probability = None  # KANSENSTORMDUUR

    # Wind-Sea level probability (for lower rivers/VZM)
    wind_sea_level_probability = None  # PWINDWESTNM

    def __init__(self, hrdlocation: str, database_path: str):
        """
        Init a Settings object for a location and database

        Parameters
        ----------
        hrdlocation : str
            The HRDLocation
        database_path : str
            Path to the database
        """
        # Save info
        self.location = hrdlocation
        self.database_path = database_path

        # Obtain the x, y coordinates and the water system
        con = sqlite3.connect(database_path)
        self.x_coordinate = int(
            con.execute(
                f"SELECT XCoordinate FROM HRDLocations WHERE Name = '{hrdlocation}'"
            ).fetchone()[0]
        )
        self.y_coordinate = int(
            con.execute(
                f"SELECT YCoordinate FROM HRDLocations WHERE Name = '{hrdlocation}'"
            ).fetchone()[0]
        )
        self.watersystem = WaterSystem(
            con.execute("SELECT GeneralId FROM General").fetchone()[0]
        )
        con.close()

        # Watersystem specific settings
        if CommonFunctions.is_lower_rivier(self.watersystem):
            self.__determine_lower_river_settings()

        # Location specific settings
        if CommonFunctions.is_coast(self.watersystem):
            self.__determine_sea_level_statistics_points()

        # Obtain and set settings
        with DatabaseSettings() as database:
            settings = database.get_settings(self.watersystem)

        # Loop through settings
        for _, row in settings.iterrows():
            # Lowercase
            key = row["SettingName"]
            if isinstance(key, str):
                key = key.lower()

            # Set settings
            if hasattr(self, key):
                setattr(self, key, row["SettingValue"])
            else:
                raise ValueError(f"[ERROR] Key '{key}' not defined in Settings.")

        # Collect all settings with statistical uncertainty (with '_metOnzHeid')
        self.__vars_with_statistical_uncertainty = {}
        for variable in [
            var
            for var in dir(self)
            if not callable(getattr(self, var)) and not var.startswith("__")
        ]:
            value = getattr(self, variable)
            if "_metOnzHeid" in str(value):
                self.__vars_with_statistical_uncertainty[variable] = value

    def include_statistical_uncertainty(self, include: bool):
        """
        Include statistical uncertainty.
        By default statistical uncertainty is included.

        Parameters
        ----------
        include : bool
            Whether to include statistical uncertainty
        """
        # Loop through all variables with statistical uncertainty
        for setting in self.__vars_with_statistical_uncertainty.keys():
            value = self.__vars_with_statistical_uncertainty[setting]
            setattr(
                self, setting, value if include else value.replace("_metOnzHeid", "")
            )

    def __determine_sea_level_statistics_points(self):
        """
        The sea level statistics along the coast, Wadden Sea and Western
        Scheldt are based on triangulation interpolation. This function
        determines which stations should be used for this interpolation.
        """
        # Coast
        if self.watersystem in [
            WaterSystem.COAST_SOUTH,
            WaterSystem.COAST_CENTRAL,
            WaterSystem.COAST_NORTH,
        ]:
            # Obtain table
            with DatabaseSettings() as database:
                refpoints = database.get_sea_level_statistic_points(self.watersystem)
                subsystems = database.get_sea_level_sub_systems(self.watersystem)

            # Determine the subregion
            if CommonFunctions.position_from_line(
                "north",
                (self.x_coordinate, self.y_coordinate),
                [refpoints.loc["Den Oever"].values, refpoints.loc["Den Helder"].values],
            ):
                subregion = 1
            elif self.y_coordinate > refpoints.loc["IJmuiden"].values[1]:
                subregion = 2
            elif self.y_coordinate > refpoints.loc["Hoek van Holland"].values[1]:
                subregion = 3
            else:
                subregion = 4

            # Determine subsystem
            subsystem = subsystems.loc[subregion].to_numpy()

        # Wadden sea
        elif self.watersystem in [
            WaterSystem.WADDEN_SEA_WEST,
            WaterSystem.WADDEN_SEA_EAST,
        ]:
            # Obtain table
            with DatabaseSettings() as database:
                refpoints = database.get_sea_level_statistic_points(self.watersystem)
                subsystems = database.get_sea_level_sub_systems(self.watersystem)

            # Determine the subregion
            if CommonFunctions.position_from_line(
                "east",
                (self.x_coordinate, self.y_coordinate),
                [
                    refpoints.loc["Lauwersoog Haven"].values,
                    refpoints.loc["Huibertgat"].values,
                ],
            ):
                subregion = 1
            elif CommonFunctions.position_from_line(
                "north",
                (self.x_coordinate, self.y_coordinate),
                [
                    refpoints.loc["Lauwersoog Haven"].values,
                    refpoints.loc["West-Terschelling"].values,
                ],
            ):
                subregion = 2
            elif CommonFunctions.position_from_line(
                "east",
                (self.x_coordinate, self.y_coordinate),
                [
                    refpoints.loc["Harlingen"].values,
                    refpoints.loc["West-Terschelling"].values,
                ],
            ):
                subregion = 3
            elif CommonFunctions.position_from_line(
                "east",
                (self.x_coordinate, self.y_coordinate),
                [
                    refpoints.loc["Den Oever"].values,
                    refpoints.loc["West-Terschelling"].values,
                ],
            ):
                subregion = 4
            elif self.x_coordinate < refpoints.loc["Den Helder"].values[0]:
                subregion = 5
            else:
                subregion = 6

            # Determine subsystem
            subsystem = subsystems.loc[subregion].to_numpy()

        # Western scheldt
        elif self.watersystem in [WaterSystem.WESTERN_SCHELDT]:
            # Set sea level points
            subsystem = np.array(
                [
                    "Zeewaterstand/Vlissingen/CondPovVlissingen_12u_zichtjaar2017_metOnzHeid.txt",
                    "Zeewaterstand/Hansweert/CondPovHansweert_12u_zichtjaar2017_metOnzHeid.txt",
                    "Zeewaterstand/Vlissingen virtueel/CondPovVlissingen-Additional_12u_zichtjaar2017_metOnzHeid.txt",
                ]
            )

        # Not implemented
        else:
            raise NotImplementedError(
                f"[ERROR] Vertices for sea level statistics not implemented for {self.watersystem}."
            )

        # If the subsystems contains three times the same file, specify the parameter related to a point
        if len(list(set(subsystem))) == 1:
            self.sea_level_probability = subsystem[0]

        # Otherwise, set sea level triangulation points
        else:
            self.sea_level_probability_point1 = subsystem[0]
            self.sea_level_probability_point2 = subsystem[1]
            self.sea_level_probability_point3 = subsystem[2]

    def __determine_lower_river_settings(self):
        """
        Selecting the right settings for the lower rivers
        """
        # Read from shape (MSTAP, MU, SIGMA, ALFA, QSTAP_Maas, QSTAP_rijn)
        PATH = (
            Path(__file__).resolve().parent.parent
            / ".."
            / "data"
            / "settings"
            / "lower_river_settings.shp"
        ).resolve()

        with fn.open(PATH, "r") as shp:
            # Define the point using Shapely's Point
            point = Point(self.x_coordinate, self.y_coordinate)

            # Iterate through each subregion in the Shapefile
            for feature in shp:
                # Get the geometry of the feature
                geometry = shape(feature["geometry"])

                # Check if the point is within the geometry
                if point.within(geometry):
                    # Read settings
                    self.m_step = feature["properties"]["m_step"]
                    self.europoort_barrier_mu = feature["properties"]["ep_mu"]
                    self.europoort_barrier_sigma = feature["properties"]["ep_sigma"]
                    self.europoort_barrier_alfa = feature["properties"]["ep_alpha"]
                    if self.watersystem in [
                        WaterSystem.RHINE_TIDAL,
                        WaterSystem.EUROPOORT,
                    ]:
                        self.q_step = feature["properties"]["qs_rhine"]
                    elif self.watersystem == WaterSystem.MEUSE_TIDAL:
                        self.q_step = feature["properties"]["qs_meuse"]
                    else:
                        raise ValueError(
                            f"[ERROR] No lower river settings for {self.watersystem}."
                        )

                    # Stop the loop
                    break

    def __repr__(self):
        """
        Formatting when a settings object is printed. This will show all
        relevant settings with their values in a list.
        """
        lst = []
        for item in dir(self):
            if not item.startswith("_"):
                value = getattr(self, item)
                if value is None:
                    continue
                if isinstance(value, (str, float, int, bool)):
                    lst.append(f"{item:30s} = {value}")

        return "\n".join(lst)
