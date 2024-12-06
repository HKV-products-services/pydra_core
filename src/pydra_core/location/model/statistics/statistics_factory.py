from .statistics import Statistics
from .water_systems.statistics_coast import StatisticsCoast
from .water_systems.statistics_eastern_scheldt import StatisticsEasternScheldt
from .water_systems.statistics_ijssel_vechtdelta import StatisticsIJsselVechtdelta
from .water_systems.statistics_lake import StatisticsLake
from .water_systems.statistics_lower_river import StatisticsLowerRiver
from .water_systems.statistics_upper_river import StatisticsUpperRiver
from ...settings.settings import Settings
from ....common.enum import WaterSystem


class StatisticsFactory:
    """
    A factory class to generate the right Statistics object for a given Settings object.

    Attributes
    ----------
    WATER_SYSTEM_STATISTICS_MAP : dict
        A dictionary containing the corresponding Statistics class for a WaterSystem
    """

    # Dictionary with Statistic classes for each WaterSystem
    WATER_SYSTEM_STATISTICS_MAP = {
        # Upper River
        WaterSystem.RHINE_NON_TIDAL: StatisticsUpperRiver,
        WaterSystem.MEUSE_NON_TIDAL: StatisticsUpperRiver,
        WaterSystem.MEUSE_VALLEY_NON_TIDAL: StatisticsUpperRiver,
        # Lower River
        WaterSystem.RHINE_TIDAL: StatisticsLowerRiver,
        WaterSystem.MEUSE_TIDAL: StatisticsLowerRiver,
        # TODO: WaterSystem.EUROPOORT
        # Coast
        WaterSystem.WADDEN_SEA_EAST: StatisticsCoast,
        WaterSystem.WADDEN_SEA_WEST: StatisticsCoast,
        WaterSystem.COAST_NORTH: StatisticsCoast,
        WaterSystem.COAST_CENTRAL: StatisticsCoast,
        WaterSystem.COAST_SOUTH: StatisticsCoast,
        WaterSystem.WESTERN_SCHELDT: StatisticsCoast,
        # Eastern Scheldt
        WaterSystem.EASTERN_SCHELDT: StatisticsEasternScheldt,
        # Lakes
        WaterSystem.IJSSEL_LAKE: StatisticsLake,
        WaterSystem.MARKER_LAKE: StatisticsLake,
        # TODO: WaterSystem.VELUWE_LAKES
        # TODO: WaterSystem.GREVELINGEN
        # IJssel-Vecht Delta
        WaterSystem.VECHT_DELTA: StatisticsIJsselVechtdelta,
        WaterSystem.IJSSEL_DELTA: StatisticsIJsselVechtdelta,
        # Volkerak-Zoommeer
        # TODO: WaterSystem.VOLKERAK_ZOOMMEER
        # Hollandsche IJssel
        # TODO: WaterSystem.HOLLANDSCHE_IJSSEL
        # Other
        # WaterSystem.COAST_DUNES
        # WaterSystem.DIEFDIJK
    }

    @staticmethod
    def get_statistics(settings: Settings) -> Statistics:
        """
        Returns the Statistics object for the specified water system.

        Parameters
        ----------
        settings : Settings
            A Settings object to get statistics for.

        Returns
        -------
        Statistics
            A Statistics object for the specified water system.

        Raises
        ------
        NotImplementedError
            If statistics for the specified water system are not implemented.
        """
        # Obtain the right Statistics class
        statistics_class = StatisticsFactory.WATER_SYSTEM_STATISTICS_MAP.get(
            settings.watersystem
        )

        # Return if the class is found, otherwise raise an error
        if statistics_class:
            return statistics_class(settings)
        else:
            raise NotImplementedError(
                f"[ERROR] Statistics for '{settings.watersystem.name}' not implemented (ID: {settings.watersystem.value})"
            )
