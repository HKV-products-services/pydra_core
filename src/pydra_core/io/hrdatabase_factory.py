from .hrdatabase_reader import HRDReader
from .water_systems.hrdatabase_wbi2017_dunes import HRDReaderWBI2017Dunes
from .water_systems.hrdatabase_wbi2017_generic import HRDReaderWBI2017
from .water_systems.hrdatabase_wbi2017_lower_river import HRDReaderWBI2017LowerRiver
from .water_systems.hrdatabase_wbi2023_easternscheldt import HRDReaderWBI2023EasternScheldt
from ..common.enum import WaterSystem
from ..location.settings.settings import Settings


class HRDReaderFactory:
    """
    A factory class to generate the right HRDReader object for a given Settings object.

    Attributes
    ----------
    WATER_SYSTEM_HRDREADER_MAP : dict
        A dictionary containing the corresponding HRDReader class for a WaterSystem
    """

    # Dictionary with HRDReader classes for each WaterSystem
    WATER_SYSTEM_HRDREADER_MAP = {
        # Upper River
        WaterSystem.RHINE_NON_TIDAL: HRDReaderWBI2017,
        WaterSystem.MEUSE_NON_TIDAL: HRDReaderWBI2017,
        WaterSystem.MEUSE_VALLEY_NON_TIDAL: HRDReaderWBI2017,
        # Lower River
        WaterSystem.RHINE_TIDAL: HRDReaderWBI2017LowerRiver,
        WaterSystem.MEUSE_TIDAL: HRDReaderWBI2017LowerRiver,
        # TODO: WaterSystem.EUROPOORT
        # Coast
        WaterSystem.WADDEN_SEA_EAST: HRDReaderWBI2017,
        WaterSystem.WADDEN_SEA_WEST: HRDReaderWBI2017,
        WaterSystem.COAST_NORTH: HRDReaderWBI2017,
        WaterSystem.COAST_CENTRAL: HRDReaderWBI2017,
        WaterSystem.COAST_SOUTH: HRDReaderWBI2017,
        WaterSystem.WESTERN_SCHELDT: HRDReaderWBI2017,
        # Eastern Scheldt
        WaterSystem.EASTERN_SCHELDT: HRDReaderWBI2023EasternScheldt,
        # Lakes
        WaterSystem.IJSSEL_LAKE: HRDReaderWBI2017,
        WaterSystem.MARKER_LAKE: HRDReaderWBI2017,
        WaterSystem.GREVELINGEN: HRDReaderWBI2017,
        WaterSystem.VELUWE_LAKES: HRDReaderWBI2017,
        # IJssel-Vecht Delta
        WaterSystem.VECHT_DELTA: HRDReaderWBI2017,
        WaterSystem.IJSSEL_DELTA: HRDReaderWBI2017,
        # Volkerak-Zoommeer
        # TODO: WaterSystem.VOLKERAK_ZOOMMEER
        # Hollandsche IJssel
        # TODO: WaterSystem.HOLLANDSCHE_IJSSEL
        # Other
        WaterSystem.COAST_DUNES: HRDReaderWBI2017Dunes,
        # WaterSystem.DIEFDIJK
    }

    @staticmethod
    def get_hrdreader(settings: Settings) -> HRDReader:
        """
        Returns the HRDReader object for the specified water system.

        Parameters
        ----------
        settings : Settings
            A Settings object to get HRDReader for.

        Returns
        -------
        HRDReader
            A HRDReader object for the specified water system.

        Raises
        ------
        NotImplementedError
            If HRDReader for the specified water system is not implemented.
        """
        # Obtain the right HRDReader class
        hrdreader_class = HRDReaderFactory.WATER_SYSTEM_HRDREADER_MAP.get(settings.watersystem)

        # Return if the class is found, otherwise raise an error
        if hrdreader_class:
            return hrdreader_class(settings.database_path)
        else:
            raise NotImplementedError(f"[ERROR] HRDatabase reader for '{settings.watersystem.name}' not implemented (ID: {settings.watersystem.value})")
