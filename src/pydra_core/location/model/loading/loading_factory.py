from .loading import Loading
from .water_systems.loading_coast import LoadingCoast
from .water_systems.loading_eastern_scheldt import LoadingEasternScheldt
from .water_systems.loading_ijssel_vechtdelta import LoadingIJsselVechtdelta
from .water_systems.loading_lake import LoadingLake
from .water_systems.loading_lower_rivier import LoadingLowerRiver
from .water_systems.loading_upper_river import LoadingUpperRiver
from ...settings.settings import Settings
from ....common.enum import WaterSystem


class LoadingFactory:
    """
    A factory class to generate the right Loading object for a given Settings object.

    Attributes
    ----------
    WATER_SYSTEM_LOADING_MAP : dict
        A dictionary containing the corresponding Loading class for a WaterSystem
    """

    # Dictionary with Loading classes for each WaterSystem
    WATER_SYSTEM_LOADING_MAP = {
        # Upper River
        WaterSystem.RHINE_NON_TIDAL: LoadingUpperRiver,
        WaterSystem.MEUSE_NON_TIDAL: LoadingUpperRiver,
        WaterSystem.MEUSE_VALLEY_NON_TIDAL: LoadingUpperRiver,
        # Lower River
        WaterSystem.RHINE_TIDAL: LoadingLowerRiver,
        WaterSystem.MEUSE_TIDAL: LoadingLowerRiver,
        # TODO: WaterSystem.EUROPOORT
        # Coast
        WaterSystem.WADDEN_SEA_EAST: LoadingCoast,
        WaterSystem.WADDEN_SEA_WEST: LoadingCoast,
        WaterSystem.COAST_NORTH: LoadingCoast,
        WaterSystem.COAST_CENTRAL: LoadingCoast,
        WaterSystem.COAST_SOUTH: LoadingCoast,
        WaterSystem.WESTERN_SCHELDT: LoadingCoast,
        # Eastern Scheldt
        WaterSystem.EASTERN_SCHELDT: LoadingEasternScheldt,
        # Lakes
        WaterSystem.IJSSEL_LAKE: LoadingLake,
        WaterSystem.MARKER_LAKE: LoadingLake,
        # TODO: WaterSystem.VELUWE_LAKES
        # TODO: WaterSystem.GREVELINGEN
        # IJssel-Vecht Delta
        WaterSystem.VECHT_DELTA: LoadingIJsselVechtdelta,
        WaterSystem.IJSSEL_DELTA: LoadingIJsselVechtdelta,
        # Volkerak-Zoommeer
        # TODO: WaterSystem.VOLKERAK_ZOOMMEER
        # Hollandsche IJssel
        # TODO: WaterSystem.HOLLANDSCHE_IJSSEL
        # Other
        # WaterSystem.COAST_DUNES
        # WaterSystem.DIEFDIJK
    }

    @staticmethod
    def get_loading(settings: Settings) -> Loading:
        """
        Returns the Loading object for the specified water system.

        Parameters
        ----------
        settings : Settings
            A Settings object to get the loading for.

        Returns
        -------
        Loading
            A Loading object for the specified water system.

        Raises
        ------
        NotImplementedError
            If loading for the specified water system are not implemented.
        """
        # Obtain the right Loading class
        loading_class = LoadingFactory.WATER_SYSTEM_LOADING_MAP.get(
            settings.watersystem
        )

        # Return if the class is found, otherwise raise an error
        if loading_class:
            return loading_class(settings)
        else:
            raise NotImplementedError(
                f"[ERROR] Loading for '{settings.watersystem.name}' not implemented (ID: {settings.watersystem.value})"
            )
