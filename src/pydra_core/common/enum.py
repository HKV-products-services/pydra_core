from enum import Enum


class WaterSystem(Enum):
    """
    A class representing different water systems in the Netherlands.

    WaterSystem is an enumeration representing a specific water system in The
    Netherlands and is defined with a unique integer value (WBI2017 Appendix
    II). The enumeration members include both tidal and non-tidal water
    systems, such as rivers, lakes, seas, and coastal regions.
    """

    RHINE_NON_TIDAL = 1
    MEUSE_NON_TIDAL = 2
    RHINE_TIDAL = 3
    MEUSE_TIDAL = 4
    IJSSEL_DELTA = 5
    VECHT_DELTA = 6
    IJSSEL_LAKE = 7
    MARKER_LAKE = 8
    WADDEN_SEA_EAST = 9
    WADDEN_SEA_WEST = 10
    COAST_NORTH = 11
    COAST_CENTRAL = 12
    COAST_SOUTH = 13
    EASTERN_SCHELDT = 14
    WESTERN_SCHELDT = 15
    COAST_DUNES = 16
    EUROPOORT = 17
    MEUSE_VALLEY_NON_TIDAL = 18
    VELUWE_LAKES = 19
    GREVELINGEN = 20
    VOLKERAK_ZOOMMEER = 21
    HOLLANDSCHE_IJSSEL = 22
    DIEFDIJK = 23


class Breakwater(Enum):
    """
    Breakwater classes for profiles.

    The 'Breakwater' is an enumeration representing specific types of
    breakwaters located in front of the profile.
    """

    NO_BREAKWATER = 0
    CAISSON = 1
    VERTICAL_WALL = 2
    RUBBLE_MOUND = 3


class WaveShapeType(Enum):
    """
    Different wave form types.

    The 'WaveShapeType' is an enumeration representing specific types of wave
    forms used for slow stochastics. It can either be a discharge wave or a
    lake level wave.
    """

    DISCHARGE = 0
    LAKE_LEVEL = 1
