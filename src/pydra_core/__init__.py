__version__ = "0.0.13"
#############################################################
# pydra_core
# Contact: n.vandervegt@hkv.nl
#############################################################

import importlib

# The classes are imported on first use. A worker process of the HBN calculation imports only the profile part
# of pydra_core, which takes a fraction of the time of the whole package.
_EXPORTS = {
    # Core
    "ExceedanceFrequencyLine": ".core.exceedance_frequency_line",
    "HBN": ".core.hbn",
    # Enums
    "Breakwater": ".common.enum",
    "WaterSystem": ".common.enum",
    # Other
    "HRDatabase": ".hrdatabase.hrdatabase",
    "Location": ".location.location",
    "Profile": ".location.profile.profile",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """
    Import an exported class on first use and keep it in the module.

    Parameters
    ----------
    name : str
        The name of the class

    Returns
    -------
    type
        The class
    """
    if name not in _EXPORTS:
        raise AttributeError(f"[ERROR] module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    """
    List the names of the module, including the classes that are not imported yet.

    Returns
    -------
    list[str]
        The sorted names
    """
    return sorted(list(globals()) + __all__)
