"""
Memory of calculations within one Python process, so that a second question on the same location is fast: the store
'crest_level_tables' for crest level tables and the store 'calculations' for the calculation objects of the ports in
'hydra/'. The labels, the sizes and how to switch the stores off are described in docs/hydra_extensions.qmd.
"""

import functools
import hashlib
import inspect
import math
import numbers
import numpy as np
import os
import threading

from collections import OrderedDict
from collections.abc import Callable, Hashable
from enum import Enum
from pathlib import Path
from typing import Any

from .crest_level_tables import CrestLevelTables
from ..location.profile.profile import Profile
from ..location.settings.settings import Settings


class Store:
    """
    A store of calculated values by label, with a maximum size; the entry used longest ago falls out.

    Attributes
    ----------
    enabled : bool
        Whether values are kept; when False, every request is calculated again and nothing is kept
    max_size : int
        The maximum number of kept values
    hits, misses : int
        The number of requests answered from the store and the number calculated, since the last 'clear'
    """

    def __init__(self, max_size: int):
        """
        Create an empty store.

        Parameters
        ----------
        max_size : int
            The maximum number of kept values
        """
        self.enabled = True
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """
        Return the number of kept values.

        Returns
        -------
        int
            The number of kept values
        """
        return len(self._entries)

    def clear(self) -> None:
        """
        Forget all kept values and reset the counters.
        """
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __contains__(self, label: Hashable) -> bool:
        """
        Determine whether a value is kept under a label.

        Parameters
        ----------
        label : Hashable
            The label

        Returns
        -------
        bool
            True if a value is kept under the label
        """
        with self._lock:
            return label in self._entries

    def put(self, label: Hashable, value: Any) -> None:
        """
        Keep a value calculated elsewhere under a label.

        Parameters
        ----------
        label : Hashable
            The label
        value : Any
            The value to keep
        """
        if not self.enabled or self.max_size < 1:
            return
        with self._lock:
            self._entries[label] = value
            self._entries.move_to_end(label)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def get_or_calculate(self, label: Hashable, calculate: Callable[[], Any]) -> Any:
        """
        Return the kept value for a label, or calculate and keep it. The calculation runs outside the lock, so two
        threads asking for the same label at the same time both calculate it.

        Parameters
        ----------
        label : Hashable
            The label
        calculate : Callable[[], Any]
            Calculates the value when none is kept

        Returns
        -------
        Any
            The kept or calculated value
        """
        if not self.enabled or self.max_size < 1:
            return calculate()
        with self._lock:
            if label in self._entries:
                self._entries.move_to_end(label)
                self.hits += 1
                return self._entries[label]
        value = calculate()
        with self._lock:
            self.misses += 1
        self.put(label, value)
        return value


crest_level_tables = Store(max_size=50_000)
calculations = Store(max_size=16)

# The number of worker processes that calculate crest level tables ahead (see 'calculate_ahead'); None is the number
# of cores minus 2, and 1 calculates every table in this process
processes: int | None = None


# Labels


# Crest level tables


class RememberedCrestLevelTable:
    """
    The function 'crest_level_table' with the store 'crest_level_tables', for a calculation with these settings:
    the same signature and the same result. The settings are labelled when it is made, as the caller passed them,
    before a port copies and changes them.
    """

    def __init__(self, settings: Settings):
        """
        Label the settings of the calculation.

        Parameters
        ----------
        settings : Settings
            The settings of the location, as the caller passed them
        """
        self._settings_label = Memory.label(settings)

    def __call__(self, profile, qcrit, h, hs, tspec, direction, factor_hs, factor_tspec) -> np.ndarray:
        """
        Return the kept crest level table for these arguments, or calculate and keep it.

        Parameters
        ----------
        profile : Profile
            The profile
        qcrit : float
            The critical discharge
        h, hs, tspec, direction : np.ndarray
            The water level, significant wave height, spectral wave period and wave direction
        factor_hs : float
            Factor for the significant wave height, used for model uncertainty
        factor_tspec : float
            Factor for the spectral wave period, used for model uncertainty

        Returns
        -------
        np.ndarray
            The crest levels, with the shape of 'h'; a kept table is read-only
        """
        if not crest_level_tables.enabled:
            return CrestLevelTables.crest_level_table(profile, qcrit, h, hs, tspec, direction, factor_hs, factor_tspec)
        table_label = self.table_label(Memory.label(profile), qcrit, h, hs, tspec, direction, factor_hs, factor_tspec)

        def calculate() -> np.ndarray:
            """
            Calculate the crest level table and make it read-only.

            Returns
            -------
            np.ndarray
                The crest levels, with the shape of 'h'
            """
            table = CrestLevelTables.crest_level_table(profile, qcrit, h, hs, tspec, direction, factor_hs, factor_tspec)
            table.setflags(write=False)
            return table

        return crest_level_tables.get_or_calculate(table_label, calculate)

    def table_label(self, profile_label, qcrit, h, hs, tspec, direction, factor_hs, factor_tspec) -> tuple:
        """
        Return the label of a table in 'crest_level_tables'.

        Parameters
        ----------
        profile_label : Hashable
            The label of the profile, 'Memory.label(profile)', made once by the caller
        qcrit : float
            The critical discharge
        h, hs, tspec, direction : np.ndarray
            The water level, significant wave height, spectral wave period and wave direction
        factor_hs : float
            Factor for the significant wave height, used for model uncertainty
        factor_tspec : float
            Factor for the spectral wave period, used for model uncertainty

        Returns
        -------
        tuple
            The label of the table
        """
        return (
            self._settings_label,
            profile_label,
            Memory.label(qcrit),
            Memory.label(h),
            Memory.label(hs),
            Memory.label(tspec),
            Memory.label(direction),
            Memory.label(factor_hs),
            Memory.label(factor_tspec),
        )


# Calculation objects


class RememberedCalculation(type):
    """
    Metaclass of the ports in 'hydra/': calling the class with the same arguments returns the kept object from the
    store 'calculations'. A subclass inherits it.
    """

    def __call__(cls, *args, **kwargs):
        """
        Return the kept object for these arguments, or build and keep it.

        Parameters
        ----------
        *args, **kwargs
            The arguments of the constructor

        Returns
        -------
        object
            The kept or new calculation object
        """
        signature = Memory.init_signature(cls)
        return Memory.remember(cls, super().__call__, signature, args, kwargs)


class Memory:
    """
    Keep and clear the calculation objects and crest level tables within the Python process.
    """

    @staticmethod
    def clear() -> None:
        """
        Forget everything kept in both stores.
        """
        crest_level_tables.clear()
        calculations.clear()

    @staticmethod
    def settings_items(settings: Settings) -> dict:
        """
        Return all values of the settings by attribute name: every attribute that is not a method, the private ones
        included.

        Parameters
        ----------
        settings : Settings
            The settings of the location

        Returns
        -------
        dict
            The value per attribute name
        """
        return {
            name: getattr(settings, name)
            for name in dir(settings)
            if not name.startswith("__") and not callable(getattr(settings, name))
        }

    @staticmethod
    def database_label(database_path: str | Path) -> tuple:
        """
        Label a database by its full path and the time it was last changed.

        Parameters
        ----------
        database_path : str | Path
            The path of the database

        Returns
        -------
        tuple
            The full path and the time of the last change in nanoseconds
        """
        return (str(Path(database_path).resolve()), os.stat(database_path).st_mtime_ns)

    @staticmethod
    def label(value: Any) -> Hashable:
        """
        Label an argument of a calculation: equal labels mean equal arguments. A type without a rule raises a
        'TypeError', so that a new kind of setting can not silently drop out of the label.

        Parameters
        ----------
        value : Any
            The argument

        Returns
        -------
        Hashable
            The label of the argument
        """
        from ..hrdatabase.hrdatabase import HRDatabase

        if value is None or isinstance(value, bool | str):
            return value
        if isinstance(value, Enum):
            return (type(value).__name__, value.value)
        if isinstance(value, numbers.Number):
            value = value.item() if isinstance(value, np.generic) else value
            # NaN is not equal to itself, so a label holding it would never be found again
            return "nan" if isinstance(value, float) and math.isnan(value) else value
        if isinstance(value, Path):
            return ("path", str(value))
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return ("array", value.shape, tuple(Memory.label(v) for v in value.ravel().tolist()))
            digest = hashlib.blake2b(np.ascontiguousarray(value).tobytes(), digest_size=16).digest()
            return ("array", value.dtype.str, value.shape, digest)
        if isinstance(value, list | tuple):
            return ("sequence", tuple(Memory.label(v) for v in value))
        if isinstance(value, set | frozenset):
            # Sorted by their text, so that the order of the set does not matter and mixed types can be sorted
            return ("set", tuple(sorted((Memory.label(v) for v in value), key=repr)))
        if isinstance(value, dict):
            return ("dict", tuple(sorted((str(k), Memory.label(v)) for k, v in value.items())))
        if isinstance(value, Settings):
            items = tuple(sorted((name, Memory.label(v)) for name, v in Memory.settings_items(value).items()))
            return ("Settings", items, Memory.database_label(value.database_path))
        if isinstance(value, Profile):
            items = tuple(
                sorted(
                    (name, Memory.label(getattr(value, name)))
                    for name in dir(value)
                    if not name.startswith("__") and not callable(getattr(value, name))
                )
            )
            return ("Profile", items)
        if isinstance(value, HRDatabase):
            return ("HRDatabase", Memory.database_label(value.database_path))
        raise TypeError(
            f"[ERROR] No label for a value of type {type(value).__name__}; add a rule for it to 'Memory.label'."
        )

    @staticmethod
    def remembered_crest_level_table(settings: Settings) -> RememberedCrestLevelTable:
        """
        Return the function 'crest_level_table' with the store 'crest_level_tables', for a calculation with these
        settings.

        Parameters
        ----------
        settings : Settings
            The settings of the location, as the caller passed them

        Returns
        -------
        RememberedCrestLevelTable
            The crest level function with the store
        """
        return RememberedCrestLevelTable(settings)

    @staticmethod
    def calculate_ahead(function: Callable, profile, qcrit: float, work: list[tuple[tuple, list[tuple]]]) -> None:
        """
        Calculate the crest level tables a calculation is going to ask 'function' for in worker processes, and keep them
        in 'crest_level_tables' under the label 'function' gives them, so that asking for them afterwards finds them.

        Only the tables not kept yet are calculated, and only from 'MIN_POINTS_PARALLEL' points together: below that,
        starting the workers takes longer than calculating the tables in this process. Nothing happens for a function of
        the caller (not a 'RememberedCrestLevelTable'), with the store switched off, or with 'processes' 1.

        Parameters
        ----------
        function : Callable
            The crest level function of the calculation
        profile : Profile
            The profile
        qcrit : float
            The critical discharge
        work : list[tuple[tuple, list[tuple]]]
            Per loading (h, hs, tspec, direction) the combinations (delta_h, factor_hs, factor_tspec); the table is that
            of 'function(profile, qcrit, h + delta_h, hs, tspec, direction, factor_hs, factor_tspec)'
        """
        if not isinstance(function, RememberedCrestLevelTable):
            return
        if not crest_level_tables.enabled or crest_level_tables.max_size < 1:
            return
        profile_label = Memory.label(profile)
        missing, labels = [], []
        for loading, combinations in work:
            h, hs, tspec, direction = loading
            todo = []
            for dh, factor_hs, factor_tspec in combinations:
                table_label = function.table_label(
                    profile_label, qcrit, h + dh, hs, tspec, direction, factor_hs, factor_tspec
                )
                if table_label not in crest_level_tables:
                    todo.append((dh, factor_hs, factor_tspec))
                    labels.append(table_label)
            if todo:
                missing.append((loading, todo))
        n_points = sum(loading[0].size * len(todo) for loading, todo in missing)
        n_processes = processes if processes is not None else CrestLevelTables.default_processes()
        if not CrestLevelTables.worth_starting_workers(n_points, n_processes, "set 'memory.processes = 1'"):
            return
        tables = CrestLevelTables.calculate_in_workers(profile, qcrit, missing, n_processes)
        for table_label, table in zip(labels, (t for per_loading in tables for t in per_loading), strict=True):
            table.setflags(write=False)
            crest_level_tables.put(table_label, table)

    @staticmethod
    def remember(target: Callable, calculate: Callable, signature: inspect.Signature, args: tuple, kwargs: dict) -> Any:
        """
        Return the kept result of 'calculate(*args, **kwargs)', or calculate and keep it, labelled with 'target' and all
        arguments with their defaults. A function among the arguments (a crest level function of the caller) can not
        be labelled: then the result is calculated and not kept.

        Parameters
        ----------
        target : Callable
            The class or function whose module and name label the result
        calculate : Callable
            Calculates the result
        signature : inspect.Signature
            The signature that binds the arguments
        args : tuple
            The positional arguments
        kwargs : dict
            The keyword arguments

        Returns
        -------
        Any
            The kept or calculated result
        """
        if not calculations.enabled:
            return calculate(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if any(callable(v) for v in bound.arguments.values()):
            return calculate(*args, **kwargs)
        calculation_label = (
            target.__module__,
            target.__qualname__,
            tuple((name, Memory.label(v)) for name, v in bound.arguments.items()),
        )
        return calculations.get_or_calculate(calculation_label, lambda: calculate(*args, **kwargs))

    @staticmethod
    @functools.cache
    def init_signature(cls: type) -> inspect.Signature:
        """
        Return the signature of '__init__' without 'self', so that it binds the arguments of a constructor call.

        Parameters
        ----------
        cls : type
            The class

        Returns
        -------
        inspect.Signature
            The signature of the constructor
        """
        signature = inspect.signature(cls.__init__)
        return signature.replace(parameters=list(signature.parameters.values())[1:])

    @staticmethod
    def remembered(function: Callable) -> Callable:
        """
        Decorate a calculation function: calling it with the same arguments returns the kept result from the store
        'calculations'.

        Parameters
        ----------
        function : Callable
            The calculation function

        Returns
        -------
        Callable
            The function with the store
        """
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            """
            Return the kept result of the function for these arguments, or calculate and keep it.

            Parameters
            ----------
            *args, **kwargs
                The arguments of the function

            Returns
            -------
            Any
                The kept or calculated result
            """
            return Memory.remember(function, function, signature, args, kwargs)

        return wrapper
