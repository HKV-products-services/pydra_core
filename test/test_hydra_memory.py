"""
The memory of calculations ('core.memory'): every setting, the profile, the loading grid and the time the database was
last changed are in the label, so changing any of them gives a new crest level table and a new calculation object;
the stores can be switched off and keep at most 'max_size' entries; the ports use them by default; the tables
calculated ahead in worker processes are the same as those calculated in this process.
"""

import copy
import os
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase, Profile
from pydra_core.core import crest_level_tables, memory
from pydra_core.hydra.percentiles import HydraPercentiles
from pydra_core.hydra.rhine_meuse_delta_percentiles import HydraRhineMeuseDeltaPercentiles

DATA = Path(__file__).parent / "data"
DATABASE = DATA / "07_IJSSEL_LAKE" / "WBI2017_IJsselmeer_7-2_v02.sqlite"
QCRIT = 0.01

# A small loading grid: water level, significant wave height, spectral wave period and wave direction
GRID = (np.array([[1.0, 1.5]]), np.array([[0.5, 0.6]]), np.array([[3.0, 3.2]]), np.array([[225.0, 225.0]]))


class Calculation(metaclass=memory.RememberedCalculation):
    """A calculation object that only holds its arguments, to see when the store builds a new one."""

    def __init__(self, hrd, settings, statistical_uncertainty: bool = True):
        self.settings = settings
        self.statistical_uncertainty = statistical_uncertainty


@pytest.fixture(autouse=True)
def empty_memory():
    memory.Memory.clear()
    yield
    for store, max_size in [(memory.crest_level_tables, 50_000), (memory.calculations, 16)]:
        store.enabled = True
        store.max_size = max_size
    memory.Memory.clear()


@pytest.fixture(scope="module")
def hrd():
    return HRDatabase(DATABASE)


@pytest.fixture
def settings(hrd):
    return hrd.get_settings(hrd.get_location_names()[0])


def profile() -> Profile:
    return Profile("test", crest_level=10.0, orientation=225.0, cota_slope=3.0)


def table(settings, crest_profile: Profile | None = None, grid: tuple = GRID) -> np.ndarray:
    return memory.Memory.remembered_crest_level_table(settings)(crest_profile or profile(), QCRIT, *grid, 1.0, 1.0)


def changed_value(name: str, value, tmp_path: Path):
    """Another value for a setting, of the same kind."""
    if name == "database_path":
        # Another file with the same content: the label holds the full path
        copied = tmp_path / Path(value).name
        shutil.copyfile(value, copied)
        return str(copied)
    if value is None:
        return 1.0
    if isinstance(value, bool):
        return not value
    if isinstance(value, Enum):
        return next(member for member in type(value) if member is not value)
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, str):
        return value + "_changed"
    if isinstance(value, Path):
        return value / "changed"
    if isinstance(value, dict):
        return {**value, "changed": "changed"}
    raise TypeError(f"No changed value for setting {name!r} of type {type(value).__name__}.")


def test_every_setting_gives_new_table_and_calculation(hrd, settings, tmp_path):
    # The same settings again: the kept table and the kept object
    first_table = table(settings)
    first_calculation = Calculation(hrd, settings)
    assert table(copy.deepcopy(settings)) is first_table
    assert Calculation(hrd, copy.deepcopy(settings)) is first_calculation
    assert (memory.crest_level_tables.hits, memory.calculations.hits) == (1, 1)

    names = memory.Memory.settings_items(settings)
    assert len(names) > 60
    for name, value in names.items():
        changed = copy.deepcopy(settings)
        setattr(changed, name, changed_value(name, value, tmp_path))
        tables, calculations = memory.crest_level_tables.misses, memory.calculations.misses
        new_table = table(changed)
        new_calculation = Calculation(hrd, changed)
        assert memory.crest_level_tables.misses == tables + 1, name
        assert memory.calculations.misses == calculations + 1, name
        assert new_calculation is not first_calculation, name
        np.testing.assert_array_equal(new_table, first_table)


def test_changed_database_gives_new_table_and_calculation(tmp_path):
    copied = tmp_path / DATABASE.name
    shutil.copyfile(DATABASE, copied)
    hrd = HRDatabase(copied)
    settings = hrd.get_settings(hrd.get_location_names()[0])
    first_calculation = Calculation(hrd, settings)
    table(settings)

    stat = os.stat(copied)
    os.utime(copied, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert Calculation(hrd, settings) is not first_calculation
    table(settings)
    assert (memory.crest_level_tables.misses, memory.calculations.misses) == (2, 2)


def test_other_arguments_give_new_table_and_calculation(hrd, settings):
    table(settings)
    first_calculation = Calculation(hrd, settings)

    steeper = profile()
    steeper.set_dike_geometry(np.array([-20.0, 20.0]), np.array([-10.0, 10.0]))
    for crest_profile, grid in [
        (steeper, GRID),
        (profile(), (GRID[0] + 0.1, *GRID[1:])),
        (profile(), (GRID[0], GRID[1], GRID[2], GRID[3] - 10.0)),
    ]:
        table(settings, crest_profile, grid)
    assert memory.crest_level_tables.misses == 4

    assert Calculation(hrd, settings, statistical_uncertainty=False) is not first_calculation
    assert Calculation(hrd, settings, statistical_uncertainty=True) is first_calculation


def test_kept_table_is_read_only(settings):
    with pytest.raises(ValueError):
        table(settings)[0, 0] = 0.0


def test_switched_off_keeps_nothing(hrd, settings):
    memory.crest_level_tables.enabled = False
    memory.calculations.enabled = False
    assert table(settings) is not table(settings)
    assert Calculation(hrd, settings) is not Calculation(hrd, settings)
    assert len(memory.crest_level_tables) == len(memory.calculations) == 0


def test_least_recently_used_falls_out(hrd, settings):
    memory.calculations.max_size = 2
    others = []
    for u_max in [41.0, 42.0]:
        other = copy.deepcopy(settings)
        other.u_max = u_max
        others.append(other)
    first = Calculation(hrd, settings)
    second = Calculation(hrd, others[0])
    assert Calculation(hrd, settings) is first
    Calculation(hrd, others[1])
    assert len(memory.calculations) == 2
    assert Calculation(hrd, settings) is first
    assert Calculation(hrd, others[0]) is not second


def test_ports_use_the_memory_by_default(hrd, settings):
    crest_profile = profile()
    first = HydraPercentiles(hrd, settings, False, False, response="hbn", profile=crest_profile, qcrit=QCRIT)
    assert HydraPercentiles(hrd, settings, False, False, response="hbn", profile=crest_profile, qcrit=QCRIT) is first
    first.percentiles(5.25)
    assert memory.crest_level_tables.misses > 0

    # A crest level function of the caller can not be labelled: the object is built again and not kept
    own = HydraPercentiles(
        hrd, settings, False, False, response="hbn", profile=crest_profile, qcrit=QCRIT,
        crest_level_table=crest_level_tables.CrestLevelTables.crest_level_table,
    )
    assert own is not first
    assert memory.calculations.misses == 1


def test_set_label_does_not_depend_on_order():
    # A caller may add a set to the settings, for example the repairs it switches off
    assert memory.Memory.label(frozenset({"b", "a", 1})) == memory.Memory.label({1, "a", "b"})
    assert memory.Memory.label(frozenset({"a"})) != memory.Memory.label(frozenset({"b"}))


def test_tables_calculated_ahead_are_the_same(monkeypatch):
    # Below the threshold nothing is calculated ahead, so lower it: the Benedenrijn without model uncertainty is small
    monkeypatch.setattr(crest_level_tables, "MIN_POINTS_PARALLEL", 1)
    monkeypatch.setattr(memory, "processes", 2)
    lower_rhine = HRDatabase(DATA / "03_RHINE_TIDAL" / "WBI2017_Benedenrijn_210_v04.sqlite")
    location_settings = lower_rhine.get_settings(lower_rhine.get_location_names()[0])
    crest_profile = profile()
    level = 5.42

    ahead = HydraRhineMeuseDeltaPercentiles(
        lower_rhine, location_settings, True, False, response="hbn", profile=crest_profile, qcrit=QCRIT
    ).breakdown(level)
    line_ahead = HydraRhineMeuseDeltaPercentiles.crest_level_line(
        lower_rhine, location_settings, True, False, crest_profile, QCRIT
    )
    # Every table the calculations asked for was calculated ahead
    assert len(memory.crest_level_tables) > 0
    assert memory.crest_level_tables.misses == 0

    own = crest_level_tables.CrestLevelTables.crest_level_table
    here = HydraRhineMeuseDeltaPercentiles(
        lower_rhine, location_settings, True, False, response="hbn", profile=crest_profile, qcrit=QCRIT,
        crest_level_table=own,
    ).breakdown(level)
    line_here = HydraRhineMeuseDeltaPercentiles.crest_level_line(
        lower_rhine, location_settings, True, False, crest_profile, QCRIT, own
    )
    for key in here:
        np.testing.assert_array_equal(ahead[key], here[key], err_msg=key)
    for key in line_here:
        np.testing.assert_array_equal(line_ahead[key], line_here[key], err_msg=key)
