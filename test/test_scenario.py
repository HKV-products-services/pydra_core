"""
The choice of the scenario: 'get_settings' and 'create_location' take the scenario of the statistics and settings in
'calculation_settings.sqlite', an unknown name is an error, the Settings carry the scenario and it is part of the label
in 'core.memory'.
"""

import copy
from pathlib import Path

import pytest

from pydra_core import HRDatabase
from pydra_core.core.memory import Memory
from pydra_core.io.calculation_settings_reader import CalculationSettingsReader
from pydra_core.io.file_hydranl import FileHydraNL
from pydra_core.location.settings.settings import DEFAULT_SCENARIO

DATABASE = Path(__file__).parent / "data" / "07_IJSSEL_LAKE" / "WBI2017_IJsselmeer_7-2_v02.sqlite"


@pytest.fixture(scope="module")
def hrd():
    return HRDatabase(str(DATABASE))


def test_the_default_scenario_is_wbi2017(hrd):
    location_name = hrd.get_location_names()[0]
    assert DEFAULT_SCENARIO == "WBI2017"
    assert hrd.get_settings(location_name).scenario == "WBI2017"
    assert hrd.create_location(location_name).get_settings().scenario == "WBI2017"


def test_the_scenario_is_passed_on(hrd):
    location_name = hrd.get_location_names()[0]
    assert hrd.get_settings(location_name, scenario="WBI2017").scenario == "WBI2017"
    assert hrd.create_location(location_name, scenario="WBI2017").get_settings().scenario == "WBI2017"


def test_wbi2017_is_one_of_the_scenarios():
    with CalculationSettingsReader() as database:
        assert "WBI2017" in database.get_scenarios()


def test_every_file_in_the_settings_exists():
    statistics = FileHydraNL.PACKAGE_PATH
    with CalculationSettingsReader() as database:
        paths = list(database.con.execute("SELECT Scenario, SettingValue FROM CalculationSettings WHERE SettingValue LIKE '%/%'"))
        for table in ["CoastSubSystems", "WaddenSeaSubSystems", "WesternScheldtSubSystems"]:
            for scenario, *points in database.con.execute(f"SELECT Scenario, Point1, Point2, Point3 FROM {table}"):
                paths += [(scenario, point) for point in points]
    missing = [(scenario, path) for scenario, path in paths if not (statistics / path).is_file()]
    assert paths and not missing
    assert all(path.startswith(f"{scenario.split('_')[0]}/") for scenario, path in paths)


def test_every_scenario_has_the_europoort_barrier_in_every_zone():
    import fiona

    with fiona.open(Path(FileHydraNL.PACKAGE_PATH).parent / "settings" / "lower_river_settings.shp") as shp:
        zones = {feature["properties"]["number"] for feature in shp}
    with CalculationSettingsReader() as database:
        for scenario in database.get_scenarios():
            assert {row[0] for row in database.con.execute("SELECT Zone FROM EuropoortBarrierZones WHERE Scenario = ?", (scenario,))} == zones
            assert database.get_europoort_barrier_zone(8, scenario) == {"Mu": 0.5, "Sigma": 0.001, "Alfa": 0.0}


def test_an_unknown_scenario_is_an_error(hrd):
    location_name = hrd.get_location_names()[0]
    with pytest.raises(ValueError, match="Unknown scenario 'BOI2023'"):
        hrd.get_settings(location_name, scenario="BOI2023")
    with pytest.raises(ValueError, match="Unknown scenario"):
        hrd.create_location(location_name, scenario="wbi2017")


def test_a_settings_object_with_a_scenario_is_an_error(hrd):
    settings = hrd.get_settings(hrd.get_location_names()[0])
    with pytest.raises(ValueError, match="carries its own scenario"):
        hrd.create_location(settings, scenario="WBI2017")


def test_the_scenario_is_in_the_label(hrd):
    settings = hrd.get_settings(hrd.get_location_names()[0])
    other = copy.deepcopy(settings)
    other.scenario = "BOI2023_2035"
    assert Memory.label(settings) != Memory.label(other)
