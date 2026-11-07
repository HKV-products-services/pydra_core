"""
The choice of the scenario: 'get_settings' and 'create_location' take the scenario of the statistics and settings in
'calculation_settings.sqlite', an unknown name is an error, the Settings carry the scenario and it is part of the label
in 'core.memory'; the BOI2023 scenarios have every setting of WBI2017 and their own files.
"""

import copy
from pathlib import Path

import numpy as np
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


def test_the_default_scenario_is_boi2023_2035(hrd):
    location_name = hrd.get_location_names()[0]
    assert DEFAULT_SCENARIO == "BOI2023_2035"
    assert hrd.get_settings(location_name).scenario == "BOI2023_2035"
    assert hrd.create_location(location_name).get_settings().scenario == "BOI2023_2035"


def test_the_scenario_is_passed_on(hrd):
    location_name = hrd.get_location_names()[0]
    assert hrd.get_settings(location_name, scenario="WBI2017").scenario == "WBI2017"
    assert hrd.create_location(location_name, scenario="WBI2017").get_settings().scenario == "WBI2017"


def test_the_scenarios():
    with CalculationSettingsReader() as database:
        assert database.get_scenarios() == ["BOI2023_2035", "BOI2023_2075_Mn", "BOI2023_2125_Mn", "WBI2017"]


def test_every_scenario_has_the_settings_of_wbi2017():
    # Hydra-NL 2.9.2 has no discharge statistics of Olst for 2075 and 2125 (a dummy file), so the IJssel delta has none
    with CalculationSettingsReader() as database:
        names = {}
        for scenario, water_system, name in database.con.execute("SELECT Scenario, WaterSystem, SettingName FROM CalculationSettings"):
            names.setdefault(scenario, set()).add((water_system, name))
    for scenario in ["BOI2023_2035", "BOI2023_2075_Mn", "BOI2023_2125_Mn"]:
        missing = names["WBI2017"] - names[scenario]
        assert missing == (set() if scenario == "BOI2023_2035" else {(5, "discharge_probability")}), scenario
        assert names[scenario] <= names["WBI2017"], scenario


def test_every_file_in_the_settings_exists_with_and_without_statistical_uncertainty():
    statistics = FileHydraNL.PACKAGE_PATH
    with CalculationSettingsReader() as database:
        paths = list(database.con.execute("SELECT Scenario, SettingValue FROM CalculationSettings WHERE SettingValue LIKE '%/%'"))
        for table in ["CoastSubSystems", "WaddenSeaSubSystems", "WesternScheldtSubSystems"]:
            for scenario, *points in database.con.execute(f"SELECT Scenario, Point1, Point2, Point3 FROM {table}"):
                paths += [(scenario, point) for point in points]
    missing = [(scenario, variant) for scenario, path in paths for variant in {path, path.replace("_metOnzHeid", "")} if not (statistics / variant).is_file()]
    assert paths and not missing
    assert all(path.startswith(f"{scenario.split('_')[0]}/") for scenario, path in paths)


@pytest.mark.parametrize("scenario", ["BOI2023_2035", "BOI2023_2075_Mn", "BOI2023_2125_Mn"])
def test_a_boi2023_location_has_its_own_statistics(hrd, scenario):
    location_name = hrd.get_location_names()[0]
    settings = hrd.get_settings(location_name, scenario=scenario)
    assert settings.wind_speed_probability.startswith("BOI2023/")
    assert settings.lake_level_rise == {"BOI2023_2035": 0.0, "BOI2023_2075_Mn": 0.25, "BOI2023_2125_Mn": 0.3}[scenario]
    statistics = hrd.create_location(settings).get_model().get_statistics()
    wbi2017 = hrd.create_location(location_name, scenario="WBI2017").get_model().get_statistics()
    assert not np.array_equal(statistics.get_lake_level().epapeak, wbi2017.get_lake_level().epapeak)


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


@pytest.mark.parametrize("scenario", ["BOI2023_2075_Mn", "BOI2023_2125_Mn"])
def test_a_scenario_without_discharge_statistics_is_an_error(scenario):
    ijssel_delta = HRDatabase(str(next((Path(__file__).parent / "data" / "05_IJSSEL_DELTA").glob("*.sqlite"))))
    with pytest.raises(ValueError, match=f"Scenario '{scenario}' has no discharge statistics for water system IJSSEL_DELTA"):
        ijssel_delta.create_location(ijssel_delta.get_location_names()[0], scenario=scenario)


def test_a_settings_object_with_a_scenario_is_an_error(hrd):
    settings = hrd.get_settings(hrd.get_location_names()[0])
    with pytest.raises(ValueError, match="carries its own scenario"):
        hrd.create_location(settings, scenario="WBI2017")


def test_the_scenario_is_in_the_label(hrd):
    settings = hrd.get_settings(hrd.get_location_names()[0])
    other = copy.deepcopy(settings)
    other.scenario = "WBI2017"
    assert Memory.label(settings) != Memory.label(other)
