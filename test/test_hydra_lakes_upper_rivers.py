"""
The Hydra-NL percentiles and illustration points of the lakes and the upper rivers ('hydra.percentiles' and
'hydra.illustration_points') against frozen results in 'test/data/hydra/'. The frozen results come from these ports,
made after they were found equal bit for bit to the implementation they were ported from. A relative tolerance of
1e-12 leaves room for rounding on another platform, not for a change in the calculation.

Writing the frozen results again, after a deliberate change of the calculation:

    python test/test_hydra_lakes_upper_rivers.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase, Profile
from pydra_core.hydra.illustration_points import HydraIllustrationPoint
from pydra_core.hydra.percentiles import HydraPercentiles

DATA = Path(__file__).parent / "data"
EXPECTED = DATA / "hydra" / "lakes_upper_rivers.json"

# Database, wind direction of the limit state field, level of the water level, level of the crest level
SYSTEMS = {
    "lake": (DATA / "07_IJSSEL_LAKE" / "WBI2017_IJsselmeer_7-2_v02.sqlite", 225.0, 2.29, 5.25),
    "upper_river": (DATA / "01_RHINE_NON_TIDAL" / "WBI2017_Bovenrijn_213_v04.sqlite", 180.0, 7.72, None),
}
UNCERTAINTIES = {"none": (False, False), "all": (True, True)}
CASES = [
    ("lake", "h", "none"),
    ("lake", "h", "all"),
    ("upper_river", "h", "none"),
    ("upper_river", "h", "all"),
    ("lake", "hbn", "none"),
]
QCRIT = 0.01


def calculate(system: str, response: str, uncertainty: str) -> dict:
    path, direction, level_h, level_hbn = SYSTEMS[system]
    statistical, model = UNCERTAINTIES[uncertainty]
    hrd = HRDatabase(path)
    settings = hrd.get_settings(hrd.get_location_names()[0], scenario="WBI2017")
    level = level_hbn if response == "hbn" else level_h
    profile = None
    if response == "hbn":
        profile = Profile("test", crest_level=10.0, orientation=direction, cota_slope=3.0)
    qcrit = QCRIT if response == "hbn" else None

    percentiles = HydraPercentiles(hrd, settings, statistical, model, response=response, profile=profile, qcrit=qcrit)
    point = HydraIllustrationPoint(hrd, settings, statistical, model, response=response, profile=profile, qcrit=qcrit)
    return {
        "percentiles": percentiles.percentiles(level),
        "wind_percentiles": percentiles.wind_percentiles_per_direction(level),
        "illustration_points": point.all_directions(level),
        "design_grid": point.design_grid(direction, level, n_u=10, n_s=10),
    }


def assert_close(actual, expected, path: str = "") -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list) and expected and not all(isinstance(v, int | float) for v in expected):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, str) or expected is None:
        assert actual == expected, path
    else:
        np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), rtol=1e-12, atol=0.0,
                                   err_msg=path)


def case_name(system: str, response: str, uncertainty: str) -> str:
    return f"{system}_{response}_{uncertainty}"


@pytest.mark.parametrize("system, response, uncertainty", CASES)
def test_hydra_lakes_upper_rivers(system: str, response: str, uncertainty: str):
    expected = json.loads(EXPECTED.read_text())[case_name(system, response, uncertainty)]
    assert_close(json.loads(json.dumps(calculate(system, response, uncertainty))), expected)


if __name__ == "__main__":
    EXPECTED.parent.mkdir(exist_ok=True)
    results = {case_name(*case): calculate(*case) for case in CASES}
    EXPECTED.write_text(json.dumps(results, indent=1))
