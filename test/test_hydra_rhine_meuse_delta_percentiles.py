"""
The Hydra-NL breakdown, percentiles and crest level line of the Rhine-Meuse delta ('hydra.rhine_meuse_delta_percentiles')
against frozen results in 'test/data/hydra/'. The frozen results come from this port, made after it was found equal bit
for bit to the implementation it was ported from. A relative tolerance of 1e-12 leaves room for rounding on another
platform, not for a change in the calculation.

The crest level line rebuilds 'LoadingModel.refine' as a matrix; 'test_refinement_matrix_equals_refine' checks on every
loading model of a location that both give the same refined table.

Writing the frozen results again, after a deliberate change of the calculation:

    python test/test_hydra_rhine_meuse_delta_percentiles.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase, Profile
from pydra_core.hydra.rhine_meuse_delta_percentiles import HydraRhineMeuseDeltaPercentiles

DATA = Path(__file__).parent / "data"
EXPECTED = DATA / "hydra" / "rhine_meuse_delta_percentiles.json"
LAKE = DATA / "07_IJSSEL_LAKE" / "WBI2017_IJsselmeer_7-2_v02.sqlite"

# Database, level of the water level, level of the crest level
SYSTEMS = {
    "lower_rhine": (DATA / "03_RHINE_TIDAL" / "WBI2017_Benedenrijn_210_v04.sqlite", 5.08, 5.42),
    "lower_meuse": (DATA / "04_MEUSE_TIDAL" / "WBI2017_Benedenmaas_37-1_v03.sqlite", 7.4, None),
}
UNCERTAINTIES = {"none": (False, False), "all": (True, True)}
CASES = [
    ("lower_rhine", "h", "none"),
    ("lower_rhine", "h", "all"),
    ("lower_meuse", "h", "none"),
    ("lower_meuse", "h", "all"),
    ("lower_rhine", "hbn", "none"),
]
QCRIT = 0.01


def profile() -> Profile:
    return Profile("test", crest_level=10.0, orientation=225.0, cota_slope=3.0)


def calculate(system: str, response: str, uncertainty: str) -> dict:
    path, level_h, level_hbn = SYSTEMS[system]
    statistical, model = UNCERTAINTIES[uncertainty]
    hrd = HRDatabase(path)
    settings = hrd.get_settings(hrd.get_location_names()[0], scenario="WBI2017")
    level = level_hbn if response == "hbn" else level_h
    qcrit = QCRIT if response == "hbn" else None

    percentiles = HydraRhineMeuseDeltaPercentiles(
        hrd, settings, statistical, model, response=response, profile=profile() if response == "hbn" else None,
        qcrit=qcrit,
    )
    breakdown = percentiles.breakdown(level)
    out = {
        # The arrays per wind direction ('freqURK', 'freqUR') are tested through the wind percentiles per direction
        "breakdown": {key: breakdown[key].tolist() for key in ("freqQK", "freqMK", "freqRK", "freqUK")},
        "percentiles": percentiles.percentiles(level),
        "wind_percentiles": {
            str(state): percentiles.wind_percentiles_per_direction(level, state) for state in (None, "open", "closed")
        },
        "direction_contributions": percentiles.direction_contributions(level),
    }
    if response == "hbn":
        line = HydraRhineMeuseDeltaPercentiles.crest_level_line(hrd, settings, statistical, model, profile(), QCRIT)
        out["line"] = {key: np.asarray(value).tolist() for key, value in line.items()}
    return out


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
def test_hydra_rhine_meuse_delta_percentiles(system: str, response: str, uncertainty: str):
    expected = json.loads(EXPECTED.read_text())[case_name(system, response, uncertainty)]
    assert_close(json.loads(json.dumps(calculate(system, response, uncertainty))), expected)


def test_refinement_matrix_equals_refine():
    hrd = HRDatabase(SYSTEMS["lower_rhine"][0])
    model = hrd.create_location(hrd.get_settings(hrd.get_location_names()[0], scenario="WBI2017")).get_model()
    stats = model.get_statistics()
    grids = {**stats.stochastics_fast, **stats.stochastics_slow}
    for (r, k), m in model.get_loading().iter_models():
        grid = {v: grids[v] for v in m.input_variables}
        w, shape = HydraRhineMeuseDeltaPercentiles.refinement_matrix(m, grid)
        refined = m.refine("h", grid)
        assert refined.shape == shape, (r, k)
        np.testing.assert_allclose(w @ np.asarray(m.h).ravel(), refined.ravel(), rtol=1e-12, atol=0.0,
                                   err_msg=f"direction {r}, closing situation {k}")


def test_refuses_a_lake():
    hrd = HRDatabase(LAKE)
    settings = hrd.get_settings(hrd.get_location_names()[0], scenario="WBI2017")
    with pytest.raises(NotImplementedError):
        HydraRhineMeuseDeltaPercentiles(hrd, settings)
    with pytest.raises(NotImplementedError):
        HydraRhineMeuseDeltaPercentiles.crest_level_line(hrd, settings, True, False, profile(), QCRIT)


if __name__ == "__main__":
    EXPECTED.parent.mkdir(exist_ok=True)
    results = {case_name(*case): calculate(*case) for case in CASES}
    EXPECTED.write_text(json.dumps(results, indent=1))
