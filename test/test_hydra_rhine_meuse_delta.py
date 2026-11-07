"""
The Hydra-NL illustration point of the Rhine-Meuse delta ('hydra.illustration_points_3d' and 'hydra.rhine_meuse_delta')
against frozen results in 'test/data/hydra/'. The frozen results come from these ports with the default scenario
BOI2023_2035, made after they were found equal bit for bit to the implementation they were ported from. On the lower
Rhine the water level is close to Hydra-NL v2.9.2; on the lower Meuse the share of the westerly directions is lower,
because the loading table is not extended below its lowest sea level, unlike Hydra-NL. A relative tolerance of 1e-12
leaves room for rounding on another platform, not for a change in the calculation.

The levels are those of Hydra-NL v2.9.2 at a return period of 10000 years, without uncertainties.

Writing the frozen results again, after a deliberate change of the calculation:

    python test/test_hydra_rhine_meuse_delta.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase, Profile
from pydra_core.hydra.library import Grid
from pydra_core.hydra.rhine_meuse_delta import HydraRhineMeuseDeltaIllustrationPoint

DATA = Path(__file__).parent / "data"
EXPECTED = DATA / "hydra" / "rhine_meuse_delta.json"
LAKE = DATA / "07_IJSSEL_LAKE" / "WBI2017_IJsselmeer_7-2_v02.sqlite"

# Database, level of the water level, level of the crest level
SYSTEMS = {
    "lower_rhine": (DATA / "03_RHINE_TIDAL" / "WBI2017_Benedenrijn_210_v04.sqlite", 3.43, 4.05),
    "lower_meuse": (DATA / "04_MEUSE_TIDAL" / "WBI2017_Benedenmaas_37-1_v03.sqlite", 5.67, None),
}
UNCERTAINTIES = {"none": (False, False), "all": (True, True)}
CASES = [
    ("lower_rhine", "h", "none"),
    ("lower_rhine", "h", "all"),
    ("lower_meuse", "h", "none"),
    ("lower_meuse", "h", "all"),
    ("lower_rhine", "hbn", "none"),
]
WEST = 270.0
QCRIT = 0.01


def calculate(system: str, response: str, uncertainty: str) -> dict:
    path, level_h, level_hbn = SYSTEMS[system]
    statistical, model = UNCERTAINTIES[uncertainty]
    hrd = HRDatabase(path)
    settings = hrd.get_settings(hrd.get_location_names()[0])
    level = level_hbn if response == "hbn" else level_h
    profile = None
    if response == "hbn":
        profile = Profile("test", crest_level=10.0, orientation=225.0, cota_slope=3.0)
    qcrit = QCRIT if response == "hbn" else None

    point = HydraRhineMeuseDeltaIllustrationPoint(
        hrd, settings, statistical, model, response=response, profile=profile, qcrit=qcrit
    )
    out = {
        "illustration_points": point.all_directions(level),
        "barrier_weights": [point.barrier_weights(r) for r in point.directions()],
        "design_grid": point.design_grid(WEST, level, 2, n_x=10, n_y=10),
    }
    if response == "h":
        out["p_failure_block"] = point.p_failure_block(WEST, level)
        # The probability mass per discharge of 'bracket_umqk', summed over the directions and the barrier states
        du, dm = Grid.step_sizes(point._ublock), Grid.step_sizes(point._mblock)
        out["mass_per_discharge"] = sum(
            np.einsum("umqk,u,m->q", point.bracket_umqk(ir), du, dm) for ir in range(len(point.directions()))
        ).tolist()
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
def test_hydra_rhine_meuse_delta(system: str, response: str, uncertainty: str):
    expected = json.loads(EXPECTED.read_text())[case_name(system, response, uncertainty)]
    assert_close(json.loads(json.dumps(calculate(system, response, uncertainty))), expected)


def test_recognises_the_rhine_meuse_delta():
    for path, expected in ((SYSTEMS["lower_rhine"][0], True), (SYSTEMS["lower_meuse"][0], True), (LAKE, False)):
        hrd = HRDatabase(path)
        assert (
            HydraRhineMeuseDeltaIllustrationPoint.is_rhine_meuse_delta(
                hrd, hrd.get_settings(hrd.get_location_names()[0])
            )
            is expected
        ), path.name


def test_refuses_a_lake():
    hrd = HRDatabase(LAKE)
    settings = hrd.get_settings(hrd.get_location_names()[0])
    with pytest.raises(NotImplementedError):
        HydraRhineMeuseDeltaIllustrationPoint(hrd, settings)


if __name__ == "__main__":
    EXPECTED.parent.mkdir(exist_ok=True)
    results = {case_name(*case): calculate(*case) for case in CASES}
    EXPECTED.write_text(json.dumps(results, indent=1))
