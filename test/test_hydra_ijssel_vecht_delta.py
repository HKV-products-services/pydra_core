"""
The Hydra-NL illustration point of the IJssel-Vecht delta ('hydra.illustration_points_3d' and
'hydra.ijssel_vecht_delta') against frozen results in 'test/data/hydra/'. The frozen results come from these ports
with the default scenario BOI2023_2035, made after they were found equal bit for bit to the implementation they were
ported from and close to Hydra-NL v2.9.2 on the same locations. A relative tolerance of 1e-12 leaves room for rounding
on another platform, not for a change in the calculation.

The levels are those of Hydra-NL v2.9.2 at a return period of 10000 years, without uncertainties.

The test databases have no KeringUD table, so the calculation reads the KeringUD CSV from the folder 'data/ini' of a
Hydra-NL installation. That folder is taken from the environment variable 'HYDRA_NL_INI', by default
'C:\\Software\\Hydra-NL_v292\\data\\ini'; without it the tests are skipped.

Writing the frozen results again, after a deliberate change of the calculation:

    python test/test_hydra_ijssel_vecht_delta.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase, Profile
from pydra_core.hydra.ijssel_vecht_delta import BarrierClosingData, HydraIJsselVechtDeltaIllustrationPoint

DATA = Path(__file__).parent / "data"
EXPECTED = DATA / "hydra" / "ijssel_vecht_delta.json"
HYDRA_INI = Path(os.environ.get("HYDRA_NL_INI", r"C:\Software\Hydra-NL_v292\data\ini"))

# Database, wind direction of the limit state field and the block probability, level of the water level, level of the
# crest level
SYSTEMS = {
    "ijssel_delta": (DATA / "05_IJSSEL_DELTA" / "WBI2023_IJsseldelta_206_v00_terBeoordeling.sqlite", 270.0, 3.99, 4.13),
    "vecht_delta": (DATA / "06_VECHT_DELTA" / "WBI2023_Vechtdelta_202_v00_terBeoordeling.sqlite", 270.0, 1.49, None),
}
UNCERTAINTIES = {"none": (False, False), "all": (True, True)}
CASES = [
    ("ijssel_delta", "h", "none"),
    ("ijssel_delta", "h", "all"),
    ("vecht_delta", "h", "none"),
    ("vecht_delta", "h", "all"),
    ("ijssel_delta", "hbn", "none"),
]
QCRIT = 0.01

needs_hydra_ini = pytest.mark.skipif(
    BarrierClosingData.find_barrier_csv(SYSTEMS["ijssel_delta"][0], HYDRA_INI) is None
    or BarrierClosingData.find_barrier_csv(SYSTEMS["vecht_delta"][0], HYDRA_INI) is None,
    reason=f"No KeringUD CSV of Hydra-NL in {HYDRA_INI}; set 'HYDRA_NL_INI' to the folder 'data/ini' of Hydra-NL.",
)


def calculate(system: str, response: str, uncertainty: str) -> dict:
    path, direction, level_h, level_hbn = SYSTEMS[system]
    statistical, model = UNCERTAINTIES[uncertainty]
    hrd = HRDatabase(path)
    settings = hrd.get_settings(hrd.get_location_names()[0])
    level = level_hbn if response == "hbn" else level_h
    profile = None
    if response == "hbn":
        profile = Profile("test", crest_level=10.0, orientation=225.0, cota_slope=3.0)
    qcrit = QCRIT if response == "hbn" else None

    point = HydraIJsselVechtDeltaIllustrationPoint(
        hrd, settings, statistical, model, response=response, profile=profile, qcrit=qcrit,
        hydra_ini_directory=HYDRA_INI,
    )
    out = {
        "illustration_points": point.all_directions(level),
        "barrier_weights": [point.barrier_weights(r) for r in point.directions()],
        "design_grid": point.design_grid(direction, level, 2, n_x=10, n_y=10),
    }
    if response == "h":
        out["p_failure_block"] = point.p_failure_block(direction, level)
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


@needs_hydra_ini
@pytest.mark.parametrize("system, response, uncertainty", CASES)
def test_hydra_ijssel_vecht_delta(system: str, response: str, uncertainty: str):
    expected = json.loads(EXPECTED.read_text())[case_name(system, response, uncertainty)]
    assert_close(json.loads(json.dumps(calculate(system, response, uncertainty))), expected)


def test_refuses_a_lake():
    hrd = HRDatabase(DATA / "07_IJSSEL_LAKE" / "WBI2017_IJsselmeer_7-2_v02.sqlite")
    settings = hrd.get_settings(hrd.get_location_names()[0])
    with pytest.raises(NotImplementedError):
        HydraIJsselVechtDeltaIllustrationPoint(hrd, settings, hydra_ini_directory=HYDRA_INI)


if __name__ == "__main__":
    EXPECTED.parent.mkdir(exist_ok=True)
    results = {case_name(*case): calculate(*case) for case in CASES}
    EXPECTED.write_text(json.dumps(results, indent=1))
