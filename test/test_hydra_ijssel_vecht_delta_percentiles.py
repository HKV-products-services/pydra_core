"""
The Hydra-NL breakdown and percentiles of the IJssel-Vecht delta ('hydra.ijssel_vecht_delta_percentiles') against
frozen results in 'test/data/hydra/'. The frozen results come from this port, made after it was found equal bit for
bit to the implementation it was ported from. A relative tolerance of 1e-12 leaves room for rounding on another
platform, not for a change in the calculation.

The test databases have no KeringUD table, so the calculation reads the KeringUD CSV from the folder 'data/ini' of a
Hydra-NL installation. That folder is taken from the environment variable 'HYDRA_NL_INI', by default
'C:\\Software\\Hydra-NL_v292\\data\\ini'; without it the tests are skipped.

Writing the frozen results again, after a deliberate change of the calculation:

    python test/test_hydra_ijssel_vecht_delta_percentiles.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase, Profile
from pydra_core.hydra.ijssel_vecht_delta import BarrierClosingData
from pydra_core.hydra.ijssel_vecht_delta_percentiles import (
    HydraIJsselVechtDeltaPercentiles,
)

DATA = Path(__file__).parent / "data"
EXPECTED = DATA / "hydra" / "ijssel_vecht_delta_percentiles.json"
HYDRA_INI = Path(os.environ.get("HYDRA_NL_INI", r"C:\Software\Hydra-NL_v292\data\ini"))

# Database, level of the water level, level of the crest level
SYSTEMS = {
    "ijssel_delta": (DATA / "05_IJSSEL_DELTA" / "WBI2023_IJsseldelta_206_v00_terBeoordeling.sqlite", 4.66, 5.48),
    "vecht_delta": (DATA / "06_VECHT_DELTA" / "WBI2023_Vechtdelta_202_v00_terBeoordeling.sqlite", 1.83, None),
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
RETURN_PERIODS = (10, 100, 1000, 10000)

needs_hydra_ini = pytest.mark.skipif(
    BarrierClosingData.find_barrier_csv(SYSTEMS["ijssel_delta"][0], HYDRA_INI) is None
    or BarrierClosingData.find_barrier_csv(SYSTEMS["vecht_delta"][0], HYDRA_INI) is None,
    reason=f"No KeringUD CSV of Hydra-NL in {HYDRA_INI}; set 'HYDRA_NL_INI' to the folder 'data/ini' of Hydra-NL.",
)


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

    percentiles = HydraIJsselVechtDeltaPercentiles(
        hrd, settings, statistical, model, response=response, profile=profile, qcrit=qcrit,
        hydra_ini_directory=HYDRA_INI,
    )
    breakdown = percentiles.breakdown(level)
    out = {
        "freqRK": breakdown["freqRK"].tolist(),
        "freqQK": breakdown["freqQK"].tolist(),
        "frequency": percentiles.frequency(level),
        "percentiles": percentiles.percentiles(level),
        "wind_percentiles": [
            percentiles.wind_percentiles_per_direction(level, state) for state in (None, "open", "closed")
        ],
        "direction_contributions": percentiles.direction_contributions(level),
    }
    if (system, response, uncertainty) == ("ijssel_delta", "h", "none"):
        levels, frequencies = percentiles.line([1.0 / t for t in RETURN_PERIODS])
        out["line"] = {"levels": levels, "frequencies": frequencies}
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
def test_hydra_ijssel_vecht_delta_percentiles(system: str, response: str, uncertainty: str):
    expected = json.loads(EXPECTED.read_text())[case_name(system, response, uncertainty)]
    assert_close(json.loads(json.dumps(calculate(system, response, uncertainty))), expected)


def test_refuses_a_lake():
    hrd = HRDatabase(DATA / "07_IJSSEL_LAKE" / "WBI2017_IJsselmeer_7-2_v02.sqlite")
    settings = hrd.get_settings(hrd.get_location_names()[0])
    with pytest.raises(NotImplementedError):
        HydraIJsselVechtDeltaPercentiles(hrd, settings, hydra_ini_directory=HYDRA_INI)


if __name__ == "__main__":
    EXPECTED.parent.mkdir(exist_ok=True)
    results = {case_name(*case): calculate(*case) for case in CASES}
    EXPECTED.write_text(json.dumps(results, indent=1))
