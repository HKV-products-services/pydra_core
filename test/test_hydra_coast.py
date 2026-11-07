"""
The Hydra-NL breakdown, percentiles and illustration points of the coast ('hydra.coast_breakdown' and
'hydra.coast_illustration_points') against frozen results in 'test/data/hydra/'. The frozen results come from these
ports, made after they were found equal bit for bit to the implementation they were ported from. A relative tolerance
of 1e-12 leaves room for rounding on another platform, not for a change in the calculation.

The Holland coast central has falling loading lines that the repair changes, the Holland coast south has none.

Writing the frozen results again, after a deliberate change of the calculation:

    python test/test_hydra_coast.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase, Profile
from pydra_core.hydra.coast_breakdown import HydraCoastBreakdown
from pydra_core.hydra.coast_illustration_points import HydraCoastIllustrationPoints

DATA = Path(__file__).parent / "data"
EXPECTED = DATA / "hydra" / "coast.json"

# Database, orientation of the profile, level per response
SYSTEMS = {
    "central": (
        DATA / "12_COAST_CENTRAL" / "WBI2017_Hollandse_Kust_Midden_13-2_44-3_v04.sqlite",
        290.0,
        {"h": 5.0, "hs": 4.57, "hbn": 17.38},
    ),
    "south": (
        DATA / "13_COAST_SOUTH" / "WBI2017_Hollandse_Kust_Zuid_14-5_14-6_14-8_211_214_218_25-1_v04.sqlite",
        300.0,
        {"h": 5.0, "hs": 2.98},
    ),
}
UNCERTAINTIES = {"none": (False, False), "all": (True, True)}
CASES = [
    ("central", "h", "none"),
    ("central", "h", "all"),
    ("central", "hs", "none"),
    ("central", "hbn", "none"),
    ("south", "h", "all"),
    ("south", "hs", "none"),
]
QCRIT = 0.01


def calculate(system: str, response: str, uncertainty: str) -> dict:
    path, orientation, levels = SYSTEMS[system]
    statistical, model = UNCERTAINTIES[uncertainty]
    hrd = HRDatabase(path)
    settings = hrd.get_settings(hrd.get_location_names()[0], scenario="WBI2017")
    level = levels[response]
    profile = None
    if response == "hbn":
        profile = Profile("test", crest_level=10.0, orientation=orientation, cota_slope=3.0)
    qcrit = QCRIT if response == "hbn" else None

    breakdown = HydraCoastBreakdown(hrd, settings, statistical, model, response=response, profile=profile, qcrit=qcrit)
    result = breakdown.breakdown(level)
    out = {
        "percentiles": HydraCoastBreakdown.breakdown_percentiles(result),
        "freq_r": result["freq_r"].tolist(),
        "route": result["route"],
    }
    if response in ("h", "hbn"):
        points = HydraCoastIllustrationPoints(
            hrd, settings, statistical, model, response=response, profile=profile, qcrit=qcrit
        )
        out["illustration_points"] = points.illustration_points(level)
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
def test_hydra_coast(system: str, response: str, uncertainty: str):
    expected = json.loads(EXPECTED.read_text())[case_name(system, response, uncertainty)]
    assert_close(json.loads(json.dumps(calculate(system, response, uncertainty))), expected)


if __name__ == "__main__":
    EXPECTED.parent.mkdir(exist_ok=True)
    results = {case_name(*case): calculate(*case) for case in CASES}
    EXPECTED.write_text(json.dumps(results, indent=1))
