"""
The illustration points of the waves at the coast ('HydraCoastIllustrationPoints' for Hm0, Tm-1,0 and Tp) against
Hydra-NL v2.9.2, BOI2023 2035, on the nine locations of the Hydra-NL demo database 'DEMO_HollandseKust'.

The reference 'test/data/DEMO_HOLLANDSE_KUST/hydranl/wave_illustration_points.json' comes from 'uitvoer.html' of the
demo calculations without and with model uncertainty (as in 'test_coast_breakdown_frequency.py'), at return periods
of 100 to 10000 years: the main illustration point, and with model uncertainty the winning factor of every direction
that contributes at least 1 %.

Hydra-NL prints the level on 0.001, the sea level on 0.01 m, the wind speed on 0.1 m/s and the factor on 0.01. Where the
contour of a wave parameter is almost flat in the sea level, the design point moves along it with a level change below
that rounding, so the main point is compared within 0.015 m and 0.11 m/s; the design points of the other directions are
not compared.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase
from pydra_core.hydra.coast_illustration_points import HydraCoastIllustrationPoints

DATA = Path(__file__).parent / "data" / "DEMO_HOLLANDSE_KUST"
DATABASE = DATA / "DEMO_HollandseKust.sqlite"
REFERENCE = DATA / "hydranl" / "wave_illustration_points.json"
LOCATIONS = [
    "HK_1_13-2_dk_00009",
    "HK_1_13-2_dk_00031",
    "HK_1_211_dk_00003",
    "HK_1_25-1_dk_00046",
    "HK_2_13-4_dk_00010",
    "HK_3_hd02-00005",
    "HK_3_hd03-00008",
    "HK_3_hy01-00006",
    "HK_3_kw03-00044",
]
CASES = [("hs", False), ("tspec", False), ("tp", False), ("hs", True), ("tspec", True), ("tp", True)]
SEA_LEVEL_TOLERANCE = 0.015
WIND_SPEED_TOLERANCE = 0.11
FACTOR_TOLERANCE = 0.006


@pytest.mark.parametrize("response, model_uncertainty", CASES)
@pytest.mark.parametrize("location_name", LOCATIONS)
def test_coast_wave_illustration_points(location_name: str, response: str, model_uncertainty: bool):
    name = f"{response}_{'model_uncertainty' if model_uncertainty else 'none'}"
    cases = json.loads(REFERENCE.read_text())[name][location_name]

    hrd = HRDatabase(DATABASE)
    points = HydraCoastIllustrationPoints(
        hrd, hrd.get_settings(location_name), model_uncertainty=model_uncertainty, response=response
    )
    factor_key = f"f_{response}"
    for case in cases:
        result = points.illustration_points(case["level"])
        message = f"return period {case['return_period']}"

        main = result["main"]
        assert main["r"] == case["main"]["r"], message
        np.testing.assert_allclose(main["m"], case["main"]["m"], rtol=0.0, atol=SEA_LEVEL_TOLERANCE, err_msg=message)
        np.testing.assert_allclose(main["u"], case["main"]["u"], rtol=0.0, atol=WIND_SPEED_TOLERANCE, err_msg=message)

        by_direction = {point["r"]: point for point in result["points"]}
        for direction, factor in case["factors"].items():
            np.testing.assert_allclose(by_direction[float(direction)][factor_key], factor, rtol=0.0,
                                       atol=FACTOR_TOLERANCE, err_msg=f"{message}, direction {direction}")
