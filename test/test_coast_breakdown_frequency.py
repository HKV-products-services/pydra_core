"""
The frequency of the coastal breakdown of the waves ('HydraCoastBreakdown.breakdown' for Hm0 and Tm-1,0) against
Hydra-NL v2.9.2, BOI2023 2035, on the nine locations of the Hydra-NL demo database 'DEMO_HollandseKust'.

Hydra-NL calculates the frequency line of a wave parameter with the same contour route as the breakdown
(BEREKEN_FAALKANSZEE), so the frequency of the breakdown at a level of Hydra-NL equals that of the frequency line. The
reference lines in 'test/data/DEMO_HOLLANDSE_KUST/hydranl/<location>/' are 'hsfreq.txt' and 'tmfreq.txt' without model
uncertainty, and 'hsfreq_model_uncertainty.txt' and 'tmfreq_model_uncertainty.txt' with it (ONZ_GOLFHOOGTE with
GH_ONZ_AANTAL = 5 for Hm0, ONZ_GOLFPERIODE with GP_ONZ_AANTAL = 5 for Tm-1,0), all with statistical uncertainty,
MSTAP = 0.1 and REP_ALLEEN_U = NEE.

The frequencies are compared on the levels of Hydra-NL between 1e-6 and 0.2 per year, with a relative tolerance of
0.5 %.

The percentiles and the contribution per direction of the water level are compared with the table "Bijdragen en
percentielen" of Hydra-NL at return periods of 100 to 10000 years ('hydranl/h_percentiles.json', from 'uitvoer.html'
of the same demo calculations). Hydra-NL prints the sea level on 0.01 m, the wind speed on 0.1 m/s and the
contribution on 0.01 %, so the tolerance is half of that plus a margin for the level, which it prints on 0.001 m. For
the waves this comparison is not made: there the contour route jumps when the level passes a grid node, and the
printed level falls on the other side of such a jump in a few cases.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase
from pydra_core.hydra.coast_breakdown import HydraCoastBreakdown

DATA = Path(__file__).parent / "data" / "DEMO_HOLLANDSE_KUST"
DATABASE = DATA / "DEMO_HollandseKust.sqlite"
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
REFERENCE_FILES = {
    ("hs", False): "hsfreq.txt",
    ("tspec", False): "tmfreq.txt",
    ("hs", True): "hsfreq_model_uncertainty.txt",
    ("tspec", True): "tmfreq_model_uncertainty.txt",
}
FREQUENCY_RANGE = (1e-6, 0.2)
TOLERANCE = 0.005
PERCENTAGE_TOLERANCE = 0.02
SEA_LEVEL_TOLERANCE = 0.006
WIND_SPEED_TOLERANCE = 0.06


def read_hydranl_frequency_line(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = path.read_text().split()
    count = int(float(values[0]))
    table = np.array(values[1 : 1 + 2 * count], float).reshape(count, 2)
    return table[:, 0], table[:, 1]


@pytest.mark.parametrize("response, model_uncertainty", REFERENCE_FILES)
@pytest.mark.parametrize("location_name", LOCATIONS)
def test_coast_breakdown_frequency(location_name: str, response: str, model_uncertainty: bool):
    path = DATA / "hydranl" / location_name / REFERENCE_FILES[response, model_uncertainty]
    levels, frequencies = read_hydranl_frequency_line(path)
    compared = (frequencies >= FREQUENCY_RANGE[0]) & (frequencies <= FREQUENCY_RANGE[1])

    hrd = HRDatabase(DATABASE)
    breakdown = HydraCoastBreakdown(
        hrd, hrd.get_settings(location_name), model_uncertainty=model_uncertainty, response=response
    )
    calculated = np.array([breakdown.breakdown(level)["frequency"] for level in levels[compared]])

    np.testing.assert_allclose(calculated, frequencies[compared], rtol=TOLERANCE, atol=0.0)


@pytest.mark.parametrize("location_name", LOCATIONS)
def test_coast_water_level_percentiles(location_name: str):
    cases = json.loads((DATA / "hydranl" / "h_percentiles.json").read_text())[location_name]

    hrd = HRDatabase(DATABASE)
    breakdown = HydraCoastBreakdown(hrd, hrd.get_settings(location_name), model_uncertainty=False, response="h")
    for case in cases:
        table = breakdown.percentiles(case["level"])
        contributions = [direction["percentage"] for direction in table["directions"]]
        message = f"return period {case['return_period']}"
        np.testing.assert_allclose(contributions, case["directions"], rtol=0.0, atol=PERCENTAGE_TOLERANCE,
                                   err_msg=message)
        np.testing.assert_allclose(table["sea_level"], case["sea_level"], rtol=0.0, atol=SEA_LEVEL_TOLERANCE,
                                   err_msg=message)
        np.testing.assert_allclose(table["wind_speed"], case["wind_speed"], rtol=0.0, atol=WIND_SPEED_TOLERANCE,
                                   err_msg=message)
