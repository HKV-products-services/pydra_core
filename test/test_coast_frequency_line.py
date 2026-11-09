"""
The frequency line of the coast against Hydra-NL v2.9.2, BOI2023 2035, on the nine locations of the Hydra-NL demo
database 'DEMO_HollandseKust'.

The reference frequency lines in 'test/data/DEMO_HOLLANDSE_KUST/hydranl/<location>/' are the files 'hfreq.txt' (water
level), 'hsfreq.txt' (significant wave height) and 'tmfreq.txt' (spectral wave period Tm-1,0) of Rekenhart_Zout, from
the demo calculations without model uncertainty and with statistical uncertainty, MSTAP = 0.1 and REP_ALLEEN_U = NEE.

pydra_core uses its own settings and statistics of the scenario BOI2023_2035; they are equal to those of the demo
calculations, including 'm_step' 0.1. The frequencies are compared on the levels of Hydra-NL between 1e-6 and 0.2 per
year, with a relative tolerance of 0.5 %, and the levels at return periods of 10 to 100000 years within 0.5 mm.
"""

from pathlib import Path

import numpy as np
import pytest

from pydra_core import ExceedanceFrequencyLine, HRDatabase

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
REFERENCE_FILES = {"h": "hfreq.txt", "hs": "hsfreq.txt", "tspec": "tmfreq.txt"}
STEP_SIZE = 0.1
FREQUENCY_RANGE = (1e-6, 0.2)
TOLERANCE = 0.005
RETURN_FREQUENCIES = np.array([1 / 10, 1 / 100, 1 / 1000, 1 / 10000, 1 / 100000])
LEVEL_TOLERANCE = 0.0005


def read_hydranl_frequency_line(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = path.read_text().split()
    count = int(float(values[0]))
    table = np.array(values[1 : 1 + 2 * count], float).reshape(count, 2)
    return table[:, 0], table[:, 1]


@pytest.mark.parametrize("result_variable", REFERENCE_FILES)
@pytest.mark.parametrize("location_name", LOCATIONS)
def test_coast_frequency_line(location_name: str, result_variable: str):
    levels, frequencies = read_hydranl_frequency_line(DATA / "hydranl" / location_name / REFERENCE_FILES[result_variable])

    hrd = HRDatabase(DATABASE)
    location = hrd.create_location(hrd.get_settings(location_name))
    calculation = ExceedanceFrequencyLine(result_variable, model_uncertainty=False)
    calculation.set_range(levels[0], levels[-1])
    calculation.set_step_size(STEP_SIZE)
    line = calculation.calculate(location)

    np.testing.assert_allclose(line.level, levels, atol=1e-6)
    compared = (frequencies >= FREQUENCY_RANGE[0]) & (frequencies <= FREQUENCY_RANGE[1])
    np.testing.assert_allclose(line.exceedance_frequency[compared], frequencies[compared], rtol=TOLERANCE, atol=0.0)


@pytest.mark.parametrize("result_variable", REFERENCE_FILES)
@pytest.mark.parametrize("location_name", LOCATIONS)
def test_coast_level_at_return_period(location_name: str, result_variable: str):
    # The default levels of the coast are those of Hydra-NL (multiples of 0.1), so the level at a return period, read
    # by linear interpolation against log(frequency) as VULLENUITVOER_ZOUT does, is that of Hydra-NL
    levels, frequencies = read_hydranl_frequency_line(DATA / "hydranl" / location_name / REFERENCE_FILES[result_variable])
    hydranl = np.interp(np.log(RETURN_FREQUENCIES), np.log(frequencies[::-1]), levels[::-1])

    hrd = HRDatabase(DATABASE)
    location = hrd.create_location(hrd.get_settings(location_name))
    line = ExceedanceFrequencyLine(result_variable, model_uncertainty=False).calculate(location)

    np.testing.assert_allclose(line.interpolate_exceedance_probability(RETURN_FREQUENCIES), hydranl, rtol=0.0,
                               atol=LEVEL_TOLERANCE)
