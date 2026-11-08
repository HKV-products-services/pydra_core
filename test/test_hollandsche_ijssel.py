"""
The Hollandsche IJssel against Hydra-NL v2.9.2, BOI2023 2035, on location '015-03_0089_HY_km0010' of the Hydra-NL
demo database 'DEMO_HollandscheIJssel' (reduced to this location).

The reference frequency lines in 'test/data/22_HOLLANDSCHE_IJSSEL/hydranl/' are the files 'hfreq.txt' (water level) and
'ffq.txt' (crest level) of Rekenhart_HIJ, with the settings of the demo calculation 'HBN_Profiel met knik_10' and only
the uncertainties of each case switched on:

    h_none            water level, without uncertainties
    h_all             water level, with statistical and model uncertainty
    hbn_none          crest level, without uncertainties
    hbn_statistical   crest level, with statistical uncertainty
    hbn_waves         crest level, with model uncertainty of the wave height and wave period
    hbn_model         crest level, with all model uncertainty

With model uncertainty of both the water level and the waves, Rekenhart_HIJ couples a water level value to the
probability of another water level (MODELONZ_WS_HIJ). 'hbn_model' therefore comes from Rekenhart_HIJ built from the
source with that order repaired; unrepaired it lies 5.5 to 6.5 cm higher.

'ip_hbn_none.txt' holds the illustration points of 'hbn_none' per state of the EPK and the HIJK, as Rekenhart_HIJ prints
them (sea level, Rhine discharge and wind speed, at the crest level of each return period).

pydra_core uses its own settings and statistics of the scenario BOI2023_2035; they are equal to those of the demo
calculation. The tolerance is 5 mm at the return periods of the demo calculation, except the water level without
uncertainties at T = 100 (+11 mm, where the Hydra-NL line has a kink).
"""

from pathlib import Path

import numpy as np
import pytest

from pydra_core import HBN, ExceedanceFrequencyLine, HRDatabase, Location, Profile
from pydra_core.hydra.hollandsche_ijssel import HydraHollandscheIJsselIllustrationPoint
from pydra_core.location.model.statistics.water_systems.statistics_lower_river import (
    StatisticsLowerRiver,
)

DATA = Path(__file__).parent / "data" / "22_HOLLANDSCHE_IJSSEL"
DATABASE = DATA / "DEMO_HollandscheIJssel.sqlite"
LOCATION = "015-03_0089_HY_km0010"
QCRIT = 0.01
STEP_SIZE = 0.02
RETURN_PERIODS = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
TOLERANCE = 0.005
TOLERANCE_KINK = {("h_none", 100): 0.015}

# One grid step of the illustration point: sea level, discharge, wind speed (with a margin for the rounding of Hydra-NL)
ONE_STEP = {"M": 0.105, "Q": 205.0, "U": 1.05}

# Illustration points more than one grid step from Hydra-NL: (return period, EPK state, HIJK state, wind direction).
# The near ties verified with beta are (3000, 2, 3, 225.0), (10000, 2, 2, 315.0) and (300000, 2, 3, 270.0).
BEYOND_ONE_STEP = {
    (10, 2, 1, 360.0),
    (30, 2, 1, 337.5),
    (3000, 2, 3, 225.0),
    (10000, 1, 3, 225.0),
    (10000, 2, 2, 315.0),
    (10000, 2, 3, 247.5),
    (30000, 2, 2, 315.0),
    (100000, 1, 1, 225.0),
    (100000, 2, 3, 337.5),
    (300000, 1, 3, 337.5),
    (300000, 2, 3, 270.0),
}

# Reference, result variable, statistical uncertainty, model uncertainty of the water level, of the waves
CASES = [
    ("h_none", "h", False, False, False),
    ("h_all", "h", True, True, True),
    ("hbn_none", "hbn", False, False, False),
    ("hbn_statistical", "hbn", True, False, False),
    ("hbn_waves", "hbn", False, False, True),
    ("hbn_model", "hbn", False, True, True),
]


def read_hydranl(name: str) -> tuple[np.ndarray, np.ndarray]:
    lines = (DATA / "hydranl" / f"{name}.txt").read_text().splitlines()[1:]
    table = np.array([[float(x) for x in line.split()[:2]] for line in lines if line.strip()])
    positive = table[:, 1] > 0
    return table[positive, 0], table[positive, 1]


def dike_profile() -> Profile:
    profile = Profile("Profiel met knik")
    profile.set_dike_crest_level(3.6)
    profile.set_dike_orientation(260.0)
    profile.set_dike_geometry([0.0, 4.0, 10.0], [0.0, 2.0, 4.0], [1.0, 1.0, 1.0])
    return profile


def level_at(levels: np.ndarray, frequencies: np.ndarray, return_period: float) -> float:
    order = np.argsort(frequencies)
    return float(np.interp(np.log(1.0 / return_period), np.log(frequencies[order]), levels[order]))


@pytest.mark.parametrize("reference, result_variable, statistical, water_level, waves", CASES, ids=[c[0] for c in CASES])
def test_frequency_line(reference, result_variable, statistical, water_level, waves):
    settings = HRDatabase(DATABASE).get_settings(LOCATION)
    settings.include_statistical_uncertainty(statistical)
    location = Location(settings, dike_profile() if result_variable == "hbn" else None)

    # Model uncertainty of the waves only: the distribution of the water level is removed
    model_uncertainty = water_level or waves
    if waves and not water_level:
        uncertainties = location.get_model().get_statistics().get_model_uncertainties()
        uncertainties.model_uncertainties = {key: d for key, d in uncertainties.model_uncertainties.items() if key[1] != "h"}

    if result_variable == "hbn":
        calculation = HBN(QCRIT, model_uncertainty=model_uncertainty, processes=1)
    else:
        calculation = ExceedanceFrequencyLine("h", model_uncertainty=model_uncertainty)
    calculation.set_step_size(STEP_SIZE)
    line = calculation.calculate_location(location)

    hydranl_levels, hydranl_frequencies = read_hydranl(reference)
    for return_period in RETURN_PERIODS:
        pydra = level_at(line.level, line.exceedance_frequency, return_period)
        hydranl = level_at(hydranl_levels, hydranl_frequencies, return_period)
        tolerance = TOLERANCE_KINK.get((reference, return_period), TOLERANCE)
        assert abs(pydra - hydranl) <= tolerance, f"T = {return_period}: pydra {pydra:.4f}, Hydra-NL {hydranl:.4f}"


def test_barrier_states_sum_to_rhine_meuse_delta():
    """
    The three states of the Hollandsche IJssel barrier sum to the probability of the Rhine-Meuse delta, per state of
    the Europoort barrier and per wind direction.
    """
    settings = HRDatabase(DATABASE).get_settings(LOCATION)
    statistics = Location(settings).get_model().get_statistics()
    for wind_direction in statistics.wind_direction.get_discretisation():
        for europoort_state in (1, 2):
            states = [3 * (europoort_state - 1) + k for k in (1, 2, 3)]
            total = sum(statistics.calculate_probability(wind_direction, k) for k in states)
            expected = StatisticsLowerRiver.calculate_probability(statistics, wind_direction, europoort_state)
            np.testing.assert_allclose(total, expected, rtol=1e-12, atol=0.0)


def test_illustration_points():
    """
    The illustration points of the crest level without uncertainties, per state of the EPK and the HIJK, against the
    hydraulic loading tables of Rekenhart_HIJ ('ip_hbn_none.txt'). Every point lies within one grid step of
    Hydra-NL (MSTAP, QSTAP, 1 m/s), except the points in 'BEYOND_ONE_STEP'. Three of those are verified near ties: two
    places on the limit state with a beta within 0.04, where pydra_core and Hydra-NL choose differently.
    """
    hrd = HRDatabase(DATABASE)
    settings = hrd.get_settings(LOCATION)
    points = {
        state: HydraHollandscheIJsselIllustrationPoint(
            hrd, settings, statistical_uncertainty=False, model_uncertainty=False, response="hbn",
            profile=dike_profile(), qcrit=QCRIT, hollandsche_ijssel_state=state,
        )
        for state in (1, 2, 3)
    }
    reference = np.loadtxt(DATA / "hydranl" / "ip_hbn_none.txt", skiprows=1)

    beyond_one_step = set()
    for return_period, level, epk, hijk, wind_direction, sea_level, discharge, wind_speed in reference:
        key = (int(return_period), int(epk), int(hijk), float(wind_direction))
        design_point = points[int(hijk)].design_point(wind_direction, level, int(epk))
        assert design_point is not None, f"no illustration point for {key}"
        design = design_point["design"]
        if (
            abs(design["M"] - sea_level) > ONE_STEP["M"]
            or abs(design["Q"] - discharge) > ONE_STEP["Q"]
            or abs(design["U"] - wind_speed) > ONE_STEP["U"]
        ):
            beyond_one_step.add(key)
    assert beyond_one_step == BEYOND_ONE_STEP
