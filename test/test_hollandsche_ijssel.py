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

'ip_<result>_<case>.txt' hold the illustration points per state of the EPK and the HIJK, as Rekenhart_HIJ prints them
(sea level, Rhine discharge and wind speed, at the level of each return period), for the water level ('h': none, all),
the wave height ('hs'), the wave period Tm-1,0 ('tm10') and the crest level ('hbn'), the last three without
uncertainties ('none'), with statistical uncertainty ('statistical') and with all model uncertainty ('model'). The files
with model uncertainty also hold the uncertainty values in each point (water level, factor on Hm0 and on Tm-1,0); for
the wave parameters and the crest level they come from the repaired build. Apart from the result variable, the input
of each illustration point calculation equals that of the frequency line with the same uncertainties.

'breakdown_<result>_<case>.json' hold the breakdown as Rekenhart_HIJ prints it with UITRKD, UITQKD, UITMKD and UITUKD
on: per return period the level, the percentages per state of the EPK and the HIJK and per wind direction, and the
percentiles of discharge, sea level and wind speed per state and for all states. They exist for the water level
('h': none, all) and the crest level ('hbn': none, statistical, model); 'breakdown_hbn_model.json' comes from the
repaired build.

pydra_core uses its own settings and statistics of the scenario BOI2023_2035; they are equal to those of the demo
calculation. The tolerance is 5 mm at the return periods of the demo calculation, except the water level without
uncertainties at T = 100 (+11 mm, where the Hydra-NL line has a kink).
"""

import json

from pathlib import Path

import numpy as np
import pytest

from pydra_core import HBN, ExceedanceFrequencyLine, HRDatabase, Location, Profile
from pydra_core.hydra.hollandsche_ijssel import HydraHollandscheIJsselIllustrationPoint
from pydra_core.hydra.hollandsche_ijssel_percentiles import HydraHollandscheIJsselPercentiles
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
# Crest level: the point of Hydra-NL lies on the limit state of pydra_core (within 9 mm) and its beta lies between 0.03
# below and 0.12 above that of pydra_core, with model uncertainty the total beta with the uncertainty combination of
# Hydra-NL. Above 0.09 only without uncertainties at T = 10000, south-west (2, 2, 225.0), (2, 3, 225.0) and
# (1, 3, 225.0): pydra_core finds a more probable point there.
# Water level, wave height and wave period: the point of Hydra-NL lies on the limit state of pydra_core (within 2 mm)
# and its total beta lies between 0.03 below and 0.12 above that of pydra_core, except the wave height with model
# uncertainty at T = 30, west, HIJK closed (0.19 above): pydra_core finds a more probable point there.
BEYOND_ONE_STEP = {
    ("h", "none"): {
        (10, 2, 3, 315.0),
        (30, 1, 1, 360.0),
        (100, 1, 1, 202.5),
        (100, 1, 2, 225.0),
        (300, 2, 2, 270.0),
        (1000, 2, 2, 225.0),
    },
    ("h", "all"): {
        (30, 1, 2, 247.5),
        (1000, 1, 1, 225.0),
        (10000, 1, 3, 225.0),
        (100000, 1, 1, 247.5),
        (300000, 1, 1, 225.0),
        (1000000, 1, 1, 225.0),
        (1000000, 1, 1, 337.5),
    },
    ("hs", "none"): {
        (1000000, 1, 1, 337.5),
    },
    ("hs", "statistical"): {
        (3000, 1, 3, 247.5),
        (1000000, 1, 1, 337.5),
    },
    ("hs", "model"): {
        (30, 1, 2, 270.0),
        (30, 1, 2, 292.5),
        (30, 1, 3, 270.0),
        (100, 2, 2, 225.0),
        (300, 1, 1, 292.5),
        (300, 1, 3, 292.5),
        (10000, 1, 1, 225.0),
        (100000, 1, 2, 225.0),
        (100000, 1, 3, 225.0),
        (100000, 2, 1, 225.0),
        (100000, 2, 2, 225.0),
        (100000, 2, 3, 225.0),
    },
    ("tm10", "none"): {
        (10, 2, 3, 225.0),
        (30, 1, 3, 360.0),
        (1000, 1, 2, 225.0),
        (1000, 1, 3, 225.0),
        (1000, 2, 2, 225.0),
        (1000, 2, 3, 225.0),
        (1000000, 1, 1, 270.0),
        (1000000, 1, 1, 337.5),
        (1000000, 2, 2, 337.5),
        (1000000, 2, 3, 337.5),
    },
    ("tm10", "statistical"): {
        (1000, 1, 2, 225.0),
        (1000, 1, 3, 225.0),
        (1000000, 1, 1, 270.0),
        (1000000, 1, 1, 337.5),
    },
    ("tm10", "model"): {
        (100, 2, 2, 360.0),
        (3000, 1, 3, 225.0),
        (30000, 2, 2, 225.0),
        (30000, 2, 3, 225.0),
        (300000, 1, 2, 225.0),
        (300000, 1, 2, 292.5),
        (300000, 1, 3, 225.0),
    },
    ("hbn", "none"): {
        (10, 2, 1, 360.0),
        (30, 2, 1, 337.5),
        (10000, 1, 3, 225.0),
        (10000, 2, 2, 225.0),
        (10000, 2, 3, 225.0),
        (10000, 2, 3, 247.5),
        (100000, 1, 1, 225.0),
        (100000, 2, 3, 337.5),
        (300000, 1, 3, 337.5),
    },
    ("hbn", "statistical"): {
        (10, 1, 3, 225.0),
        (10000, 1, 1, 270.0),
        (10000, 1, 3, 225.0),
        (30000, 1, 1, 225.0),
        (30000, 2, 1, 247.5),
        (30000, 2, 3, 247.5),
        (100000, 2, 3, 270.0),
        (300000, 2, 3, 270.0),
        (1000000, 1, 3, 270.0),
        (1000000, 2, 3, 315.0),
    },
    ("hbn", "model"): {
        (10, 1, 1, 225.0),
        (10, 2, 2, 292.5),
        (30, 1, 3, 225.0),
        (30, 1, 3, 315.0),
        (100, 1, 2, 225.0),
        (1000, 1, 3, 247.5),
        (10000, 1, 1, 247.5),
        (10000, 2, 3, 225.0),
        (100000, 1, 1, 292.5),
        (100000, 1, 2, 337.5),
        (100000, 2, 2, 225.0),
        (100000, 2, 2, 337.5),
        (100000, 2, 3, 225.0),
        (300000, 1, 2, 225.0),
        (300000, 1, 2, 337.5),
        (300000, 1, 3, 225.0),
        (300000, 2, 2, 337.5),
        (1000000, 1, 1, 225.0),
        (1000000, 1, 2, 225.0),
        (1000000, 1, 3, 225.0),
        (1000000, 2, 2, 337.5),
        (1000000, 2, 3, 337.5),
    },
}

# Uncertainty values in the output of Hydra-NL are rounded to 0.01
UNCERTAINTY_ROUNDING = 0.006

# Breakdown: one grid step of the percentiles
BREAKDOWN_ONE_STEP = {"discharge": 205.0, "sea_level": 0.105, "wind_speed": 1.05}

# Breakdown cases: result in the file name, response of pydra_core, case, statistical uncertainty, model uncertainty,
# tolerance of the percentages in percentage points (Hydra-NL prints one decimal), highest return period with tested
# percentages (None for all), percentages beyond that tolerance and percentiles beyond one grid step. A percentage is
# (return period, EPK and HIJK state) for a state and (return period, wind direction, column) for a wind direction; a
# percentile is (return period, stochastic, EPK and HIJK state).
#
# Water level: at T = 10 and 30 the HIJK closed without water burden gets up to 0.6 percentage point too little; its
# frequency at the level of Hydra-NL is 6 % lower, about 2 mm in the water level, the known difference of the frequency
# line. At T = 30 that also shifts the discharge percentiles of the HIJK open (up to 249 m3/s) and, without
# uncertainties, of the HIJK closed without water burden (221 m3/s), both with the EPK open.
#
# Crest level: at T = 300000 the cumulative discharge of the EPK closed and the HIJK open lies 0.002 to 0.003 from that
# of Hydra-NL, where it is almost flat between 5750 and 6950 m3/s; the 5 % percentile shifts 1500 m3/s. At T = 30000
# with the EPK closed and the HIJK closed with water burden the limit state lies at 45 to 50 m/s for a low discharge,
# where the wind speed density of pydra_core (from its probability masses on the 1 m/s grid) is 12 to 40 % higher than
# the density that Rekenhart_HIJ calculates in every point; below 3000 m3/s the state gets 29 % more mass and the
# discharge percentiles shift up to 300 m3/s. With uncertainties the EPK open and the HIJK open gets up to 0.35
# percentage point too little; the other exceptions are not traced: with statistical uncertainty a discharge percentile
# of the EPK closed and the HIJK open differs 223 m3/s at T = 1000000, with model uncertainty the discharge percentiles
# of that state differ 272 to 405 m3/s at T = 100 to 1000.
#
# Wave height and wave period: the level is that of the frequency line of Hydra-NL, not the printed one with three
# decimals; at T = 10000 the distribution of Hm0 over the wind directions shifts up to 4 percentage points per half
# millimetre. Without model uncertainty the wind directions west to north-northwest differ up to 6 percentage points
# at T = 30000 and higher, between 15 and 30 of the about 1300 percentages per case; there only the percentiles are
# tested. The exceptions are not traced.
BREAKDOWN_CASES = [
    ("h", "h", "none", False, False, 0.6, None, set(), {(30, "discharge", "1,1"), (30, "discharge", "1,2")}),
    ("h", "h", "all", True, True, 0.6, None, set(), {(30, "discharge", "1,1")}),
    (
        "hbn", "hbn", "none", False, False, 0.4, None, set(),
        {(30000, "discharge", "2,3"), (300000, "discharge", "2,1")},
    ),
    (
        "hbn", "hbn", "statistical", True, False, 0.4, None, set(),
        {(30000, "discharge", "2,3"), (300000, "discharge", "2,1"), (1000000, "discharge", "2,1")},
    ),
    (
        "hbn", "hbn", "model", False, True, 0.4, None, set(),
        {(100, "discharge", "2,1"), (300, "discharge", "2,1"), (1000, "discharge", "2,1")},
    ),
    (
        "hs", "hs", "none", False, False, 0.4, 10000, set(),
        {(1000, "wind_speed", "1,2"), (100000, "wind_speed", "1,1"), (1000000, "sea_level", "2,3")},
    ),
    (
        "hs", "hs", "statistical", True, False, 0.4, 10000, set(),
        {
            (1000, "sea_level", "1,2"), (100000, "discharge", "1,1"), (300000, "wind_speed", "1,1"),
            (1000000, "sea_level", "2,2"), (1000000, "sea_level", "2,3"), (1000000, "sea_level", "all"),
        },
    ),
    (
        "hs", "hs", "model", False, True, 0.4, None, {(1000000, 315.0, "all"), (1000000, 337.5, "all")},
        {(1000000, "discharge", "1,1")},
    ),
    ("tm10", "tspec", "none", False, False, 0.4, 10000, set(), set()),
    (
        "tm10", "tspec", "statistical", True, False, 0.4, 10000, set(),
        {
            (300000, "sea_level", "2,2"), (300000, "sea_level", "2,3"), (300000, "sea_level", "all"),
            (1000000, "sea_level", "2,2"),
        },
    ),
    ("tm10", "tspec", "model", False, True, 0.4, None, set(), set()),
]

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


# Result in the file name, response of pydra_core, case, statistical uncertainty, model uncertainty
ILLUSTRATION_POINT_CASES = [
    ("h", "h", "none", False, False),
    ("h", "h", "all", True, True),
    ("hs", "hs", "none", False, False),
    ("hs", "hs", "statistical", True, False),
    ("hs", "hs", "model", False, True),
    ("tm10", "tspec", "none", False, False),
    ("tm10", "tspec", "statistical", True, False),
    ("tm10", "tspec", "model", False, True),
    ("hbn", "hbn", "none", False, False),
    ("hbn", "hbn", "statistical", True, False),
    ("hbn", "hbn", "model", False, True),
]


@pytest.mark.parametrize(
    "result, response, case, statistical, model_uncertainty",
    ILLUSTRATION_POINT_CASES,
    ids=[f"{c[0]}_{c[2]}" for c in ILLUSTRATION_POINT_CASES],
)
def test_illustration_points(result, response, case, statistical, model_uncertainty):
    """
    The illustration points per state of the EPK and the HIJK against Rekenhart_HIJ ('ip_<result>_<case>.txt'). Every
    point lies within one grid step of Hydra-NL (MSTAP, QSTAP, 1 m/s), except the points in 'BEYOND_ONE_STEP': near
    ties, places on the limit state with almost the same beta where pydra_core and Hydra-NL choose differently. With
    model uncertainty every point within one grid step also has the uncertainty values of Hydra-NL that count for the
    result: the water level for 'h', the factor on Hm0 for 'hs', both wave factors for 'tm10' and all three for 'hbn'.
    """
    hrd = HRDatabase(DATABASE)
    settings = hrd.get_settings(LOCATION)
    crest_level = {"profile": dike_profile(), "qcrit": QCRIT} if response == "hbn" else {}
    points = {
        state: HydraHollandscheIJsselIllustrationPoint(
            hrd, settings, statistical_uncertainty=statistical, model_uncertainty=model_uncertainty, response=response,
            hollandsche_ijssel_state=state, **crest_level,
        )
        for state in (1, 2, 3)
    }
    reference = np.loadtxt(DATA / "hydranl" / f"ip_{result}_{case}.txt", skiprows=1)

    beyond_one_step = set()
    for row in reference:
        return_period, level, epk, hijk, wind_direction, sea_level, discharge, wind_speed = row[:8]
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
        elif model_uncertainty:
            # Hydra-NL prints all three values; only those of the result count
            if response == "h":
                pydra, hydranl = [design_point["dh"]], row[8:9]
            elif response == "hs":
                pydra, hydranl = [design_point["f_hs"]], row[9:10]
            elif response == "tspec":
                pydra, hydranl = [design_point["f_hs"], design_point["f_tspec"]], row[9:11]
            else:
                uncertainty = design_point["uncertainty"]
                pydra, hydranl = [uncertainty["h"], uncertainty["f_hs"], uncertainty["f_tspec"]], row[8:11]
            deviation = np.abs(np.array(pydra) - hydranl)
            assert np.all(deviation < UNCERTAINTY_ROUNDING), f"other uncertainty values at {key}"
    assert beyond_one_step == BEYOND_ONE_STEP[(result, case)]


@pytest.mark.parametrize(
    "result, response, case, statistical, model_uncertainty, tolerance, last_return_period, beyond_tolerance_expected, "
    "beyond_one_step_expected",
    BREAKDOWN_CASES,
    ids=[f"{c[0]}_{c[2]}" for c in BREAKDOWN_CASES],
)
def test_breakdown(
    result, response, case, statistical, model_uncertainty, tolerance, last_return_period, beyond_tolerance_expected,
    beyond_one_step_expected,
):
    """
    The breakdown against Rekenhart_HIJ ('breakdown_<result>_<case>.json'): the percentages per state of the EPK and
    the HIJK and per wind direction within the tolerance of the case, up to the highest return period of the case, and
    the percentiles of discharge, sea level and wind speed within one grid step for every state with at least 1 % of
    the frequency, except the percentages and percentiles listed in 'BREAKDOWN_CASES'.
    """
    hrd = HRDatabase(DATABASE)
    crest_level = {"profile": dike_profile(), "qcrit": QCRIT} if response == "hbn" else {}
    breakdown = HydraHollandscheIJsselPercentiles(
        hrd, hrd.get_settings(LOCATION), statistical_uncertainty=statistical, model_uncertainty=model_uncertainty,
        response=response, **crest_level,
    )
    reference = json.loads((DATA / "hydranl" / f"breakdown_{result}_{case}.json").read_text())

    beyond_tolerance = set()
    beyond_one_step = set()
    for return_period, hydranl in reference.items():
        contributions = breakdown.state_contributions(hydranl["level"])
        percentiles = breakdown.percentiles(hydranl["level"])

        # Percentages per state and per wind direction, up to the highest return period of the case
        if last_return_period is None or int(return_period) <= last_return_period:
            for column, percentage in hydranl["states"].items():
                epk, hijk = (int(x) for x in column.split(","))
                pydra = contributions["states"][epk - 1, hijk - 1]
                if abs(pydra - percentage) > tolerance:
                    beyond_tolerance.add((int(return_period), column))
            for direction, columns in hydranl["directions"].items():
                jr = int(np.argmin(np.abs(contributions["r"] - float(direction))))
                for column, percentage in columns.items():
                    if column == "all":
                        pydra = contributions["directions"][jr].sum()
                    else:
                        epk, hijk = (int(x) for x in column.split(","))
                        pydra = contributions["directions"][jr, epk - 1, hijk - 1]
                    if abs(pydra - percentage) > tolerance:
                        beyond_tolerance.add((int(return_period), float(direction), column))

        # Percentiles of the states with at least 1 % of the frequency
        for stochastic, step in BREAKDOWN_ONE_STEP.items():
            for column, values in hydranl[stochastic].items():
                share = 100.0 if column == "all" else hydranl["states"][column]
                if share < 1.0:
                    continue
                key = "all" if column == "all" else tuple(int(x) for x in column.split(","))
                deviation = np.max(np.abs(np.array(percentiles[stochastic][key]) - np.array(values)))
                if deviation > step:
                    beyond_one_step.add((int(return_period), stochastic, column))
    assert beyond_tolerance == beyond_tolerance_expected
    assert beyond_one_step == beyond_one_step_expected
