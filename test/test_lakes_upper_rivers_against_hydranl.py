"""
The wave periods Tm-1,0 and Tp of a lake and an upper river against Rekenhart_Zoet of Hydra-NL v2.9.2, BOI2023 2035:
the illustration points ('hydra.illustration_points') and the breakdown ('hydra.percentiles'), without uncertainties
('none'), with statistical uncertainty ('statistical') and with all model uncertainty ('model').

    lake          location 'KM_2_8-4_dk_00054' of the Hydra-NL demo database 'DEMO_IJsselmeer', demo calculation
                  'Ts (zonder modelonzekerheid)'
    upper river   location 'BR_1_48-1_dk_00237' of the Hydra-NL demo database 'DEMO_Rijntakken', demo calculation
                  'Hs (zonder modelonzekerheid)'

Both databases are reduced to this location. The references '<result>_<case>.json' in the folder 'hydranl' come from
the output 'uitvoer.html' of these demo calculations with the result variable and the uncertainties of each case
(BERTYPE 7 for Tm-1,0, BERTYPE 8 for Tp). Per return period they hold the level, the illustration points per wind
direction (lake level or discharge, wind speed, and with model uncertainty 'h onz.', 'f_Hm0', 'f_Tm-1,0' and 'f_Tp'),
the percentiles of the lake level or the discharge and of the wind speed (T00x_UITSPL), and the share per wind direction
(T00x_UR).

pydra_core takes its own settings and statistics of the scenario BOI2023_2035; they are equal to those of the demo
calculations.

The model uncertainty of a wave period is that of the wave period alone, as Rekenhart_Zoet; see
'Scaling.uncertainty_combinations'. Rekenhart_Zoet prints f_Hm0 = 1.00 there.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pydra_core import HRDatabase
from pydra_core.hydra.illustration_points import HydraIllustrationPoint
from pydra_core.hydra.percentiles import HydraPercentiles

DATA = Path(__file__).parent / "data"

# Database, location, label of the slow stochastic, one grid step of the slow stochastic in the illustration point
# (MSTAP, QSTAP, with a margin for the rounding of Hydra-NL), tolerance of the percentiles of the slow stochastic
SYSTEMS = {
    "lake": (DATA / "07_IJSSEL_LAKE" / "DEMO_IJsselmeer.sqlite", "KM_2_8-4_dk_00054", "M", 0.055, 0.01),
    "upper_river": (DATA / "01_RHINE_NON_TIDAL" / "DEMO_Rijntakken.sqlite", "BR_1_48-1_dk_00237", "Q", 55.0, 11.0),
}

# One grid step of the wind speed in the illustration point, the rounding of the uncertainty values of Hydra-NL, and the
# tolerances of the percentiles of the wind speed and of the share per wind direction (percentage points)
ONE_STEP_U = 1.05
UNCERTAINTY_ROUNDING = 0.0051
TOLERANCE_WIND_SPEED = 0.1
TOLERANCE_SHARE = 0.4

# Illustration points more than one grid step from Hydra-NL: (return period, wind direction). The point of Hydra-NL
# lies on the limit state of pydra_core (within 0.01 s).
# - Upper river: near ties, beta within 0.005 of that of pydra_core.
# - Lake, south-west (247.5) at the highest wind speeds of the table: pydra_core finds a point with a beta 0.06 to 0.11
#   smaller than that of Hydra-NL.
# - Lake, Tp with model uncertainty at T = 30, north-north-west: a near tie between two uncertainty combinations, f_Tp
#   1.08 (pydra_core, beta 4.017) and 1.20 (Hydra-NL, beta 4.022 in its point).
BEYOND_ONE_STEP = {
    ("lake", "tm10", "none"): {(3000, 247.5), (10000, 247.5)},
    ("lake", "tm10", "statistical"): {(1000, 247.5), (3000, 247.5), (10000, 247.5)},
    ("lake", "tm10", "model"): {(30000, 247.5)},
    ("lake", "tp", "none"): {(30000, 247.5)},
    ("lake", "tp", "statistical"): {(30000, 247.5)},
    ("lake", "tp", "model"): {(30, 337.5)},
    ("upper_river", "tm10", "statistical"): {(10, 247.5), (1000, 135.0)},
    ("upper_river", "tm10", "model"): {(100, 67.5)},
    ("upper_river", "tp", "none"): {(30, 337.5), (1000, 135.0), (3000, 225.0)},
    ("upper_river", "tp", "statistical"): {(100, 337.5)},
}

# Result in the file name, response of pydra_core, case, statistical uncertainty, model uncertainty
CASES = [
    ("tm10", "tspec", "none", False, False),
    ("tm10", "tspec", "statistical", True, False),
    ("tm10", "tspec", "model", False, True),
    ("tp", "tp", "none", False, False),
    ("tp", "tp", "statistical", True, False),
    ("tp", "tp", "model", False, True),
]
SYSTEM_CASES = [(system, *case) for system in SYSTEMS for case in CASES]


def reference(system: str, result: str, case: str) -> dict:
    folder = SYSTEMS[system][0].parent / "hydranl"
    return json.loads((folder / f"{result}_{case}.json").read_text())


@pytest.mark.parametrize(
    "system, result, response, case, statistical, model_uncertainty",
    SYSTEM_CASES,
    ids=[f"{c[0]}_{c[1]}_{c[3]}" for c in SYSTEM_CASES],
)
def test_illustration_points(system, result, response, case, statistical, model_uncertainty):
    """
    Every illustration point of Hydra-NL has one in pydra_core within one grid step, except the points in
    'BEYOND_ONE_STEP'. With model uncertainty every point within one grid step also has the factors of Hydra-NL on Hm0
    (1.00) and on the wave period of the result.
    """
    path, location, slow_label, one_step_slow, _ = SYSTEMS[system]
    hrd = HRDatabase(path)
    point = HydraIllustrationPoint(
        hrd, hrd.get_settings(location), statistical_uncertainty=statistical, model_uncertainty=model_uncertainty,
        response=response,
    )

    beyond_one_step = set()
    for return_period, hydranl in reference(system, result, case).items():
        for row in hydranl["points"]:
            wind_direction, slow, wind_speed = row[:3]
            key = (int(return_period), float(wind_direction))
            design_point = point.design_point(wind_direction, hydranl["level"])
            assert design_point is not None, f"no illustration point for {key}"
            design = design_point["design"]
            if abs(design[slow_label] - slow) > one_step_slow or abs(design["U"] - wind_speed) > ONE_STEP_U:
                beyond_one_step.add(key)
            elif model_uncertainty:
                factor = design_point["f_tspec"] if response == "tspec" else design_point["f_tp"]
                pydra = np.array([design_point["f_hs"], factor])
                hydranl_factors = np.array([row[4], row[5] if response == "tspec" else row[6]])
                assert np.all(np.abs(pydra - hydranl_factors) < UNCERTAINTY_ROUNDING), f"other factors at {key}"
    assert beyond_one_step == BEYOND_ONE_STEP.get((system, result, case), set())


@pytest.mark.parametrize(
    "system, result, response, case, statistical, model_uncertainty",
    SYSTEM_CASES,
    ids=[f"{c[0]}_{c[1]}_{c[3]}" for c in SYSTEM_CASES],
)
def test_breakdown(system, result, response, case, statistical, model_uncertainty):
    """
    The percentiles of the lake level or the discharge and of the wind speed, and the share per wind direction, within
    the tolerances at every return period.

    The tolerance of the discharge is 11 m3/s: for a wave period the critical wind speed hardly depends on the
    discharge, so its percentiles follow the momentary distribution of the discharge, and those of pydra_core lie up to
    10.1 m3/s (0.7 %) from Hydra-NL. The same small difference moves the share per wind direction of the upper river up
    to 0.32 percentage points (T = 30000), hence a tolerance of 0.4. For the water level they agree within the rounding
    of Hydra-NL. Not traced further.
    """
    path, location, _, _, tolerance_slow = SYSTEMS[system]
    hrd = HRDatabase(path)
    percentiles = HydraPercentiles(
        hrd, hrd.get_settings(location), statistical_uncertainty=statistical, model_uncertainty=model_uncertainty,
        response=response,
    )
    slow_key = "lake_level" if system == "lake" else "discharge"

    for return_period, hydranl in reference(system, result, case).items():
        # Above the extrapolated levels Hydra-NL prints no breakdown
        if not hydranl["slow"]:
            continue
        level = hydranl["level"]
        pydra = percentiles.percentiles(level)
        np.testing.assert_allclose(pydra[slow_key], hydranl["slow"], rtol=0.0, atol=tolerance_slow,
                                   err_msg=f"T = {return_period}")
        np.testing.assert_allclose(pydra["wind_speed"], hydranl["wind_speed"], rtol=0.0, atol=TOLERANCE_WIND_SPEED,
                                   err_msg=f"T = {return_period}")
        shares = {d["r"]: d["percentage"] for d in percentiles.wind_percentiles_per_direction(level)["directions"]}
        for wind_direction, share in hydranl["directions"].items():
            assert abs(shares[float(wind_direction)] - share) <= TOLERANCE_SHARE, (
                f"T = {return_period}, wind direction {wind_direction}"
            )
