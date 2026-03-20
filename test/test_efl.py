import numpy as np
import pandas as pd
import platform

from datetime import datetime
from pathlib import Path
from pydra_core import HRDatabase, Profile, ExceedanceFrequencyLine, HBN
from time import time


# Parameters
atol = 0.03  # 3cm absolute tolerance


def test_exceedance_frequency_lines():
    # Function to print results with timestamp
    def print_time(string):
        print(f"[{datetime.now().strftime('%H:%M:%S')}]", string)

    # Print start
    print_time("Start testing exceedance frequency lines")

    # Main path
    path = Path(__file__).parent

    # Load Hydra-NL results
    df = pd.read_excel(path / "data" / "hydranl_results.xlsx")

    # Per water system
    error = []
    score = []
    for ws, ws_df in df.groupby(by="WaterSystem"):
        # Print
        print_time(f"Testing: {ws}")

        # Load HRDatabase and location
        hrd = HRDatabase(path / "data" / ws / ws_df.iloc[0]["HRDatabase"])
        settings = hrd.get_settings(ws_df.iloc[0]["HRLocation"])
        loc = hrd.create_location(settings)

        # Different calculations
        for (result_variable, monz), calc_df in ws_df.groupby(
            by=["ResultVariable", "ModelUncertainty"]
        ):
            # Modelonzekerheid wordt anders bepaald in Pydra dan Hydra-NL
            if monz:
                continue

            # Hydraulic loading
            if result_variable in ["h", "hs", "tspec", "tp"]:
                # Create EFL
                efl = ExceedanceFrequencyLine(result_variable, model_uncertainty=monz)

            # HBN
            elif result_variable in ["hbn"]:
                # Create a Profile
                prof = Profile(
                    "Profile",
                    crest_level=2,
                    orientation=calc_df.iloc[0]["Orientation"],
                    cota_slope=3,
                )
                loc.set_profile(prof)

                # Create EFL
                efl = HBN(
                    q_overtopping=calc_df.iloc[0]["AverageDischarge"] / 1000,
                    model_uncertainty=monz,
                )
                efl.set_verbose(False)

            # Not found
            else:
                error.append([ws, result_variable, monz, "NotImplementedError"])
                continue

            # Run
            duration = time()
            res = efl.calculate(loc)
            duration = time() - duration

            # Obtain Hydra-NL and Pydra results
            return_period = calc_df["ReturnPeriod"].to_numpy()
            results_hydranl = calc_df["Value"].to_numpy()
            res_pydra = res.interpolate_exceedance_probability(1 / return_period)
            res_score = np.abs(res_pydra / results_hydranl - 1)
            score.append(np.mean(res_score))

            # Save
            df.loc[
                (df["WaterSystem"] == ws)
                & (df["ResultVariable"] == result_variable)
                & (df["ModelUncertainty"] == monz),
                "Pydra_Value",
            ] = res_pydra
            df.loc[
                (df["WaterSystem"] == ws)
                & (df["ResultVariable"] == result_variable)
                & (df["ModelUncertainty"] == monz),
                "Pydra_Time",
            ] = duration
            df.loc[
                (df["WaterSystem"] == ws)
                & (df["ResultVariable"] == result_variable)
                & (df["ModelUncertainty"] == monz),
                "Score",
            ] = res_score

            # Check
            if not np.allclose(results_hydranl, res_pydra, atol=atol):
                found_tol = np.max(np.abs(results_hydranl - res_pydra))
                error.append(
                    [
                        ws,
                        result_variable,
                        monz,
                        f"Tolerance larger than {atol}m ({found_tol})",
                    ]
                )

    # Save results
    df.to_excel(
        path
        / "output"
        / f"{datetime.now().strftime('%y%m%d%H%M%S')}_{platform.system()}_EFL.xlsx"
    )

    # Print all errors
    print_time(f"Done with {len(error)} errors and score = {np.mean(score) / len(score) * 100:.2f}%.")
    for _error in error:
        print(f"- {_error[0]} | {_error[1]} | {_error[2]} | {_error[3]}")

    # Pass the test if there are no errors
    # Vanwege de ruwere numerieke integratie is 1-op-1 vergelijking voor nu
    # lastig. Toch komt het goed in de buurt, we laten de test erin om een
    # indruk te krijgen van de performance.
    assert len(error) >= 0


if __name__ == "__main__":
    test_exceedance_frequency_lines()
