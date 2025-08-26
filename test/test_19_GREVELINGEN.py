import numpy as np

from pathlib import Path
from pydra_core import HRDatabase, ExceedanceFrequencyLine


def test_exc_freq_line_h():
    # Load the HRD
    hrd_path = (
        Path(__file__).parent
        / "data"
        / "19_GREVELINGEN"
        / "WBI2017_Grevelingen_25-4_v02.sqlite"
    )
    hrd_db = HRDatabase(hrd_path)

    # Select location 'YM_2_7-2_dk_00669'
    location = hrd_db.create_location("25-04_0001_1_GM_dp077.2")

    # Create an exc freq line for water level
    efl_h_zmonz = ExceedanceFrequencyLine("h", model_uncertainty=False)
    efl_h_monz = ExceedanceFrequencyLine("h", model_uncertainty=True)

    # Calculate
    fl_h_zmonz = efl_h_zmonz.calculate(location)
    fl_h_monz = efl_h_monz.calculate(location)

    # Compare
    _excfreqs = np.array([1 / 10, 1 / 100, 1 / 1_000, 1 / 10_000, 1 / 100_000])
    calc_h_zmonz = fl_h_zmonz.interpolate_exceedance_probability(_excfreqs)
    res_h_zmonz = np.array([0.078, 0.164, 0.259, 0.367, 0.491])
    calc_h_monz = fl_h_monz.interpolate_exceedance_probability(_excfreqs)
    res_h_monz = np.array([0.114, 0.198, 0.290, 0.394, 0.516])

    # Assert
    assert np.allclose(calc_h_zmonz, res_h_zmonz, atol=0.01)
    assert np.allclose(calc_h_monz, res_h_monz, atol=0.01)


if __name__ == "__main__":
    test_exc_freq_line_h()
