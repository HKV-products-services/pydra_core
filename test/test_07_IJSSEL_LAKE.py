import numpy as np

from pathlib import Path
from pydra_core import HRDatabase, ExceedanceFrequencyLine


def test_exc_freq_line_h():
    # Load the HRD
    hrd_path = (
        Path(__file__).parent
        / "data"
        / "07_IJSSEL_LAKE"
        / "WBI2017_IJsselmeer_7-2_v02.sqlite"
    )
    hrd_db = HRDatabase(hrd_path)

    # Select location 'YM_2_7-2_dk_00669'
    location = hrd_db.create_location("YM_2_7-2_dk_00669")

    # Create an exc freq line for water level
    _levels = np.arange(0.6, 3.0, 0.1)
    efl_h_zmonz = ExceedanceFrequencyLine("h", model_uncertainty=False, levels=_levels)
    efl_h_monz = ExceedanceFrequencyLine("h", model_uncertainty=True, levels=_levels)

    # Calculate
    fl_h_zmonz = efl_h_zmonz.calculate(location)
    fl_h_monz = efl_h_monz.calculate(location)

    # Compare
    _excfreqs = np.array(
        [1 / 10, 1 / 100, 1 / 1_000, 1 / 10_000, 1 / 100_000, 1 / 1_000_000]
    )
    calc_h_zmonz = fl_h_zmonz.interpolate_exceedance_probability(_excfreqs)
    res_h_zmonz = np.array(
        [0.6012617, 0.89149811, 1.22942988, 1.59466267, 1.96410966, 2.29767858]
    )
    calc_h_monz = fl_h_monz.interpolate_exceedance_probability(_excfreqs)
    res_h_monz = np.array(
        [0.88067023, 1.22164364, 1.53385229, 1.87439753, 2.23825322, 2.59632654]
    )

    # Assert
    assert np.allclose(calc_h_zmonz, res_h_zmonz, rtol=0.001, atol=0.001)
    assert np.allclose(calc_h_monz, res_h_monz, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    test_exc_freq_line_h()
