import numpy as np
import platform

from pydra_core import Profile, Breakwater


def test_profile():
    # Only for Windows for now
    sys_platform = platform.system()
    if sys_platform != "Windows":
        return

    # Create new profile
    prof = Profile()
    prof.set_breakwater(Breakwater.CAISSON, 1.0)
    prof.set_foreland_geometry([-30, -10, 0], [-2.0, -1.0, 0.0])
    prof.set_dike_geometry([0, 6, 9, 18], [0.0, 3.0, 3.05, 6.0], [0.7, 0.8, 1])
    prof.set_dike_crest_level(5.0)
    prof.set_dike_orientation(20)

    # Wave conditions
    h = 2.0
    hs = 3.0
    tspec = 8.0
    wdir = 350

    # Calculate q_avg
    q_avg = prof.calculate_overtopping(h, hs, tspec, wdir) * 1000
    assert np.isclose(q_avg, 2.5, atol=0.1)

    # Remove foreland and calculate req crest level
    prof.remove_foreland()
    req_crest = prof.calculate_crest_level(0.01, h, hs, tspec, wdir)
    assert np.isclose(req_crest, 4.18, atol=0.01)

    # Remove breakwater and calculate ru2p
    prof.remove_breakwater()
    ru2p = prof.calculate_runup(h, hs, tspec, wdir)
    assert np.isclose(ru2p, 7.32, atol=0.01)


if __name__ == "__main__":
    test_profile()
