import numpy as np

from dataclasses import dataclass
from pathlib import Path

from ...common.interpolate import Interpolate


@dataclass
class FrequencyLine:
    """
    Represents a frequency line with associated levels and exceedance frequencies.

    Parameters
    ----------
    level : np.ndarray
        An array of levels.
    exceedance_frequency : np.ndarray
        An array of corresponding exceedance frequencies.
    """

    # Init variables
    level: np.ndarray
    exceedance_frequency: np.ndarray

    def interpolate_exceedance_probability(self, exceedance_probability: np.ndarray):
        exceedance_probability = np.atleast_1d(exceedance_probability)
        order_x = np.argsort(exceedance_probability)
        order_xp = np.argsort(self.exceedance_frequency)
        f = Interpolate.inextrp1d(
            x=np.log(exceedance_probability)[order_x],
            xp=np.log(self.exceedance_frequency)[order_xp],
            fp=self.level[order_xp],
        )
        return f[np.argsort(order_x)]

    def interpolate_level(self, level: np.ndarray):
        return np.exp(
            Interpolate.inextrp1d(
                x=level, xp=self.level, fp=np.log(self.exceedance_frequency)
            )
        )

    def to_file(self, path: Path, overwrite=False):
        if not overwrite and path.exists():
            raise OSError(
                f'Path "{path}" already exists. Choose overwrite=True or give another path.'
            )

        with path.open("w") as f:
            f.write(f"{len(self.level):5d}\n")
            for niveau, ovfreq in zip(self.level, self.exceedance_frequency):
                f.write(f"{niveau:11.6f} {ovfreq:14.7e}\n")

    def drop_zeros(self):
        idx = self.exceedance_frequency == 0.0
        if idx.all():
            raise ValueError("All exceedance frequencies are 0.")
        self.exceedance_frequency = self.exceedance_frequency[~idx]
        self.level = self.level[~idx]
