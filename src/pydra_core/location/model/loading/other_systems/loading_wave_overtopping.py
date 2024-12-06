import numpy as np
import pandas as pd

from itertools import product

from ..loading import Loading
from ..loading_model.loading_model import LoadingModel
from ....location import Location
from ....profile.profile import Profile


class LoadingWaveOvertopping(Loading):
    """
    This Loading is used to calculate HBNs
    """

    def __init__(self, location: Location, ws_range: np.ndarray):
        """
        Init the Loading object for the Waves Overtopping

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit the from parent
        super().__init__(location.get_settings())

        # Init
        self.ws_loading = location.get_model().get_loading()
        self.ws_range = ws_range
        self.read_loading()

    def read_loading(self) -> None:
        """
        Read the HR and create Loading Models
        """
        # Bepaal bij welke waterstanden de golfcondities afgeleid moeten worden
        # Wanneer het aantal unieke waterstanden kleiner is dan het aantal waterstanden
        # in het opgegeven bereik, worden de unieke waterstanden gebruikt
        ws_per_r = self.bepaal_kh_waterstanden(self.ws_loading, self.ws_range)

        # Bepaal de golfcondities bij deze waterstanden
        for richting, waterstanden in ws_per_r.items():
            # Leidt golfcondities af voor een bepaalde richting
            golfcond_r, windsnelheden = self.ws_loading.get_wave_conditions(
                richting, waterstanden, extrapolate=True
            )

            # Create a DataFrame
            df = pd.DataFrame(columns=["wlev", "u"] + list(golfcond_r.keys()))

            # Iterate over the windspeed and waterlevel arrays
            ws_u = np.array(list(product(waterstanden, windsnelheden)))
            golf = np.array(
                [golfcond_r[key].ravel() for key in list(golfcond_r.keys())]
            ).T
            df = pd.DataFrame(
                np.concatenate((ws_u, golf), axis=1),
                columns=["wlev", "u"] + list(golfcond_r.keys()),
            )

            # Create a Loading Model
            model = LoadingModel(richting, None, ["wlev", "u"], list(golfcond_r.keys()))
            model.initialise(df)

            # Save the model
            self.model[(richting, model.closing_situation)] = model

        # Extend the loading models
        self._extend_loadingmodels()

    def calculate_hbn(
        self,
        profile: Profile,
        qcrit: float = 10,
        factor_hs: float = 1.0,
        factor_tspec: float = 1.0,
    ) -> None:
        """
        Add hbn result variables to each of the LoadingModels.
        If 'hbn' is already defined, it will overwrite the old result variable.

        Parameters
        ----------
        profile : Profile
            The profile
        qcrit : float
            The critical discharge
        factor_hs : float
            Factor for the significant wave height, used for model uncertainty
        factor_tspec : float
            Factor for the spectral wave period, used for model uncertainty
        """
        for _, model in self.iter_models():
            model.calculate_hbn(profile, qcrit, factor_hs, factor_tspec)

    def bepaal_kh_waterstanden(self, ws_belasting, ws_range):
        """
        Bepaal waterstanden waarvoor kruinhoogtes berekend moeten worden.

        Per richting wordt bepaald of het aantal unieke waterstanden
        kleiner is dan het aantal waterstanden in een gediscretiseerde range.

        Als dit het geval is, kies de unieke waterstanden. Zo niet, gebruik de range.
        """
        ws_per_r = {r: [] for r in ws_belasting.r}
        for (richting, _), model in ws_belasting.iter_models():
            ws_per_r[richting] += np.unique(model.h).tolist()

        for richting, waterstanden in ws_per_r.items():
            h_all = np.array(waterstanden)[
                np.unique(np.round(waterstanden, 3), return_index=True)[1]
            ]
            # Als er minder waterstanden voorkomen bij deze richting, dan de voorgenomen
            # range, kies dan deze waterstanden
            if len(h_all) <= len(ws_range):
                ws_per_r[richting] = h_all
            else:
                ws_per_r[richting] = ws_range

        return ws_per_r
