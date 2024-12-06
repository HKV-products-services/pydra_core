import numpy as np

from .base_model import BaseModel
from .loading.loading_factory import LoadingFactory
from .statistics.statistics_factory import StatisticsFactory
from ..settings.settings import Settings
from ...common.probability import ProbabilityFunctions


class WaterSystem(BaseModel):
    def __init__(self, settings: Settings):
        """
        Water System model
        """
        # Inherit
        super().__init__(settings)

        # Statistics
        self.statistics = StatisticsFactory.get_statistics(self.settings)

        # Loading
        self.loading = LoadingFactory.get_loading(self.settings)

    def process_slow_stochastics(
        self, exceedance_probability: np.ndarray, axis: int = None
    ) -> np.ndarray:
        """
        Convert probabilities given a discharge to probabilities per duration.

        This function converts probabilities represented by a discharge to
        probabilities per specified duration based on the given stochastic
        variable ('stochastic'). The probabilities are computed for each
        duration (basisduur).

        Parameters
        ----------
        exceedance_probability : np.ndarray
            Numpy array with the discharge on the first dimension.

        Returns
        -------
        np.ndarray
            Probabilities per specified duration (basisduur).
        """
        # Input stochastics
        statistics = self.get_statistics()
        slow_stochastics = list(statistics.stochastics_slow.keys())
        _axis = range(1, len(slow_stochastics) + 1) if axis is None else axis

        # Put slow variable on first axis if not already
        newpos = range(len(np.atleast_1d(_axis)))
        if newpos != _axis:
            ep_x = np.moveaxis(exceedance_probability, _axis, newpos)
        else:
            ep_x = exceedance_probability

        # Calculate exceedance frequencies per peak discharge / lake level
        ep_ks = self.process_wave_shape(ep_x)

        # Steps in probability density
        if len(slow_stochastics) == 1 and slow_stochastics[0] == "q":
            p_peak = ProbabilityFunctions.probability_density(
                statistics.discharge.qpeak, statistics.discharge.epqpeak
            ).probability

        elif len(slow_stochastics) == 1 and slow_stochastics[0] == "a":
            p_peak = ProbabilityFunctions.probability_density(
                statistics.lake_level.apeak, statistics.lake_level.epapeak
            ).probability

        elif len(slow_stochastics) == 2 and slow_stochastics == ["a", "q"]:
            dq = ProbabilityFunctions.probability_density(
                statistics.discharge.qblok, statistics.discharge.epqpeak
            ).delta
            dm = ProbabilityFunctions.probability_density(
                statistics.lake_level.ablok, statistics.lake_level.epapeak
            ).delta
            p_peak = statistics.density_aq_peak * dm[:, None] * dq[None, :]

        elif len(slow_stochastics) == 2 and slow_stochastics == ["q", "a"]:
            dq = ProbabilityFunctions.probability_density(
                statistics.discharge.qblok, statistics.discharge.epqpeak
            ).delta
            da = ProbabilityFunctions.probability_density(
                statistics.lake_level.ablok, statistics.lake_level.epapeak
            ).delta
            p_peak = statistics.density_aq_peak.T * dq[:, None] * da[None, :]

        # Reshape steps when multiplying to trapezoidal probs
        shp = p_peak.shape + (1,) * (ep_ks.ndim - p_peak.ndim)

        # Trapezoidal probability
        p_trapezoidal = (p_peak.reshape(shp) * ep_ks).sum(
            axis=tuple(range(p_peak.ndim))
        )

        return p_trapezoidal

    def process_wave_shape(self, exceedance_probability: np.ndarray) -> np.ndarray:
        """
        Process the discharge waveshapes into the exceedance probabilities
        conditioned on the discharge level. The discharge level is represented
        on the first axis, which has dimensions equal to the number of
        discharge steps in the statistics.

        Parameters
        ----------
        exceedance_probability : np.ndarray
            Numpy array containing the exceedance probabilities with the
            discharge on the first dimension.

        Returns
        -------
        blokondkans : np.ndarray
            Exceedance probabilities of the discharge per duration (basisduur).
        """
        # Input variables
        statistics = self.get_statistics()
        slow_stochastics = list(statistics.stochastics_slow.keys())
        fp = exceedance_probability

        # Result variables
        intidxs = []
        xs = []
        xps = []

        # Verzamel de interpolatieindices voor alle trage stochasten
        for i, _stochastic in enumerate(slow_stochastics):
            if _stochastic == "q":
                xp = statistics.discharge.qpeak
                x = statistics.discharge.get_wave_shape().get_wave_shapes()
                tijden = statistics.discharge.get_wave_shape().time
            elif _stochastic == "a":
                xp = statistics.lake_level.apeak
                x = statistics.lake_level.get_wave_shape().get_wave_shapes()
                tijden = statistics.lake_level.get_wave_shape().time
            else:
                raise ValueError(
                    f"Eerste trage stochast is niet 'q' of 'm'/'a' (gegeven eerste trage stochast: {_stochastic})"
                )

            # Check if the length matches
            if exceedance_probability.shape[i] != x.shape[1]:
                raise ValueError(
                    f"The number of elements in the stochast axis ({i}) should be {x.shape[1]} ({x.shape}), but is {exceedance_probability.shape[i]} ({exceedance_probability.shape})."
                )

            # Bepaal de interpolatie-indices, de xp-punten rondom x waartussen wordt geïnterpoleerd
            intidx = np.array(
                [(xp[None, :] <= ix[:, None]).sum(1) - 1 for ix in x], dtype=np.uint16
            )
            intidx = np.minimum(np.maximum(intidx, 0), len(xp) - 2)

            intidxs.append(intidx)
            xs.append(x)
            xps.append(xp)

        # Bepaal de tijdsduur van het eerste en het laatste blokje uit de golfvorm
        block_duration = tijden[1] - tijden[0]
        eindduur = tijden[-1] - tijden[-2]

        # Alloceer arrays. Het is sneller om in de loop de arrays over te schrijven dan om ze opnieuw aan te maken
        ondkanstot = np.ones(exceedance_probability.shape)
        blokondkans = np.zeros(exceedance_probability.shape)
        ovkansen = np.zeros(exceedance_probability.shape)

        nt = len(tijden)

        for it in range(nt - 1):
            # Interpoleer de overschrijdingskansen
            if len(slow_stochastics) == 1:
                # Bepaal de index van de interpolatiewaarden (afvoeren / meerpeil) voor de betreffende tijdstap
                iix = intidxs[0][it]

                # Bepaal de fracties van de fp waarden obv xp, die moet worden ingevuld voor f (obv x)
                fracs = (x[it] - xp[iix]) / (xp[iix + 1] - xp[iix])

                # Bepaal de vorm van de output array, dit is (Ntraag, Ntijd, Nderest1, Nderest2, ...)
                fracshp = fracs.shape + (1,) * (fp.ndim - 1)

                # Interpoleer alle overschrijdingskansen in één keer door de fracties met de resultaatwaarden (fp) te vermenigvuldigen
                ovkansen[:] = (1 - fracs.reshape(fracshp)) * fp[iix] + fp[
                    iix + 1
                ] * fracs.reshape(fracshp)

            elif len(slow_stochastics) == 2:
                iix1 = intidxs[0][it]
                iix2 = intidxs[1][it]

                # Bepaal de fracties van de fp waarden obv xp, die moet worden ingevuld voor f (obv x)
                fracs1 = (xs[0][it] - xps[0][iix1]) / (xps[0][iix1 + 1] - xps[0][iix1])
                fracs2 = (xs[1][it] - xps[1][iix2]) / (xps[1][iix2 + 1] - xps[1][iix2])

                # Bepaal de vorm van de output array, dit is (Ntraag, Ntijd, Nderest1, Nderest2, ...)
                frac2shp = fracs2.shape + (1,) * (fp.ndim - 2)

                # 2D interpolatie
                # Merk op dat dit via een loop gaat. Dit is sneller dan in enkele array-operaties, omdat
                # de arrays zo groot zijn dat fancy-indexing traag wordt
                for i, (ix, ixp1) in enumerate(zip(iix1, iix1 + 1)):
                    # Interpoleer eerst over de eerste stochast
                    fy1 = (1 - fracs1[i]) * fp[ix][iix2] + fp[ixp1][iix2] * fracs1[i]
                    fy2 = (1 - fracs1[i]) * fp[ix][iix2 + 1] + fp[ixp1][
                        iix2 + 1
                    ] * fracs1[i]

                    # Interpoleer de tweede stochast
                    ovkansen[i] = fy1 + (fy2 - fy1) * fracs2.reshape(frac2shp)

                # fy1 = (1 - fracs1.reshape(frac1shp)) * fp[np.ix_(iix1, iix2  )] + fp[np.ix_(iix1+1, iix2  )] * fracs1.reshape(frac1shp)
                # fy2 = (1 - fracs1.reshape(frac1shp)) * fp[np.ix_(iix1, iix2+1)] + fp[np.ix_(iix1+1, iix2+1)] * fracs1.reshape(frac1shp)

            # Door het interpoleren kunnen overschrijdingskansen gecreëerd worden, die een fractie groter zijn dan 1.0
            ovkansen[:] = np.minimum(ovkansen, 1.0)

            # De berekende overschrijdingskansen gelden voor een blokje met de duur, die gelijk is
            # aan de door de gebruiker gekozen block_duration van de wind. Voor het discretiseren van de
            # golfvormen heeft de gebuiker een andere duur opgegeven. De bovenstaande overschrijdings-
            # kansen worden hiervoor gecorrigeerd
            if block_duration != statistics.wind_speed.block_duration_wind:
                ovkansen[:] = 1.0 - (1.0 - ovkansen) ** (
                    block_duration / statistics.wind_speed.block_duration_wind
                )

            # Bereken de onderschrijdingskans voor de basisduur gebruikmakend van de overschrijdingskansen per blokje
            # Factoren zijn 1
            blokondkans[:] = 1.0 - ovkansen

            # Eerste blok is half
            if it == 0:
                blokondkans[:] **= 0.5

            # Neem het voorlaatste en laatste blok samen
            if it == (nt - 1):
                # Als de eindduur kleiner is dan een halve block_duration, dan is de factor kleiner dan 1
                if eindduur < (0.5 * block_duration):
                    factor = ((0.5 * block_duration) + eindduur) / block_duration
                    blokondkans[:] **= factor

                # Als de eindduur groter of even lang is als een halve block_duration
                else:
                    # Vermenig de block_duration ook nog met het laatste blokje
                    factor = (eindduur - (block_duration / 2.0)) / block_duration
                    blokondkans[:] *= (1.0 - ovkansen[:, :]) ** factor

            ondkanstot *= blokondkans

        # Return
        return 1 - ondkanstot
