import numpy as np

from ..statistics import Statistics
from ..stochastics.barrier.barrier_ramspol import BarrierRamspol
from ..stochastics.discrete_probability import DiscreteProbability
from ..stochastics.discharge import Discharge
from ..stochastics.lake_level import LakeLevel
from ..stochastics.model_uncertainty import ModelUncertainty
from ..stochastics.wind_speed import WindSpeed
from ....settings.settings import Settings
from .....common.probability import ProbabilityFunctions


class StatisticsIJsselVechtdelta(Statistics):
    """
    Statistics class for the IJssel-Vechtdelta
    Water systems: IJssel Delta, Vecht Delta
    """

    def __init__(self, settings: Settings):
        """
        Init the Statistics class

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Inherit initialisation method from parent
        super().__init__(settings)

        # Wind
        self.wind_direction = DiscreteProbability(settings.wind_direction_probability)
        self.wind_speed = WindSpeed(settings)

        # Lake level
        self.lake_level = LakeLevel(settings)

        # Discharge
        self.discharge = Discharge(settings)

        # Combined probability(density) of lake level and discharge
        self.density_aq_peak, self.density_aq, self.ovduration_aq = (
            self.__prob_combined()
        )

        # Ramspol
        self.barrier = BarrierRamspol(settings, self.wind_direction)

        # Model uncertainty
        self.model_uncertainties = ModelUncertainty(settings)

        # Discrete, slow, fast stochatics
        self.stochastics_discrete = {
            "r": self.wind_direction.get_discretisation(),
            "k": self.barrier.k,
        }
        self.stochastics_fast = {"u": self.wind_speed.get_discretisation()}
        self.stochastics_slow = {
            "a": self.lake_level.get_discretisation(),
            "q": self.discharge.get_discretisation(),
        }

    def calculate_probability(
        self, wind_direction: float, closing_situation: int = 1, given: list = []
    ):
        """
        Calculate the probability of occurence for the discretisation given the wind direction.

        Parameters
        ----------
        direction : float
            Wind direction
        closing_situation : int
            Closing situation, (irrelevant for Coast)
        given : list
            Given stochasts
        """
        # Probility wind speed
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
        kanswind = ProbabilityFunctions.probability_density(
            self.wind_speed.get_discretisation(),
            self.wind_speed.get_exceedance_probability()[:, ir],
        ).probability
        if "u" in given:
            kanswind[:] = 1.0

        # Lake level and discharge
        if ("a" in given) and ("q" in given):
            # If both are conditional, fill the matrix with ones
            kans_aq = np.ones((len(self.lake_level), len(self.discharge)))
        else:
            # If one of two is not conditional, use the momentary probability
            kans_aq = self.density_aq
            if "a" not in given:
                kans_aq[...] = ProbabilityFunctions.conditional_probability(
                    kans_aq, axis=0
                )
            if "q" not in given:
                kans_aq[...] = ProbabilityFunctions.conditional_probability(
                    kans_aq, axis=1
                )

        # Probility for wind direction
        kanswr = 1.0 if "r" in given else self.wind_direction.get_probability()[ir]

        # Combine all probabilities
        kansen = kanswind[:, None, None] * kans_aq[None, :, :] * kanswr

        # Ramspol closing probability
        keringkans = self.barrier.calculate_closing_probability(None, closing_situation)
        kansen *= keringkans

        # Swap axis
        kansen = np.swapaxes(kansen, 1, 2)

        # Return probability
        return kansen

    def __prob_combined(self) -> np.ndarray:
        """Bereken de gezamelijke kans op afvoer en meerpeil. De functie wordt zowel
        gebruikt in de VIJD als het VZM, vandaar dat deze niet aan een klasse gekoppeld is.

        Parameters
        ----------
        afvoer : afvoer.Afvoer
            Afvoerstatistiekobject
        meerpeil : meerpeil.Meerpeil
            Meerpeilstatistiekobject
        invoergeg:
            Invoergegevensobject

        Returns
        -------
        np.ndarray
            Gezamelijke kansdichtheid van meerpeil en afvoer
        """
        # Bereken de kansdichtheid van de piekafvoer
        kansafvoer = ProbabilityFunctions.probability_density(
            self.discharge.qblok, self.discharge.epqpeak
        )
        kansmeerpeil = ProbabilityFunctions.probability_density(
            self.lake_level.ablok, self.lake_level.k_apeak
        )

        ovmaxm = self.lake_level.epapeak[-1]
        sigma = self.settings.sigma_aq

        # Bereken de gezamenlijke kansdichtheid van de piekwaardes van de afvoer en het meerpeil
        assert self.lake_level.nablok == self.lake_level.napeak
        nm = self.lake_level.nablok
        assert self.discharge.nqblok == self.discharge.nqpeak
        nq = self.discharge.nqblok
        dichtheid_mqpiek = np.full((nm, nq), 0.0)

        for iq in range(nq):
            somkans = 0.0

            for im in reversed(range(nm)):
                deltam = kansmeerpeil.delta[im]

                y = self.lake_level.k_apeak[im] - self.discharge.eqpeak_exp[iq]

                dichtheid_mqpiek[im, iq] = (
                    kansmeerpeil.density[im]
                    * np.exp(-(y**2) / (2.0 * sigma**2))
                    / (sigma * (2.0 * np.pi) ** 0.5)
                )

                # Als de som over de kansen groter wordt dan 1, dan wordt voor het beschouwde, die kansmassa
                # aangehouden dat de som over de kansen gelijk wordt aan 1. De verdere blokjes krijgen geen kans.
                if (somkans <= (1.0 - ovmaxm)) & (
                    (somkans + dichtheid_mqpiek[im, iq] * deltam) >= (1.0 - ovmaxm)
                ):
                    dichtheid_mqpiek[im, iq] = (1.0 - ovmaxm - somkans) / deltam
                    somkans = 1.0 - ovmaxm
                    break

                somkans += dichtheid_mqpiek[im, iq] * deltam

            if somkans > (1.0 - ovmaxm):
                dichtheid_mqpiek[:, iq] /= somkans

            dichtheid_mqpiek[:, iq] *= kansafvoer.density[iq]

        # En de blokstat
        # Overschrijdingsduren van afvoer- en meerpeilgolf bepalen
        overschrijdingsduren_mq = (
            self.discharge.wave_shape.bepaal_gezamenlijke_overschrijding(
                golfvormen_st1=self.lake_level.wave_shape,
                niveaus_st1=self.lake_level.ablok,
                golfvormen_st2=self.discharge.wave_shape,
                niveaus_st2=self.discharge.qblok,
            )
        )

        # Momentane kansen bepalen
        dq = ProbabilityFunctions.probability_density(
            self.discharge.qpeak, self.discharge.epqpeak
        ).delta
        dm = ProbabilityFunctions.probability_density(
            self.lake_level.apeak, self.lake_level.epapeak
        ).delta
        kans_mqpiek = dichtheid_mqpiek * dm[:, None] * dq[None, :]
        ovkansmq = (kans_mqpiek[:, :, None, None] * overschrijdingsduren_mq).sum(
            (0, 1)
        ) / (self.settings.base_duration)

        # Corrigeer kleine onvolkomenheden
        ovkansmq = np.maximum.accumulate(ovkansmq[::-1])[::-1]

        tussenkans = np.zeros((nq, nm))
        for iq in range(nq):
            tussenkans[iq, :] = ProbabilityFunctions.probability_density(
                self.lake_level.ablok, ovkansmq[:, iq]
            ).probability

        # Corrigeer kleine onvolkomenheden
        tussenkans = np.maximum.accumulate(tussenkans[::-1])[::-1]

        dichtheid_mqblok = np.zeros((nm, nq))
        for im in range(nm):
            dichtheid_mqblok[im, :] = ProbabilityFunctions.probability_density(
                self.discharge.qblok, tussenkans[:, im]
            ).probability

        dichtheid_mqblok /= dichtheid_mqblok.sum()

        return dichtheid_mqpiek, dichtheid_mqblok, overschrijdingsduren_mq
