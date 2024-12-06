import numpy as np

from ..statistics import Statistics
from ..stochastics.barrier.barrier_europoort import BarrierEuropoort
from ..stochastics.discharge import Discharge
from ..stochastics.discrete_probability import DiscreteProbability
from ..stochastics.model_uncertainty import ModelUncertainty
from ..stochastics.sea_level.sea_level_lower_river import SeaLevelLowerRiver
from ..stochastics.wind_speed import WindSpeed
from ....settings.settings import Settings
from .....common.interpolate import Interpolate
from .....common.probability import ProbabilityFunctions


class StatisticsLowerRiver(Statistics):
    """
    Statistics class for the Coast
    Water systems: Coast (North, Central, South), Waddensea (West, East) and Western scheldt
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

        # Sea level
        self.sea_level = SeaLevelLowerRiver(settings)

        # Wind
        self.wind_direction = DiscreteProbability(settings.wind_direction_probability)
        self.wind_speed = WindSpeed(settings)

        # Discharge
        self.discharge = Discharge(settings)

        # Europoort Barrier
        self.barrier = BarrierEuropoort(
            settings,
            self.wind_direction,
            self.wind_speed,
            self.sea_level,
            self.discharge,
        )

        # Model uncertainty
        self.model_uncertainties = ModelUncertainty(settings)

        # Discrete, slow, fast stochatics
        self.stochastics_discrete = {
            "r": self.wind_direction.get_discretisation(),
            "k": self.barrier.k,
        }
        self.stochastics_fast = {
            "u": self.wind_speed.get_discretisation(),
            "m": self.sea_level.get_discretisation(),
        }
        self.stochastics_slow = {"q": self.discharge.get_discretisation()}

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
        # Sector West: Discharge, sea level and Europoort barrier
        if ((wind_direction >= 0.0) and (wind_direction <= 11.25)) or (
            (wind_direction > 212.75) and (wind_direction <= 360.0)
        ):
            if "q" in given:
                kansafvoer = np.ones_like(self.discharge.get_discretisation())
            else:
                # Note: use the momentary probability of the discharge
                kansafvoer = ProbabilityFunctions.probability_density(
                    self.discharge.get_discretisation(),
                    1 - self.discharge.get_exceedance_probability(),
                ).probability

            # Kans op windrichting
            ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
            kanswr = 1.0 if "r" in given else self.wind_direction.get_probability()[ir]

            # Bepaal sluitkansen voor de windrichting
            sluitkans = self.barrier.calculate_closing_probability(
                wind_direction, closing_situation
            )
            if "k" in given:
                sluitkans[:] = 1.0

            # Overschrijdingskans windsnelheid
            ovkansu_m = self.ovkansu_m(wind_direction)
            kanswind = np.array(
                [
                    ProbabilityFunctions.probability_density(
                        self.wind_speed.get_discretisation(), ovkansu_m[:, im]
                    ).probability
                    for im in range(len(self.sea_level))
                ]
            ).T[:, :, None]
            if "u" in given:
                kanswind[:] = 1.0

            # Kans zeewaterstand
            kanszws = ProbabilityFunctions.probability_density(
                self.sea_level.get_discretisation(),
                self.sea_level.get_exceedance_probability()[:, ir],
            ).probability[None, :, None]
            if "m" in given:
                kanszws[:] = 1.0

            # Combineer alle kansen
            comb = kanswind * kanszws * kansafvoer[None, None, :] * sluitkans * kanswr
            return np.swapaxes(comb, 1, 2)

        # Otherwise, sector east (without sea level and Europoort)
        else:
            # Kans windsnelheid
            ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
            kanswind = ProbabilityFunctions.probability_density(
                self.wind_speed.get_discretisation(),
                self.wind_speed.get_exceedance_probability()[:, ir],
            ).probability[:, None]
            if "u" in given:
                kanswind[:] = 1.0

            # Afvoer
            if "q" in given:
                kansafvoer = np.ones_like(self.discharge.get_discretisation())
            else:
                # Let op! gebruikt de MOMENTANE kans van een afvoer
                kansafvoer = ProbabilityFunctions.probability_density(
                    self.discharge.get_discretisation(),
                    1 - self.discharge.get_exceedance_probability(),
                ).probability

            # Kans op windrichting
            kanswr = 1.0 if "r" in given else self.wind_direction.get_probability()[ir]

            # Combineer alle kansen
            return kanswind * kansafvoer[None, :] * kanswr

    def ovkansu_m(self, wind_direction: float):
        """
        Bereken de overschrijdingskans van de windsnelheid volgens het
        correlatiemodel met de

        Parameters
        ----------
        richting : float
            Windrichting

        Returns
        -------
        ondkans : np.ndarray
            2D-array per windsnelheid en zeewaterstand
        """
        #  Bepaal transformatiewaarde van de windsnelheid voor het juiste rooster van windsnelheden
        if self.settings.transitional_wind == 0:
            kru = self.bereken_kru_polynoom(wind_direction)
        elif self.settings.transitional_wind == 1:
            kru = self.bereken_kru_tabel(wind_direction)
        else:
            raise ValueError(self.settings.transitional_wind)

        #  Bereken de windsnelheidkansen per richting en zeewaterstand voor de windsnelheid in de statistiek
        return 1.0 - self.ondkanswindsnelheid(
            wind_direction, kru, self.sea_level.get_discretisation()
        )

    def ondkanswindsnelheid(self, wind_direction, kru, m):
        """
        Bepaling van de onderschrijdingskans van de windsnelheid gegeven de
        zeewaterstand en de windrichting

        Parameters
        ----------
        wind_direction : float
            Windrichting
        kru : np.ndarray
            Transformatiewaarde van de windsnelheid

        Returns
        -------
        kans : np.array [len(kru), len(m)]
            Onderschrijdingskans van de windsnelheid gegeven de zeewaterstand en de windrichting
        """
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
        windparams = self.sea_level.pwinds[:, ir]

        # Bepaal de onderschrijdingskans van de windsnelheid uit de
        # Gumbelverdeelde getransformeerde waarde van de windsnelheid
        alpha = (
            windparams[5]
            * (m + self.sea_level.translation_m - windparams[0])
            / windparams[1]
        )
        ondkans = np.exp(-np.exp((-kru[:, None] + alpha[None, :]) / windparams[6]))

        # Afknotten van de Gumbelverdeling
        ondkans = np.minimum(1.0, (1.0 / (1.0 - self.settings.fu)) * ondkans)

        return ondkans

    def bereken_kru_tabel(self, wind_direction: float):
        """
        Berekenen van de transformatietabel van de windsnelheid

                Creeer eerst een vector met waarden voor het hele bereik van K_r(u)
        Ken vectorafmetingen toe aan parameters
        Alloceer geheugen voor het hele bereik van K_r(u)
        Maak lokaal een vector met zeewaterstanden
        Alloceer geheugen voor de zeewaterstanden
        Voor alle windsnelheden
        {
        Voor alle zeewaterstanden
        {
            Bepaal de onderschrijdingskans van K_r(u) gegeven de zeewaterstand
                en de windrichting
        }
        Bereken de integraal over de zeewaterstanden
        Verwijder de K_r(u)'s met onderschijdingskansen, die bij lagere waardes
            ook al voorkomen
        Voor kleine onder- en overschrijdingskansen de K_r(u) berekenen met
            extrapolatie
        }
        Geef het gealloceerde geheugen weer vrij
        """
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)

        #  Creeer eerst een vector met waarden voor het hele bereik van K_r(u)
        kru = np.arange(-3.0, 20.001, 0.2)

        #  Maak lokaal een vector met zeewaterstanden
        m_max = max(7.0, self.sea_level.get_discretisation()[-1])
        m = np.linspace(0.75, m_max, int(round((m_max - 0.75) / 0.05)) + 1)

        # Bepaal de onderschrijdingskans van K_r(u) gegeven de zeewaterstand en de windrichting

        # Bepaal de overschrijdingskansen van de zeewaterstand gegeven de windrichting op het
        # juiste rooster (N.B. kansdichtheden worden niet geenterpoleerd##)
        # TODO: logaritmisch?
        ovkanszws = Interpolate.inextrp1d(
            x=m,
            xp=self.sea_level.get_discretisation(),
            fp=self.sea_level.get_exceedance_probability()[:, ir],
        )
        ovkanszws[ovkanszws > 1.0] = 1.0
        ovkanszws = np.maximum.accumulate(ovkanszws[::-1])[::-1]

        # Bereken de kansdichtheid uit de overschrijdingskansen
        kanszws = ProbabilityFunctions.probability_density(m, ovkanszws).probability

        # Bereken de integraal over de zeewaterstanden
        ondkans_y_r = (
            self.ondkanswindsnelheid(wind_direction, kru, m) * kanszws[None, :]
        ).sum(1)

        # Verwijder de K_r(u)'s met onderschijdingskansen, die bij lagere waardes ook al voorkomen
        ondkans_uniek, idx = np.unique(ondkans_y_r, return_index=True)
        kru_uniek = kru[idx]

        # Voor kleine onder- en overschrijdingskansen de K_r(u) berekenen met extrapolatie
        # Bepaal eerste het gebied dat BINNEN deze grenzen valt
        ovkans_u = self.wind_speed.get_exceedance_probability()[:, ir]
        idx = (ovkans_u >= 1.0e-7) & (ovkans_u <= 1 - 1.0e-8)

        # Alloceer nieuwe array voor kru
        kru = np.zeros_like(ovkans_u)

        # Interpoleer dit gebied
        kru[idx] = Interpolate.inextrp1d(
            x=1.0 - ovkans_u[idx], xp=ondkans_uniek, fp=kru_uniek
        )

        # Extrapoleer de andere waarden hier omheen
        kru[~idx] = Interpolate.inextrp1d(
            x=self.wind_speed.get_discretisation()[~idx],
            xp=self.wind_speed.get_discretisation()[idx],
            fp=kru[idx],
        )

        return kru

    def bereken_kru_polynoom(self, wind_direction: float):
        """
        Bepaal transformatiewaarden van de windsnelheden

        Parameters
        ---------

        Returns
        -------
        kru
            Vector met transformatiewaarden voor de windsnelheid op het gewenste rooster van de windsnelheden
        """
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
        windparams = self.sea_level.pwinds[:, ir]

        #  Bepaal transformatiewaarden van de windsnelheden
        kru = (
            windparams[2] * self.wind_speed.get_discretisation() ** 2
            + windparams[3] * self.wind_speed.get_discretisation()
            + windparams[4]
        )

        return kru
