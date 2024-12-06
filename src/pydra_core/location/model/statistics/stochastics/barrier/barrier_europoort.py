import numpy as np

from scipy.stats import norm

from .barrier import Barrier
from ..discharge import Discharge
from ..discrete_probability import DiscreteProbability
from ..sea_level.sea_level import SeaLevel
from ..wind_speed import WindSpeed
from ....loading.loading_model.loading_model import LoadingModel
from .....settings.settings import Settings
from ......common.interpolate import InterpStruct
from ......io.database_hr import DatabaseHR


class BarrierEuropoort(Barrier):
    def __init__(
        self,
        settings: Settings,
        wind_direction: DiscreteProbability,
        wind_speed: WindSpeed,
        sea_level: SeaLevel,
        discharge: Discharge,
    ):
        # Inherit
        super().__init__(settings, wind_direction)

        # Save statistics
        self.discharge = discharge
        self.sea_level = sea_level
        self.wind_speed = wind_speed

        # Assert barrier == 1
        if self.settings.europoort_barrier != 1:
            raise NotImplementedError("[ERROR] Barrier != 1 not implemented.")

        # Init standard variables
        self.k = [0, 11]
        self.nk = len(self.k)

        # Database connection
        with DatabaseHR(settings.database_path) as database:
            # Closing levels
            self.closing_levels = LoadingModel(None, None, ["u", "q", "r"], ["m"])
            self.closing_levels.initialise(
                database.get_closing_levels_table_europoort()
            )

        # Extend
        self.closing_levels.extend("q", [self.settings.q_min, self.settings.q_max])
        self.closing_levels.extend("u", [self.settings.u_max])

        # Prepare probabilities
        self.prepare_barrier_probability(sea_level)

    def calculate_closing_probability(
        self, wind_direction: float, closing_situation: int = None
    ) -> np.ndarray:
        # Translate to wind direction id
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)
        nqblok = len(self.discharge)
        nmblok = len(self.sea_level)

        # Bepaal de index van de richting in de sluitpeilen
        if wind_direction not in self.closing_levels.r:
            raise KeyError(
                f"[ERROR] Wind direction {wind_direction} not defined in closing levels loading."
            )

        ir = self.closing_levels.r.tolist().index(wind_direction)
        if closing_situation == 1:  # Open
            ik = 0  # 00
            sluitkans = np.zeros((len(self.wind_speed), nmblok, nqblok))
        elif closing_situation == 2:  # Closed
            ik = 1  # 11
            sluitkans = np.zeros((len(self.wind_speed), nmblok, nqblok))
        elif closing_situation is None:  # Both
            ik = slice(None)
            sluitkans = np.zeros((len(self.wind_speed), nmblok, nqblok, len(self.k)))
        else:
            raise KeyError(closing_situation)

        # De sluitpeilen (een zeewaterstand) zijn bekend per windsnelheid, windrichting en afvoer
        # Eerste worden de sluitpeilen uitgebreid voor alle windsnelheden en afvoerniveaus in de statistiek

        # Interpoleer over de windsnelheden, de eerste as van peilen
        sluitpeil = InterpStruct(
            x=self.wind_speed.get_discretisation(), xp=self.closing_levels.u
        ).interp(fp=self.closing_levels.m[:, :, ir], axis=0)

        # Interpoleer dit resultaat over de afvoeren, de tweede as van de peilen
        sluitpeil = InterpStruct(
            x=self.discharge.get_discretisation(), xp=self.closing_levels.q
        ).interp(fp=sluitpeil, axis=1)

        # Daarna wordt de sluitkans bepaald, door de zeewaterstanden bij deze windsnelheid, zeewaterstand en keringsituatie
        # te interpoleren in de berekende sluitkansen per cm.

        # De sluitkansen zijn afhankelijk van het sluitpeil, de zeewaterstand en (uiteraard) de sluitsituatie,
        # de afmetingen van self.kansen is daarmee ncmsp, nmblok, nk

        # De sluitkans moet bepaald worden per windsnelheid, zeewaterstand, afvoer en keringsituatie

        # berekend voor de in de database aanwezig windsnelheden, windrichting, en elke centimeter zeewaterstand

        # Voor elke afvoer, interpoleer de kansen
        for iq in range(nqblok):
            # bepaal de sluitkansen voor de combinatie van afvoer, kering en zeewaterstand
            if isinstance(ik, int):
                sluitkans[:, :, iq] = InterpStruct(
                    x=sluitpeil[:, iq], xp=self.cmsp
                ).interp(self.kansen[:, :, ik], axis=0)
            else:
                sluitkans[:, :, iq, :] = InterpStruct(
                    x=sluitpeil[:, iq], xp=self.cmsp
                ).interp(self.kansen[:, :, :], axis=0)

        return sluitkans

    def prepare_barrier_probability(self, sea_level: SeaLevel):
        """
        bereken het minimale sluitpeil, op cm's naar beneden afgerond

        bepaal het aantal cm's van het minimale tot het maximale sluitpeil
        alloceer geheugen

        voor elke cm tussen het minimum en maximum
            bepaal de hoogte van het sluitpeil

        voor elke keringsituatie
            bereken de keringskans per zeewaterstand en sluitpeil

        Parameters
        ----------
        belasting : type (tprand2001)
            structure met alle sluitpeilen
        statistiek : type (tpstatistiek)
            structure met waterstanden, golfparameters en hydraulische belastingen als gevolg daarvan
        """
        # bereken het minimale sluitpeil, op cm's naar beneden afgerond
        step_size = 0.01
        min_cl = np.floor(self.closing_levels.m.min() / step_size) * step_size
        max_cl = np.ceil(self.closing_levels.m.max() / step_size) * step_size

        # bepaal het aantal cm's van het minimale tot en met het maximale sluitpeil
        self.nsp = int((max_cl - min_cl) / step_size) + 1

        # alloceer geheugen
        self.kansen = np.zeros((self.nsp, len(sea_level), len(self.k)))

        # vul de vector met sluitpeilen voor elke cm
        self.cmsp = min_cl + np.arange(self.nsp) * step_size

        # Settings
        barrier = int(self.settings.europoort_barrier)
        distribution = self.settings.europoort_barrier_distribution
        mu = self.settings.europoort_barrier_mu
        sigma = self.settings.europoort_barrier_sigma
        alfa = self.settings.europoort_barrier_alfa

        # bereken de keringskans voor elke keringtoestand, zeewaterstand en sluitpeilcentimeter
        for ik in range(self.nk):
            #  bepaal kans op commando keringen dicht
            if barrier != 2:
                #  Normal distribution
                if distribution == 0:
                    s = (
                        self.cmsp[:, None]
                        - sea_level.get_discretisation()[None, :]
                        - mu
                    ) / sigma
                    kansdicht = 1.0 - norm(loc=0.0, scale=1.0).cdf(x=s)

                # Cosinus-squared distribution
                elif distribution == 1:
                    s = np.pi * sigma * (3.0 / (np.pi * np.pi - 6.0)) ** 0.5
                    kansdicht = 0.5 * (
                        1.0
                        + (
                            self.cmsp[:, None]
                            - sea_level.get_discretisation()[None, :]
                            - mu
                        )
                        / s
                        + np.sin(
                            np.pi
                            * (
                                self.cmsp[:, None]
                                - sea_level.get_discretisation()[None, :]
                                - mu
                            )
                            / s
                        )
                        / np.pi
                    )

                # Undefined distribution
                else:
                    raise ValueError("Fout type kansverdeling voorspelde zeewaterstand")

            if barrier == 1:
                if int(round(self.k[ik])) == 0:
                    self.kansen[:, :, ik] = 1.0 - (1.0 - alfa) * kansdicht
                elif int(round(self.k[ik])) == 11:
                    self.kansen[:, :, ik] = (1.0 - alfa) * kansdicht

            # Assert barrier == 1 (for now)
            else:
                raise NotImplementedError("[ERROR] Barrier != 1 not implemented.")
