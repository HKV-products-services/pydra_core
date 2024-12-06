import numpy as np

from .barrier import Barrier
from ..discrete_probability import DiscreteProbability
from ..sea_level.sea_level import SeaLevel
from ..wind_speed import WindSpeed
from ....loading.loading_model.loading_model import LoadingModel
from .....settings.settings import Settings
from ......common.interpolate import InterpStruct, Interpolate
from ......io.database_hr import DatabaseHR
from ......io.file_hydranl import FileHydraNL


class BarrierEasternScheldt(Barrier):
    def __init__(
        self,
        settings: Settings,
        wind_direction: DiscreteProbability,
        wind_speed: WindSpeed,
        sea_level: SeaLevel,
    ):
        # Inherit
        super().__init__(settings, wind_direction)

        # Lees kansen op sluitscenario's: de kans op bemand of onbemand
        self.t = np.array([0.0, 1.0])

        # pfk bevat de kans op een aantal falende schuiven (bijv 0, 16, 31, 62) voor
        # een bemande dan wel onbemande sluiting, gegeven dat de kering faalt dus
        self.k, self.p_fail_k = FileHydraNL.read_file_ncolumns(
            settings.barrier_scenario_probability
        )
        self.k = self.k.astype(int)

        # Lees kansen op sluitscenario's: de kans op sluiting gegeven de zeewaterstand en aantal falende sluizen
        self.n_closing = len(self.k)
        self.nk = len(self.t) * len(self.k)

        # sluitkansenk bevat de kans op een sluitvraag (al dan niet succesvol) gegeven de zeewaterstand
        # Voor een stategiesluiting, handmatige sluiting, noodsluiting door het systeem, of geen sluiting
        self.m_closing, self.p_closing_k = FileHydraNL.read_file_ncolumns(
            settings.barrier_closing_probability
        )

        # Database connection
        with DatabaseHR(settings.database_path) as database:
            # Read closing situations
            self.closing_situations = database.get_closing_situations_eastern_scheldt()

            # Maak sluitpeilen structure (u, m, r, d, p)
            self.closing_levels = LoadingModel(
                None, None, ["u", "m", "r", "d", "p"], ["h_rpb"]
            )
            self.closing_levels.initialise(
                database.get_closing_levels_table_eastern_scheldt()
            )

        # h_roompot_buiten bevat de waterstand bij RPBU. Het sluitpeil van de oosterscheldekering is 3.0 m+NAP
        # bij hogere waterstanden zal de kering dus (pogen) te sluiten. Nu is de kans op al dan niet sluiten
        # ook afhankelijk van deze waterstand. Hoe hoger de waterstand, hoe groter de kans dat het misgaat.
        # Daarom zit er in de database een tabel met optredende waterstanden bij Roompot Buiten. Wanneer
        # Deze waterstand < 3.0 is, zit de waarde -9.0 in de database, om aan te geven dat de kering open blijft.

        # Interpoleer sluitpeilen op het grid van de windsnelheid en zeewaterstand
        # Eerst over de windsnelheid
        intstr = InterpStruct(
            x=wind_speed.get_discretisation(), xp=self.closing_levels.u
        )
        levels_u = intstr.interp(
            fp=self.closing_levels.h_rpb,
            axis=self.closing_levels.input_variables.index("u"),
        )

        # Dan over de zeewaterstand
        intstr = InterpStruct(
            x=sea_level.get_discretisation(), xp=self.closing_levels.m
        )
        self.h_rpb = intstr.interp(
            fp=levels_u, axis=self.closing_levels.input_variables.index("m")
        )

        # Controleer op waarden van -9 en kleiner.
        self.not_closing = self.h_rpb <= -9.0

    def calculate_closing_probability(
        self, wind_direction: float, closing_situation: int
    ) -> np.ndarray:
        """
        Bereken de kans op het al dan niet falen van N sluizen
        in de Oosterscheldekering en het type sluiting (nood of strategie)
        De kansen zijn conditioneel op de zeewaterstand.
        """
        # Translate to wind direction id
        ir = self.wind_direction.get_discretisation().tolist().index(wind_direction)

        # Vertaal sluitsituatie (id) naar nood/strategie...
        sluittype = self.closing_situations[closing_situation][0]
        if sluittype not in ["Reguliere sluiting", "Noodsluiting"]:
            raise KeyError(sluittype)

        # ...en aantal schuiven
        nfaal = self.closing_situations[closing_situation][1]
        ik = self.k.tolist().index(nfaal)

        # Hernoem voor beter begrip
        kans_strategie_m = self.p_closing_k[:, 0]
        kans_handmatig_m = self.p_closing_k[:, 1]
        kans_noodsluit_m = self.p_closing_k[:, 2]
        # kans_geensluit_m = self.p_closing_k[:, 3]

        kansfalen_bemand = self.p_fail_k[:, 0]
        kansfalen_onbemand = self.p_fail_k[:, 1]

        # Kansen op noodsluiting, en N falende schuiven
        # Noodsluiting door mens (bemand) of systeem (onbemand)
        if sluittype == "Noodsluiting":
            kansfalensluiting = (
                kansfalen_bemand[ik] * kans_handmatig_m
                + kansfalen_onbemand[ik] * kans_noodsluit_m
            )

        elif sluittype == "Reguliere sluiting":
            # Strategiesluiting (bemand)
            kansfalensluiting = kansfalen_bemand[ik] * kans_strategie_m

            # En dan is er nog de kans dat de kering niet hoeft te sluiten (ongeacht bemand of onbemand)
            # Deze kans moet ingedeeld worden in een van de sluitscenario's. Neem hiervoor het sluitscenario
            # met zoveel mogelijk falende sluitingen, en de strategiesluiting
            # if nfaal == self.k.max():
            # kansfalensluiting[:] += kans_geensluit_m

        # Bereken de faalkansen aan de hand van de zeewaterstanden die voor de belastingcombinaties
        # optreden bij Roompot Buiten door te interpoleren op de zeewaterstanden
        keringsituatiekansen = Interpolate.inextrp1d(
            x=self.h_rpb[:, :, ir, :, :], xp=self.m_closing, fp=kansfalensluiting
        )

        # Als laatste is de kans op de keringsituatie nog afhankelijk van het al dan niet halen van het sluitpeil
        # Als het sluitpeil niet gehaald wordt, is de kans op de keringsituatie met sluiting 0, tenzij het om de
        # situatie zonder sluiting gaat
        idx = self.not_closing[:, :, ir, :, :]
        if (sluittype == "Reguliere sluiting") and (nfaal == self.k.max()):
            keringsituatiekansen[idx] = 1.0
        else:
            keringsituatiekansen[idx] = 0.0

        return keringsituatiekansen
