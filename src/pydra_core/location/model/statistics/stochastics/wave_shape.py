import numpy as np

from scipy.stats import norm

from ....settings.settings import Settings
from .....common.enum import WaveShapeType
from .....common.interpolate import Interpolate, InterpStruct
from .....common.probability import ProbabilityFunctions
from .....io.file_hydranl import FileHydraNL


class WaveShape:
    """
    Class to describe the wave form statistics.
    Used in describing wave forms in river discharge and lake level.
    """

    def __init__(self, settings: Settings, type: WaveShapeType):
        """
        Constructor class for the WaveShape statistics.

        Parameters
        ----------
        settings : Settings
            The Settings object
        """
        # Placeholder
        self.wave_shapes = None

        # Obtain the parameters
        self.base_duration = settings.base_duration
        self.pw = settings.waveshape_pw
        if type == WaveShapeType.DISCHARGE:
            self.top_duration = settings.top_duration_q
            self.lower_limit = settings.q_min
            self.upper_limit = settings.q_limit
            self.ifh = settings.ifh_q
            self.ifb = settings.ifb_q
        elif type == WaveShapeType.LAKE_LEVEL:
            self.top_duration = settings.top_duration_a
            self.lower_limit = settings.a_min
            self.upper_limit = settings.a_limit
            self.ifh = settings.ifh_a
            self.ifb = settings.ifb_a
        else:
            raise TypeError("[ERROR] Unknown WaveShape type.")

        # Time
        self.time = np.arange(
            0.0, settings.base_duration + 1e-6, settings.waveshape_time_step
        )
        self.ntime = len(self.time)
        if settings.base_duration != max(self.time):
            settings.base_duration = max(self.time)
            print(
                f"[NOTE] Base duration adjusted to blok wind ({settings.waveshape_time_step}h)."
            )

    def initialise_wave_shapes(self, peak: list, climate_change: float = 0.0):
        """
        Initialize the wave shapes for discharge or lake level.

        Parameters
        ----------
        peak : list
            A list with peak values or the discharge / lake level
        climate_change : float
            Increase of discharge or lake level to account for climate change (default = 0.0)
        """
        # Save
        self.peak = peak
        self.npeak = len(peak)

        # Determine shape of the wave
        if not self.pw:
            # Init wave shapes
            self.wave_shapes = self.__initialise_wave_shape(climate_change)
            self.wave_shapes = np.maximum(self.wave_shapes, self.lower_limit)

        # Read from a file
        else:
            raise NotImplementedError()

    def transition_wave(self, transition: float) -> None:
        """
        Transition wave form

        Parameters
        ----------
        transition : float
            Transition in hours
        """
        tijdenlang = np.zeros(2 * self.ntime - 1)
        golflang = np.zeros(2 * self.ntime - 1)

        if transition >= 0.0:
            # Als de verschuiving positief is in de tijd, dan wordt de tijdas
            # vooruit verschoven en eenzelfde periode wordt voor deze
            # verschoven tijdas geplaatst
            it = np.arange(self.ntime)
            tijdenlang[it + self.ntime - 1] = self.time[it] + transition

            it = np.arange(self.ntime - 1)
            tijdenlang[it] = tijdenlang[it + self.ntime - 1] - self.base_duration

        else:
            # Als de verschuiving negatief is in de tijd, dan wordt de tijdas
            # terug verschoven en eenzelfde periode wordt achter deze verschoven
            # tijdas geplaatst
            it = np.arange(self.ntime)
            tijdenlang[it] = self.time[it] + transition
            it = np.arange(1, self.ntime)
            tijdenlang[it + self.ntime - 1] = tijdenlang[it] + self.base_duration

        #  Bij de verschoven en uitgebreide tijdas worden de golfvormen twee keer gezet
        #  Door interpolatie worden op het reguliere rooster de uiteindelijke
        #  verschoven golfvormen verkregen
        intstr = InterpStruct(x=self.time, xp=tijdenlang)
        for ip in range(self.npeak):
            golflang[: self.ntime] = self.wave_shapes[:, ip]
            golflang[self.ntime :] = golflang[1 : self.ntime]
            self.wave_shapes[:, ip] = intstr.interp(fp=golflang, extrapolate=False)

    def get_wave_shapes(self) -> np.ndarray:
        """
        Return the wave shapes.

        Returns
        -------
        np.ndarray
            Array with size (t, peak)
        """
        return self.wave_shapes

    def exceedance_timestamps(self, levels: np.ndarray):
        """
        Determine the exceedance durations for several levels in the waveshapes,
        given the peak levels of the wave.

        Parameters
        ----------
        levels : np.ndarray
            Levels for which the exceedance duration is determined.

        Returns
        -------
        np.ndarray
            Exceedance duration [N peak x N levels]
        """
        # Check if the wave shapes are defined
        if self.wave_shapes is None:
            raise ValueError("[ERROR] Wave shaped are not defined.")

        # Init empty array
        ovduur = np.zeros((self.npeak, len(levels)))

        # Bepaal voor elk blokniveau de overschrijdingsduren
        for ip in range(self.npeak):
            # Get one wave shape
            wave_shape = self.wave_shapes[:, ip]

            # Initieer lege arrays
            t_boven = np.zeros_like(levels)
            t_onder = np.zeros_like(levels)

            # Bepaal waar de niveaus gelijk of hoger zijn aan de golfvorm, hiervoor zijn er geen overschrijdingen
            idx = levels > wave_shape.max()
            t_boven[idx] = -1.0
            t_onder[idx] = -1.0

            # Aangenomen wordt dat de golf een monotoon opgaande en neergaande flank heeft.
            # Onder deze aanname kan met interpolatie de kruisingen bepaald worden.
            diff = wave_shape[1:] - wave_shape[:-1]

            # Het verschil is de afstand van het volgende punt ten opzichte van het huidige punt
            opgaand = np.concatenate([[True], diff > 0])
            neergaand = np.concatenate([diff < 0, [True]])
            t_boven[~idx] = np.interp(
                x=levels[~idx], xp=wave_shape[opgaand], fp=self.time[opgaand]
            )
            t_onder[~idx] = np.interp(
                x=levels[~idx],
                xp=wave_shape[neergaand][::-1],
                fp=self.time[neergaand][::-1],
            )
            ovduur[ip, :] = t_onder - t_boven

        # Return
        return ovduur

    def instantaneous_exceedance_probability(self, exceedance_peaks, levels):
        """
        Calculate the instantaneous exceedance probability of the discharge/water level
        for each peak value of a discharge/water level wave.
        {
            Calculate the time when the discharge/water level is higher than a given level
                and determine the time when the discharge/water level is lower than the given level.
            Determine the duration of exceedance.
            Use the composite trapezoidal rule for calculating the instantaneous exceedance probability
                of the discharge/water level.
        }
        """
        # Calculate the vector with probability densities of the discharge/water level
        peak_probability = ProbabilityFunctions.probability_density(
            self.peak, exceedance_peaks
        ).probability
        instantaneous_prob = np.zeros_like(levels)
        exceedance_duration = self.exceedance_timestamps(levels)

        for ib in range(len(levels)):
            integral = (exceedance_duration[:, ib] * peak_probability).sum()

            # Calculate the instantaneous exceedance probability of the discharge/water level
            instantaneous_prob[ib] = integral / self.base_duration

        return instantaneous_prob

    def cum_norm_s_naar_y(self, y, mu, sigma):
        """
        Bepaal de cumulatieve verdelingsfunctie voor het meerpeil onder de transformatie van s naar y = K(s)
        {
            y= Parameterwaarde
            mu= Gemiddelde van de normale verdeling
            sigma= Standaarddeviatie van de normale verdeling
        }
        """
        if sigma <= 0:
            # Geef foutmelding
            raise ValueError("Sigma <= 0")

        # Integratiegrenzen en stapgrootte
        stap = 0.05
        x = np.arange(0, 20 + 0.1 * stap, stap)

        # Voor de parameterwaarde wordt de integraal over het rooster berekend. Let wel: delta = 0
        yn = (y[None, :] - x[:, None] - mu) / sigma

        dx = np.diff(np.r_[x[0], (x[1:] + x[:-1]) / 2.0, x[-1]])
        f_y_sigma = (np.exp(-x[:, None]) * norm().cdf(x=yn) * dx[:, None]).sum(0)

        return f_y_sigma

    def __initialise_wave_shape(self, climate_change: float) -> np.ndarray:
        """
        Create the wave shapes for discharge or lake level.

        Parameters
        ----------
        climate_change : float
            Increase of discharge or lake level to account for climate change (default = 0.0)

        Returns
        -------
        np.ndarray
            A 2D array with wave shapes (time, q_m_peak)
        """
        # Read the table with top durations (t_top) given q or m (q_m)
        _q_m, _t_top = FileHydraNL.read_file_2columns(self.top_duration)

        # Correct for climate change (increase discharge or lake level)
        if climate_change > 0.0:
            _q_m += climate_change

        # Check if the top duration is larger than the base duration
        if (_t_top > self.base_duration).any():
            print(
                f"[WARNING] Top duration from file '{self.top_duration}' is larger than base duration."
            )

        # Increase the arrays with 1 element to allow for extrapolation
        q_m = np.r_[_q_m, 2.0 * _q_m[-1] - _q_m[-2]]
        t_top = np.r_[_t_top, _t_top[-1]]

        # Calculate for each peak the top duration
        top_durations = Interpolate.inextrp1d(x=self.peak, xp=q_m, fp=t_top)

        # Create an empty array of size (t, peak) for the wave shapes
        wave_shapes = np.zeros((len(self.time), len(self.peak)))

        # Create a trapezoid wave shape
        for ip, top_duration in enumerate(top_durations):
            # If the discretisation step for wave shapes is equal to the base_duration
            if (
                self.peak[ip] < q_m[0]
                or (self.time[1] - self.time[0]) == self.base_duration
            ):
                # Vul de vector met tijdstippen
                ttabel = [0.0, self.base_duration]

                # Vul de vector met golfwaarden
                gtabel = [self.peak[ip], self.peak[ip]]

            # If the top duration is equal to 0.0
            elif top_duration == 0.0:
                # Als ingesnoerd wordt totaan het tijdstip waarop de golfvorm de maximale waarde heeft
                if self.ifb == 0.0:
                    #   Vul de vector met tijdstippen
                    ttabel = np.array(
                        [
                            0,
                            self.base_duration / 2.0 - 0.0001,
                            self.base_duration / 2.0,
                            self.base_duration / 2.0 + 0.0001,
                            self.base_duration,
                        ]
                    )

                    #   Vul de vector met golfwaarden
                    gtabel = np.array(
                        [
                            self.lower_limit,
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.peak[ip],
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.lower_limit,
                        ]
                    )

                # Als de golfvorm niet ingesnoerd wordt
                elif self.ifh == 1.0:
                    # Vul de vector met tijdstippen
                    ttabel = np.array(
                        [0.0, self.base_duration / 2.0, self.base_duration]
                    )

                    # Vul de vector met golfwaarden
                    gtabel = np.array(
                        [self.lower_limit, self.peak[ip], self.lower_limit]
                    )

                # Als er sprake is van een blokgolf
                elif self.ifb == (1.0 / (1.0 - self.ifh)):
                    # Vul de vector met tijdstippen
                    ttabel = np.array(
                        [0.0, self.base_duration / 2.0, self.base_duration]
                    )

                    # Vul de vector met golfwaarden
                    gtabel = np.array(
                        [
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.peak[ip],
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                        ]
                    )

                # Als de golfvorm wel ingesnoerd wordt
                else:
                    ttabel = np.zeros(5)
                    gtabel = np.zeros(5)

                    # Vul de vector met tijdstippen
                    ttabel[0] = 0.0
                    ttabel[2] = self.base_duration / 2.0
                    ttabel[4] = self.base_duration
                    ttabel[1] = ttabel[0] + self.ifh * (ttabel[2] - ttabel[0])
                    ttabel[1] = ttabel[2] - self.ifb * (ttabel[2] - ttabel[1])
                    ttabel[3] = ttabel[4] + self.ifh * (ttabel[2] - ttabel[4])
                    ttabel[3] = ttabel[2] + self.ifb * (ttabel[3] - ttabel[2])

                    # Vul de vector met golfwaarden
                    gtabel = np.array(
                        [
                            self.lower_limit,
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.peak[ip],
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.lower_limit,
                        ]
                    )

            # Als de invoergeg.topduur ongelijk aan nul is
            else:
                # Als de invoergeg.topduur gelijk is aan de invoergeg.basisduur
                if top_duration == self.base_duration:
                    # Vul de vector met tijdstippen
                    ttabel = np.array([0.0, self.base_duration])

                    # Vul de vector met golfwaarden
                    gtabel = np.array([self.peak[ip], self.peak[ip]])

                # Als ingesnoerd wordt totaan het tijdstip waarop de golfvorm de maximale waarde heeft
                elif self.ifb == 0.0:
                    # n = 6

                    # Vul de vector met tijdstippen
                    ttabel = np.array(
                        [
                            0.0,
                            self.base_duration / 2.0 - top_duration / 2.0 - 0.0001,
                            self.base_duration / 2.0 - top_duration / 2.0,
                            self.base_duration / 2.0 + top_duration / 2.0,
                            self.base_duration / 2.0 + top_duration / 2.0 + 0.0001,
                            self.base_duration,
                        ]
                    )

                    # Vul de vector met golfwaarden
                    gtabel = np.array(
                        [
                            self.lower_limit,
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.peak[ip],
                            self.peak[ip],
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.lower_limit,
                        ]
                    )

                # Als de golfvorm niet wordt ingesnoerd
                elif self.ifh == 1.0:
                    # Vul de vector met tijdstippen
                    ttabel = np.array(
                        [
                            0.0,
                            self.base_duration / 2.0 - top_duration / 2.0,
                            self.base_duration / 2.0 + top_duration / 2.0,
                            self.base_duration,
                        ]
                    )

                    # Vul de vector met golfwaarden
                    gtabel = np.array(
                        [
                            self.lower_limit,
                            self.peak[ip],
                            self.peak[ip],
                            self.lower_limit,
                        ]
                    )

                # Als er sprake is van een blokgolf
                elif self.ifb == (1.0 / (1.0 - self.ifh)):
                    # Vul de vector met tijdstippen
                    ttabel = np.array(
                        [
                            0.0,
                            self.base_duration / 2.0 - top_duration / 2.0,
                            self.base_duration / 2.0 + top_duration / 2.0,
                            self.base_duration,
                        ]
                    )

                    # Vul de vector met golfwaarden
                    gtabel = np.array(
                        [
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.peak[ip],
                            self.peak[ip],
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                        ]
                    )

                # Als de golfvorm wel ingesnoerd wordt
                else:
                    ttabel = np.zeros(6)
                    gtabel = np.zeros(6)

                    # Vul de vector met tijdstippen
                    ttabel[0] = 0.0
                    ttabel[2] = self.base_duration / 2.0 - top_duration / 2.0
                    ttabel[3] = self.base_duration / 2.0 + top_duration / 2.0
                    ttabel[5] = self.base_duration
                    ttabel[1] = ttabel[0] + self.ifh * (ttabel[2] - ttabel[0])
                    ttabel[1] = ttabel[2] - self.ifb * (ttabel[2] - ttabel[1])
                    ttabel[4] = ttabel[5] + self.ifh * (ttabel[3] - ttabel[5])
                    ttabel[4] = ttabel[3] + self.ifb * (ttabel[4] - ttabel[3])

                    # Vul de vector met golfwaarden
                    gtabel = np.array(
                        [
                            self.lower_limit,
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.peak[ip],
                            self.peak[ip],
                            self.lower_limit
                            + self.ifh * (self.peak[ip] - self.lower_limit),
                            self.lower_limit,
                        ]
                    )

            # Interpoleer de golfvorm naar de gewenste tijdas
            wave_shapes[:, ip] = np.interp(self.time, ttabel, gtabel)

        # Top de golfvormen af op het gewenste niveau
        wave_shapes = np.minimum(wave_shapes, self.upper_limit)

        # Return wave shapes
        return wave_shapes

    @staticmethod
    def bepaal_gezamenlijke_overschrijding(
        golfvormen_st1, niveaus_st1, golfvormen_st2, niveaus_st2
    ) -> np.ndarray:
        # Bepaal overschrijdingstijdstippen van beide stochasten in de golfvorm
        kruisng = np.zeros((len(niveaus_st1), len(niveaus_st2), 4))
        helling = np.zeros((len(niveaus_st1), len(niveaus_st2), 4), dtype=int)
        duren = np.zeros(
            (
                golfvormen_st1.npeak,
                golfvormen_st2.npeak,
                len(niveaus_st1),
                len(niveaus_st2),
            ),
            dtype=float,
        )

        # Bepaal eerst alle tijdstippen voor stochast 1
        kruisingen_st1 = np.zeros(
            (golfvormen_st1.npeak, len(niveaus_st1), 2), dtype=float
        )
        helling_st1 = np.zeros((golfvormen_st1.npeak, len(niveaus_st1), 2), dtype=int)
        for ip1 in range(golfvormen_st1.npeak):
            # Bepaal voor elk blokniveau de overschrijdingsduren
            kruisingen_st1[ip1, :, :], helling_st1[ip1, :, :] = (
                golfvormen_st1._overschrijdingstijdstip_op_af_v2(
                    golfvormen_st1.wave_shapes[:, ip1], golfvormen_st1.time, niveaus_st1
                )
            )

        arange = np.arange(max(len(niveaus_st1), len(niveaus_st2)))

        # loop vervolgens over stochast 2
        kruisingen_st2 = np.zeros((len(niveaus_st2), 2))
        helling_st2 = np.zeros((len(niveaus_st2), 2))
        for ip2 in range(golfvormen_st2.npeak):
            # Bepaal voor elk blokniveau de overschrijdingsduren
            kruisingen_st2[:, :], helling_st2[:, :] = (
                golfvormen_st2._overschrijdingstijdstip_op_af_v2(
                    golfvormen_st2.wave_shapes[:, ip2], golfvormen_st2.time, niveaus_st2
                )
            )

            # Bepaal voor elk piekniveau van stochast 2 de gezamenlijke overschrijdingen
            # Doe dit door de tijdstippen voor elke combinatie samen te voegen, te sorteren
            # op absolute waarde, en de overgang van laatste opgaand naar eerste neergaand
            # te bepalen
            for ip1 in range(golfvormen_st1.npeak):
                idx1 = kruisingen_st1[ip1, :, 0] != -1.0
                idx2 = kruisingen_st2[:, 0] != -1.0

                len1 = idx1.sum()
                len2 = idx2.sum()
                kruisng = np.zeros((len1, len2, 4))
                helling = np.zeros((len1, len2, 4), dtype=np.int8)

                # Vul arrays
                kruisng[:, :, :2] = kruisingen_st1[ip1, idx1, None, :]
                kruisng[:, :, 2:] = kruisingen_st2[None, idx2, :]

                helling[:, :, :2] = helling_st1[ip1, idx1, None, :]
                helling[:, :, 2:] = helling_st2[None, idx2, :]

                # Sorteer
                order = np.argsort(np.absolute(kruisng), axis=2)
                i1, i2 = arange[:len1, None, None], arange[None, :len2, None]
                kruisng[...] = kruisng[i1, i2, order]
                helling[...] = helling[i1, i2, order]

                # Bepaal alle plekken waar een tekenwisseling (kruising tussen golf en niveau) plaatsvindt
                wh = np.where((helling[:, :, 1:] ^ helling[:, :, :-1]))

                # Bereken het tijdsverschil bij tekenwisseling
                diff = kruisng[wh[0], wh[1], wh[2] + 1] - kruisng[wh[0], wh[1], wh[2]]
                # Zet tijdverschil bij neergaand naar opgaand op 0
                diff[
                    (helling[wh[0], wh[1], wh[2] + 1] < helling[wh[0], wh[1], wh[2]])
                ] = 0.0
                # print(diff.max(), diff.min())
                duren[ip1, ip2, wh[0], wh[1]] += diff

        return duren

    @staticmethod
    def _overschrijdingstijdstip_op_af_v2(
        vormgolf: np.ndarray, tijden: np.ndarray, niveaus: np.ndarray
    ):
        # Bepaal waar de golfvorm boven het te testen niveau zit
        diff = (vormgolf[None, :] - niveaus[:, None]) >= 0.0
        # Bepaal alle plekken waar een tekenwisseling (kruising tussen golf en niveau) plaatsvindt
        locn, loct = np.where((diff[:, 1:] ^ diff[:, :-1]))

        # Bepaal het snijpunt bij de kruising
        dtdq = (tijden[loct + 1] - tijden[loct]) / (vormgolf[loct + 1] - vormgolf[loct])
        # Voeg dit punt toe aan de tijd
        duur = tijden[loct] + dtdq * (niveaus[locn] - vormgolf[loct])

        # Maak lege array aan
        t1t2 = np.full((len(niveaus), 2), np.nan, dtype=float)
        helling = np.full((len(niveaus), 2), 0, dtype=np.int8)
        # Waar het niveau hoger ligt dan de piekwaarde, -1 in
        t1t2[niveaus > vormgolf.max(), :] = -1
        # Controleer of alle niveaus of 0 of 2 keer voorkomen
        r = np.bincount(locn)
        assert ((r == 2) | (r == 0)).all()

        # Vul in in array met tijden. Vermenigvuldig met het teken van de helling *-1
        t1t2[locn, np.arange(len(locn)) % 2] = duur
        # Bepaal helling. Opgaand heeft dus -, neergaand +. De som geeft de totale overschrijdingsduur
        helling[locn, np.arange(len(locn)) % 2] = -np.sign(dtdq)

        # Als er NaN's in de array zitten, is dit op plekken waar het niveau in zijn geheel onder
        # of op het laagste punt van de golf ligt. Vul hier de min en max van de tijden in
        idx = np.isnan(t1t2)
        t1t2[idx[:, 0], 0] = tijden.min()
        t1t2[idx[:, 1], 1] = tijden.max()
        helling[idx[:, 0], 0] = -1
        helling[idx[:, 1], 1] = 1

        # Vermenigvuldig de even elementen met -1
        # idx = duur < 0
        # duur[::2] *= -1
        return t1t2, helling
