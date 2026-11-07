from .statistics_lower_river import StatisticsLowerRiver


class StatisticsHollandscheIJssel(StatisticsLowerRiver):
    """
    Statistics class for the Hollandsche IJssel
    The statistics of the Rhine-Meuse delta (sea level and wind coupled, discharge independent, Europoort barrier),
    with a weight for the Hollandsche IJssel barrier on top.

    The six closing situations of the database are the Europoort barrier (open, closed) times the Hollandsche
    IJssel barrier (open, closed without water burden, closed with water burden):

        closing situation   Europoort barrier   Hollandsche IJssel barrier
        1                   open                open
        2                   open                closed, without water burden
        3                   open                closed, with water burden
        4                   closed              open
        5                   closed              closed, without water burden
        6                   closed              closed, with water burden
    """

    def calculate_probability(self, wind_direction: float, closing_situation: int = 1, given: list = []):
        """
        Calculate the probability of occurence for the discretisation given the wind direction.

        The probability of a closing situation is the probability of the Europoort barrier state times a weight
        for the Hollandsche IJssel barrier. The weight is the same everywhere: the loading tables of the closed
        states already contain the closing, so the barrier is closed unless it fails. A closing is one with
        water burden with a given probability; the open state is the same for both closing types and is not
        split. The weights sum to 1, so the sum over the Hollandsche IJssel barrier states is the probability
        of the Rhine-Meuse delta.

        Parameters
        ----------
        wind_direction : float
            Wind direction
        closing_situation : int
            Closing situation, 1 to 6
        given : list
            Given stochasts
        """
        europoort_state, hollandsche_ijssel_state = divmod(int(closing_situation) - 1, 3)
        probability = super().calculate_probability(wind_direction, europoort_state + 1, given)
        if "k" in given:
            return probability

        alfa = self.settings.hollandsche_ijssel_barrier_alfa
        if hollandsche_ijssel_state == 0:
            return probability * alfa
        p_water_burden = self.settings.hollandsche_ijssel_water_burden_probability
        p_type = p_water_burden if hollandsche_ijssel_state == 2 else 1.0 - p_water_burden
        return probability * (1.0 - alfa) * p_type
