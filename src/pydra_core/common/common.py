from .enum import WaterSystem


class CommonFunctions:
    """
    A class with common functions.
    """

    @staticmethod
    def is_coast(watersystem: WaterSystem) -> bool:
        """
        Returns True if a watersystem belongs to the coast loading model.

        Parameters
        ----------
        watersystem : WaterSystem
            The watersystem

        Returns
        -------
        bool
            True if the watersystem is Coast
        """
        if watersystem in [
            WaterSystem.COAST_SOUTH,
            WaterSystem.COAST_CENTRAL,
            WaterSystem.COAST_NORTH,
            WaterSystem.COAST_DUNES,
            WaterSystem.WADDEN_SEA_WEST,
            WaterSystem.WADDEN_SEA_EAST,
            WaterSystem.WESTERN_SCHELDT,
        ]:
            return True
        return False

    @staticmethod
    def is_lower_rivier(watersystem: WaterSystem) -> bool:
        """
        Returns True if a watersystem belongs to the lower river loading model.

        Parameters
        ----------
        watersystem : WaterSystem
            The watersystem

        Returns
        -------
        bool
            True if the watersystem is a lower river
        """
        if watersystem in [
            WaterSystem.MEUSE_TIDAL,
            WaterSystem.RHINE_TIDAL,
            WaterSystem.EUROPOORT,
        ]:
            return True
        return False

    @staticmethod
    def position_from_line(where, pt, refpts):
        """
        Check where a point is located relative to a line
        """

        def left(pt, refpts):
            A, B = refpts
            # Determine if the point is located left or right of the line
            sign = ((B[0] - A[0]) * (pt[1] - A[1])) - ((B[1] - A[1]) * (pt[0] - A[0]))
            if sign > 0:
                left = True
            else:
                left = False
            return left

        if where == "north" or where == "south":
            # Sorteer punten links naar recht, en kijk het links ligt
            if refpts[0][0] > refpts[1][0]:
                leftpoint, rightpoint = refpts[1], refpts[0]
            else:
                leftpoint, rightpoint = refpts[0], refpts[1]

            # Bekijk of het punt links of recht van de lijn links naar rechts ligt
            if where == "north":
                return left(pt, [leftpoint, rightpoint])
            else:
                return left(pt, [rightpoint, leftpoint])

        if where == "west" or where == "east":
            # Sorteer punten onder naar boven, en kijk het links ligt
            if refpts[0][1] > refpts[1][1]:
                downpoint, toppoint = refpts[1], refpts[0]
            else:
                downpoint, toppoint = refpts[0], refpts[1]

            if where == "west":
                return left(pt, [downpoint, toppoint])
            else:
                return left(pt, [toppoint, downpoint])
