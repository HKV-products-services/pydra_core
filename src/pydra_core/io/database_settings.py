import pandas as pd
import sqlite3

from pathlib import Path

from ..common.enum import WaterSystem


class DatabaseSettings:
    """
    Settings database
    """

    def __init__(self) -> None:
        self.con = None

    def __enter__(self) -> "DatabaseSettings":
        # Init the connection
        self.con = sqlite3.connect(
            Path(__file__).resolve().parent.parent
            / "data"
            / "settings"
            / "calculation_settings.sqlite"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Close the connection
        self.con.close()

    def get_settings(self, watersystem: WaterSystem) -> pd.DataFrame:
        """
        Return all settings from the database

        Parameters
        ----------
        watersystem : WaterSystem
            WaterSystem to get the settings for

        Returns
        -------
        pd.DataFrame
            A DataFrame with settings
        """
        # Query
        sql = f"SELECT SettingName, SettingValue FROM CalculationSettings WHERE WaterSystem = {watersystem.value}"
        settings = pd.read_sql(sql, self.con)

        # Parse to int and float if possible
        for n, row in settings.iterrows():
            settings.loc[n, "SettingValue"] = self.__parse_value(row["SettingValue"])

        # Return the settings
        return settings

    def get_sea_level_statistic_points(self, watersystem: WaterSystem) -> pd.DataFrame:
        """
        Return the reference points for sea level statistics

        Parameters
        ----------
        watersystem : WaterSystem
            Watersystem for which to return the reference points

        Returns
        -------
        pd.DataFrame
            All reference points
        """
        # Obtain the right table name
        if watersystem in [
            WaterSystem.COAST_SOUTH,
            WaterSystem.COAST_CENTRAL,
            WaterSystem.COAST_NORTH,
        ]:
            tablename = "CoastReferencePoints"
        elif watersystem in [WaterSystem.WADDEN_SEA_WEST, WaterSystem.WADDEN_SEA_EAST]:
            tablename = "WaddenSeaReferencePoints"
        else:
            raise NotImplementedError(
                f"[ERROR] No sea level reference points implemented for {watersystem}."
            )

        # SQL
        data = pd.read_sql(
            f"SELECT Name, X, Y FROM {tablename}", self.con, index_col="Name"
        )

        # Return data
        return data

    def get_sea_level_sub_systems(self, watersystem: WaterSystem) -> pd.DataFrame:
        """
        Return all subsystems of the coast or Wadden sea.

        Parameters
        ----------
        watersystem : WaterSystem
            Watersystem for which to return the reference points

        Returns
        -------
        pd.DataFrame
            All subsystems
        """
        # Obtain the right table name
        if watersystem in [
            WaterSystem.COAST_SOUTH,
            WaterSystem.COAST_CENTRAL,
            WaterSystem.COAST_NORTH,
        ]:
            tablename = "CoastSubSystems"
        elif watersystem in [WaterSystem.WADDEN_SEA_WEST, WaterSystem.WADDEN_SEA_EAST]:
            tablename = "WaddenSeaSubSystems"
        else:
            raise NotImplementedError(
                f"[ERROR] No sea level reference points implemented for {watersystem}."
            )

        # SQL
        data = pd.read_sql(
            f"SELECT SubSystemId, Point1, Point2, Point3 FROM {tablename}",
            self.con,
            index_col="SubSystemId",
        )

        # Return data
        return data

    def get_input_variable_ids(self) -> dict:
        """
        Return a dictionary with the input variable ids (ivid) and symbols (isymbol)

        Returns
        -------
        dict
            A dictionary with {ivid : isymbol}
        """
        # Query
        sql = "SELECT InputVariableId, InputVariableSymbol FROM InputVariables"
        data = self.con.execute(sql).fetchall()

        # Process data
        results = {data[i][0]: data[i][1] for i in range(len(data))}

        # Return the dictionary
        return results

    def get_result_variable_ids(self) -> dict:
        """
        Return a dictionary with the result variable ids (rvid) and symbols (rsymbol)

        Returns
        -------
        dict
            A dictionary with {rvid : rsymbol}
        """
        # Query
        sql = "SELECT ResultVariableId, ResultVariableSymbol FROM ResultVariables"
        data = self.con.execute(sql).fetchall()

        # Process data
        results = {data[i][0]: data[i][1] for i in range(len(data))}

        # Return the dictionary
        return results

    def __parse_value(self, value):
        # Parse value from string
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return int(value)
        except ValueError:
            pass
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        return value
