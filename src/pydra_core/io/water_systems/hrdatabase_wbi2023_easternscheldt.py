import pandas as pd

from typing import Union

from ..calculation_settings_reader import CalculationSettingsReader
from .hrdatabase_wbi2017_generic import HRDReaderWBI2017
from ...common.enum import WaterSystem
from ...location.settings.settings import Settings


class HRDReaderWBI2023EasternScheldt(HRDReaderWBI2017):
    """
    HR database reader for WBI2023 Eastern Scheldt databases.
    Extends HRDReaderWBI2017 with Eastern Scheldt barrier closing levels and a
    different result table layout.
    """

    def get_closing_levels(self) -> pd.DataFrame:
        """
        Read the closing levels for the Eastern Scheldt barrier.

        Returns
        -------
        pd.DataFrame
            A Dataframe with the closing level (h_rpb), given r, u, m, d, p
        """
        # Read the table from the ClosingCriterionsOSK
        table = pd.read_sql("SELECT * FROM ClosingCriterionsOSK", con=self.con)

        # Rename the entries
        table.rename(
            columns={
                "WindDirection": "r",
                "WindSpeed": "u",
                "WaterLevel": "m",
                "StormDuration": "d",
                "PhaseDifference": "p",
                "WaterLevelRPB": "h_rpb",
            },
            inplace=True,
        )

        # Return table
        return table

    def get_closing_situations(self) -> dict:
        """
        Read the closing situations from the database (ClosingSituationId : (Description : FailingLocks)).
        e.g. 1 : ("Reguliere sluiting", 0)

        Only works for the Eastern Scheldt.
        """
        # Check watersystem
        if self.get_water_system() != WaterSystem.EASTERN_SCHELDT:
            raise ValueError("[ERROR] Function can only be called for the Eastern Scheldt")

        # Read table
        sql = """
                SELECT C.ClosingSituationId, T.Description, C.FailingLocks
                FROM ClosingSituations C
                INNER JOIN ClosingSituationTypes T ON C.ClosingSituationTypeId = T.ClosingSituationTypeId
                """
        results = self.con.execute(sql).fetchall()

        # Post processing into an dictionary
        results = {i[0]: (i[1], i[2]) for i in results}

        # Return
        return results

    def get_result_table(self, hrdlocation: Union[str, Settings]) -> pd.DataFrame:
        """
        Function to export the loadcombinations of a location to a pandas DataFrame

        Parameters
        ----------
        hrdlocation : Union[str, Settings]
            HRDLocation
        """
        # Obtain HRDLocationId
        hrdlocationid = self.get_hrdlocation_id(hrdlocation)
        with CalculationSettingsReader() as database:
            ivids = database.get_input_variable_ids()
            rvids = database.get_result_variable_ids()

        # First collect the dataids. Also replace wind direction ids with real ids
        SQL = """
        SELECT D.HydraulicLoadId, D.ClosingSituationId, W.Direction AS "Wind direction"
        FROM HydroDynamicData D
        INNER JOIN HRDWindDirections W ON D.HRDWindDirectionId=W.HRDWindDirectionId;"""
        dataids = pd.read_sql(SQL, self.con, index_col="HydraulicLoadId")
        dataids.rename(columns={"Wind direction": "r", "ClosingSituationId": "k"}, inplace=True)

        # Collect the result data. Replace HRDResultColumnId with variable id's
        SQL = """
        SELECT RD.HydraulicLoadId, RV.ResultVariableId, RD.Value
        FROM HydroDynamicResultData RD
        INNER JOIN HRDResultVariables RV ON RD.HRDResultColumnId = RV.HRDResultColumnId
        WHERE HRDLocationId = {};""".format(hrdlocationid)
        resultdata = pd.read_sql(SQL, self.con, index_col=["HydraulicLoadId", "ResultVariableId"]).unstack()

        # Reduce columnindex to single level index (without 'Value')
        resultdata.columns = [rvids[rid] for rid in resultdata.columns.get_level_values(1)]

        # Create dictionary for mapping HRDInputColumnId to InputVariableId
        SQL = """
        SELECT ID.HydraulicLoadId, IV.InputVariableId, ID.Value
        FROM HydroDynamicInputData ID
        INNER JOIN HRDInputVariables IV ON ID.HRDInputColumnId = IV.HRDInputColumnId"""
        inputdata = pd.read_sql(SQL, self.con, index_col=["HydraulicLoadId", "InputVariableId"]).unstack()

        # Reduce columnindex to single level index (without 'Value')
        inputdata.columns = [ivids[ivid] for ivid in inputdata.columns.get_level_values(1)]

        # Join data and sort values
        resultaat = dataids.join(inputdata).join(resultdata).sort_values(by=["r", "u", "m"])

        # In the WBI2023 the water levels and waves are in the same table, but have different input variables
        # Split the water levels and wave results
        idx = pd.isnull(resultaat["hs"])

        waterlevels = resultaat.loc[idx].dropna(how="all", axis=1)
        waveconditions = resultaat.loc[~idx].dropna(how="all", axis=1)

        # Replace m_os by m for the water level and m for h for the wave conditions
        waterlevels.rename(columns={"m_os": "m"}, inplace=True)
        waveconditions.rename(columns={"m": "h"}, inplace=True)

        return waterlevels, waveconditions
