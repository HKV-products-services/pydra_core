import pandas as pd

from typing import Union

from ..calculation_settings_reader import CalculationSettingsReader
from ..hrdatabase_reader import HRDReader
from ...common.enum import WaterSystem
from ...location.settings.settings import Settings


class HRDReaderWBI2017(HRDReader):
    """
    HR database reader for WBI2017 databases.
    """

    def get_input_variables(self) -> list:
        """
        Return the input variables

        Returns
        -------
        list
            List with input variables
        """
        # Query
        sql = "SELECT InputVariableId FROM HRDInputVariables"
        data = self.con.execute(sql).fetchall()

        # Settings database
        with CalculationSettingsReader() as database:
            ivids = database.get_input_variable_ids()
        data = [ivids[i[0]] for i in data]

        # Shift wind speed in front
        if "u" in data:
            data.pop(data.index("u"))
            data.insert(0, "u")

        # Return results
        return data

    def get_result_variables(self) -> list:
        """
        Return the result variables

        Returns
        -------
        list
            List with result variables
        """
        # Query
        sql = "SELECT ResultVariableId FROM HRDResultVariables"
        data = self.con.execute(sql).fetchall()

        # Settings database
        with CalculationSettingsReader() as database:
            rvids = database.get_result_variable_ids()
        data = [rvids[i[0]] for i in data]

        # For the coast, if not defined, the local water level is equal to the sea level
        if self.get_water_system() in [
            WaterSystem.WADDEN_SEA_EAST,
            WaterSystem.WADDEN_SEA_WEST,
            WaterSystem.COAST_NORTH,
            WaterSystem.COAST_CENTRAL,
            WaterSystem.COAST_SOUTH,
            WaterSystem.WESTERN_SCHELDT,
        ]:
            if "h" not in rvids:
                data.insert(0, "h")

        # Return results
        return data

    def get_model_uncertainties(self, hrdlocation: Union[int, str, Settings]) -> pd.DataFrame:
        """
        Return the model uncertainties

        Parameters
        ----------
        hrdlocation : Union[int, str, Settings]
            HRDLocation in form of HRDLocationId, HRDLocationName or Settings object

        Returns
        -------
        pd.DataFrame
            DataFrame with the distribution per closing situation
        """
        # Obtain the hrdlocationid
        if isinstance(hrdlocation, (str, Settings)):
            hrdlocation = self.get_hrdlocation_id(hrdlocation)

        # Obtain all model uncertainties from the database for the hrdlocation
        sql = f"""
                SELECT umf.HRDLocationId, umf.ClosingSituationId, hrv.ResultVariableId, umf.Mean, umf.Standarddeviation
                FROM UncertaintyModelFactor umf
                INNER JOIN HRDResultVariables hrv
                ON umf.HRDResultColumnId = hrv.HRDResultColumnId
                WHERE umf.HRDLocationId = {hrdlocation}
                """
        data = pd.read_sql(sql, self.con, index_col="HRDLocationId")

        # Adjust dataframe
        with CalculationSettingsReader() as database:
            rvids = database.get_result_variable_ids()
        data.rename(
            columns={
                "ClosingSituationId": "k",
                "ResultVariableId": "rvid",
                "Mean": "mean",
                "Standarddeviation": "stdev",
            },
            inplace=True,
        )
        data["rvid"] = data["rvid"].replace(rvids)

        # Return the model uncertainties
        return data

    def get_correlation_uncertainties(self, hrdlocation: Union[int, str, Settings]) -> pd.DataFrame:
        """
        Return the correlation between model uncertainties

        Parameters
        ----------
        hrdlocation : Union[int, str, Settings]
            HRDLocation in form of HRDLocationId, HRDLocationName or Settings object
        """
        # Check if correlations are present
        table_check_query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='UncertaintyCorrelationFactor';
        """
        table_exists = pd.read_sql(table_check_query, self.con)
        if table_exists.empty:
            return None

        # Obtain the hrdlocationid
        if isinstance(hrdlocation, (str, Settings)):
            hrdlocation = self.get_hrdlocation_id(hrdlocation)

        # ResultVariableIds
        with CalculationSettingsReader() as database:
            rvids = database.get_result_variable_ids()

        # Data uit correlatie tabel
        sql = f"""
                SELECT ucf.HRDLocationId, ucf.ClosingSituationId, hrv.ResultVariableId, ucf.HRDResultColumnId2, ucf.Correlation
                FROM UncertaintyCorrelationFactor ucf
                INNER JOIN HRDResultVariables hrv
                ON ucf.HRDResultColumnId = hrv.HRDResultColumnId
                WHERE ucf.HRDLocationId = {hrdlocation}
                """
        data = pd.read_sql(sql, self.con, index_col="HRDLocationId")

        # Vertaal tabel naar HRDResultColumnId2
        # Zo niet, negeer en ga verder, neem aan dat de HRDResultColumnId2 heeft dezelfde Ids als HRDResultColumnId
        try:
            sql = """
                    SELECT HRDResultColumnId2, ResultVariableId
                    FROM HRDResultVariables2 hrv2
                    INNER JOIN HRDResultVariables hrv ON hrv.HRDResultColumnId = hrv2.HRDResultColumnId
                    """
            data_hrdid2 = self.con.execute(sql).fetchall()
            hrdid2_to_rvid = {_hrdid: _hrdid2 for _hrdid, _hrdid2 in data_hrdid2}
            data = data.replace({"HRDResultColumnId2": hrdid2_to_rvid})
        except Exception as e:
            print(f"ERROR: {e}, continuing without")
            pass

        # Replace column names
        data.rename(
            columns={
                "ClosingSituationId": "k",
                "ResultVariableId": "rvid",
                "HRDResultColumnId2": "rvid2",
                "Correlation": "rho",
            },
            inplace=True,
        )

        # Check of alle ResultVariableId(2) rvids zijn
        if not set(data["rvid"]).issubset(set(rvids)) or not set(data["rvid2"]).issubset(set(rvids)):
            raise ValueError("ERROR")

        # Change ResultVariableId(2) to rvids
        data["rvid"] = data["rvid"].replace(rvids)
        data["rvid2"] = data["rvid2"].replace(rvids)

        # Return the model uncertainties
        return data

    def get_result_table(self, hrdlocation: Union[str, Settings]) -> pd.DataFrame:
        """
        Function to read the load combinations of a location to a pandas DataFrame

        Parameters
        ----------
        hrdlocation : Union[str, Settings]
            HRDLocation

        Returns
        -------
        pd.DataFrame
            A DataFrame with load combinations
        """
        # Obtain HRDLocationId
        hrdlocationid = self.get_hrdlocation_id(hrdlocation)
        with CalculationSettingsReader() as database:
            ivids = database.get_input_variable_ids()
            rvids = database.get_result_variable_ids()

        # Obtain all data from the HydroDynamicData table
        # (HydroDynamicDataId, HRDLocationId, ClosingSituationId, HRDWindDirectionID)
        query = f"SELECT * FROM HydroDynamicData WHERE HRDLocationId = {hrdlocationid}"
        hydrodynamicdata = pd.read_sql(query, self.con, index_col="HydroDynamicDataId")

        # Obtain all data from the HydroDynamicInputData table
        hydrodynamicdataids = ",".join(hydrodynamicdata.index.values.astype(str).tolist())
        query = """
        SELECT ID.HydroDynamicDataId, IV.InputVariableId, ID.Value
        FROM HydroDynamicInputData ID
        INNER JOIN HRDInputVariables IV ON ID.HRDInputColumnId = IV.HRDInputColumnId
        WHERE HydroDynamicDataId IN ({});
        """.format(hydrodynamicdataids)
        hydrodynamicinputdata = pd.read_sql(query, self.con, index_col=["HydroDynamicDataId", "InputVariableId"]).unstack()
        ivcols = [ivids[i] for i in hydrodynamicinputdata.columns.get_level_values(1)]
        hydrodynamicinputdata.columns = ivcols

        # Obtain all data from the HydroDynamicResultData table
        query = """
        SELECT RD.HydroDynamicDataId, RV.ResultVariableId, RD.Value
        FROM HydroDynamicResultData RD
        INNER JOIN HRDResultVariables RV ON RD.HRDResultColumnId = RV.HRDResultColumnId
        WHERE HydroDynamicDataId IN ({});
        """.format(hydrodynamicdataids)
        hydrodynamicresultdata = pd.read_sql(query, self.con, index_col=["HydroDynamicDataId", "ResultVariableId"]).unstack()
        rvcols = [rvids[i] for i in hydrodynamicresultdata.columns.get_level_values(1)]
        hydrodynamicresultdata.columns = rvcols

        # Merge the three tables
        results = hydrodynamicdata.join(hydrodynamicinputdata).join(hydrodynamicresultdata)

        # Vervang windrichting
        windrdict = self.get_wind_directions()
        for wid, r in windrdict.items():
            if r == 0.0:
                windrdict[wid] = 360.0
        results["Wind direction"] = [windrdict[i] for i in results["HRDWindDirectionId"].array]
        results.drop(["HRDLocationId", "HRDWindDirectionId"], axis=1, inplace=True)

        # Replace discrete stochasts
        results.rename(columns={"Wind direction": "r", "ClosingSituationId": "k"}, inplace=True)

        # Move r to the front
        results.insert(0, "r", results.pop("r"))
        results.sort_values(by=["k", "r"] + ivcols, inplace=True)

        # Return results
        return results
