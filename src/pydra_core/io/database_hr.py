import pandas as pd
import sqlite3

from pathlib import Path
from typing import List, Union

from .database_settings import DatabaseSettings
from ..common.enum import WaterSystem
from ..location.settings.settings import Settings


class DatabaseHR:
    """
    HR database sqlite
    """

    def __init__(self, path_to_database: str) -> None:
        # Check if the path is valid
        if not Path(path_to_database).exists():
            raise OSError(path_to_database)

        # Save the path
        self.path_to_database = path_to_database
        self.con = None

    def __enter__(self) -> "DatabaseHR":
        # Init the connection
        self.con = sqlite3.connect(self.path_to_database)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Close the connection
        self.con.close()

    def get_water_system(self) -> WaterSystem:
        """
        Obtain the water system from the .sqlite database

        Returns
        -------
        WaterSystem
            Corresponding water system
        """
        # Obtain the water system ID from the sqlite
        wsid = self.con.execute("SELECT GeneralId FROM General").fetchone()[0]

        # Return the WaterSystem
        return WaterSystem(wsid)

    def get_hrdlocations_names(self) -> List[str]:
        """
        Obtain a list with all names of the hrdlocations

        Returns
        -------
        list[str]
            A list with all names of hrdlocations
        """
        # Obtain the water system ID from the sqlite
        hrdlocations = self.con.execute("SELECT Name FROM HRDLocations").fetchall()

        # Convert the result to a list of strings
        names = [row[0] for row in hrdlocations]

        # Return names
        return names

    def get_hrdlocation_id(self, hrdlocation: Union[str, Settings]) -> int:
        """
        Returns the HRDLocationID

        Parameters
        ----------
        hrdlocation : Union[str, Settings]
            HRDLocation

        Returns
        -------
        int
            HRDLocationId
        """
        # Obtain the HRDLocationName from a Settings object
        if isinstance(hrdlocation, Settings):
            hrdlocation = hrdlocation.location

        # Obtain the water system ID from the sqlite
        hrdlocationid = self.con.execute(
            f"SELECT HRDLocationId FROM HRDLocations WHERE Name = '{hrdlocation}'"
        ).fetchone()[0]

        # Return HRDLocationId
        return hrdlocationid

    def get_hrdlocation_xy(
        self, hrdlocation: Union[str, Settings]
    ) -> Union[float, float]:
        """
        Returns the X and Y coordinate of the HRDLocation

        Parameters
        ----------
        hrdlocation : Union[str, Settings]
            HRDLocation

        Returns
        -------
        Union[float, float]
            X and Y coordinate
        """
        # Obtain the HRDLocationName from a Settings object
        if isinstance(hrdlocation, Settings):
            hrdlocation = hrdlocation.location

        # Obtain the water system ID from the sqlite
        hrdlocationid = self.con.execute(
            f"SELECT XCoordinate, YCoordinate FROM HRDLocations WHERE Name = '{hrdlocation}'"
        ).fetchone()

        # Return HRDLocationId
        return hrdlocationid

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
        with DatabaseSettings() as database:
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
        with DatabaseSettings() as database:
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

    def get_model_uncertainties(
        self, hrdlocation: Union[int, str, Settings]
    ) -> pd.DataFrame:
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
        with DatabaseSettings() as database:
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
        data["rvid"].replace(rvids, inplace=True)

        # Return the model uncertainties
        return data

    def get_correlation_uncertainties(
        self, hrdlocation: Union[int, str, Settings]
    ) -> pd.DataFrame:
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
        with DatabaseSettings() as database:
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
        if not set(data["rvid"]).issubset(set(rvids)) or not set(
            data["rvid2"]
        ).issubset(set(rvids)):
            raise ValueError("ERROR")

        # Change ResultVariableId(2) to rvids
        data["rvid"].replace(rvids, inplace=True)
        data["rvid2"].replace(rvids, inplace=True)

        # Return the model uncertainties
        return data

    def get_wind_directions(self) -> dict:
        """
        Obtain a dictionary with HRDWindDirectionIds and Directions

        Returns
        -------
        dict
            A dictionary {HRDWindDirectionId : Direction}
        """
        # Wind directions
        results = self.con.execute("SELECT * FROM HRDWindDirections").fetchall()

        # Process wind directions such that {wind_id : wind_direction}
        wind_direction = {wid: wr for wid, wr in results}

        # Return wind direction dictionary
        return wind_direction

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
        with DatabaseSettings() as database:
            ivids = database.get_input_variable_ids()
            rvids = database.get_result_variable_ids()

        # Obtain all data from the HydroDynamicData table
        # (HydroDynamicDataId, HRDLocationId, ClosingSituationId, HRDWindDirectionID)
        query = f"SELECT * FROM HydroDynamicData WHERE HRDLocationId = {hrdlocationid}"
        hydrodynamicdata = pd.read_sql(query, self.con, index_col="HydroDynamicDataId")

        # Obtain all data from the HydroDynamicInputData table
        hydrodynamicdataids = ",".join(
            hydrodynamicdata.index.values.astype(str).tolist()
        )
        query = """
        SELECT ID.HydroDynamicDataId, IV.InputVariableId, ID.Value
        FROM HydroDynamicInputData ID
        INNER JOIN HRDInputVariables IV ON ID.HRDInputColumnId = IV.HRDInputColumnId
        WHERE HydroDynamicDataId IN ({});
        """.format(hydrodynamicdataids)
        hydrodynamicinputdata = pd.read_sql(
            query, self.con, index_col=["HydroDynamicDataId", "InputVariableId"]
        ).unstack()
        ivcols = [ivids[i] for i in hydrodynamicinputdata.columns.get_level_values(1)]
        hydrodynamicinputdata.columns = ivcols

        # Obtain all data from the HydroDynamicResultData table
        query = """
        SELECT RD.HydroDynamicDataId, RV.ResultVariableId, RD.Value
        FROM HydroDynamicResultData RD
        INNER JOIN HRDResultVariables RV ON RD.HRDResultColumnId = RV.HRDResultColumnId
        WHERE HydroDynamicDataId IN ({});
        """.format(hydrodynamicdataids)
        hydrodynamicresultdata = pd.read_sql(
            query, self.con, index_col=["HydroDynamicDataId", "ResultVariableId"]
        ).unstack()
        rvcols = [rvids[i] for i in hydrodynamicresultdata.columns.get_level_values(1)]
        hydrodynamicresultdata.columns = rvcols

        # Merge the three tables
        results = hydrodynamicdata.join(hydrodynamicinputdata).join(
            hydrodynamicresultdata
        )

        # Vervang windrichting
        windrdict = self.get_wind_directions()
        for wid, r in windrdict.items():
            if r == 0.0:
                windrdict[wid] = 360.0
        results["Wind direction"] = [
            windrdict[i] for i in results["HRDWindDirectionId"].array
        ]
        results.drop(["HRDLocationId", "HRDWindDirectionId"], axis=1, inplace=True)

        # Replace discrete stochasts
        results.rename(
            columns={"Wind direction": "r", "ClosingSituationId": "k"}, inplace=True
        )

        # Move r to the front
        results.insert(0, "r", results.pop("r"))
        results.sort_values(by=["k", "r"] + ivcols, inplace=True)

        # Return results
        return results

    def get_closing_levels_table_europoort(self) -> dict:
        """
        Read the closing levels for the Europoort barrier.

        Returns
        -------
        pd.DataFrame
            A Dataframe with the closing level at sea (m) given r, u, q
        """

        # If there is a table called 'Sluitfunctie Europoortkering', use it
        try:
            # Read the table
            table = pd.read_sql(
                "SELECT * FROM [Sluitfunctie Europoortkering]", con=self.con
            )

        # Otherwise use the default functions
        except Exception as e:
            print(f"{e}: Using default functions")
            PATH = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "statistics"
                / "Sluitpeilen"
            )
            if self.get_water_system() in [
                WaterSystem.RHINE_TIDAL,
                WaterSystem.EUROPOORT,
            ]:
                table = pd.read_csv(
                    PATH / "Sluitfunctie Europoortkering Rijn 2017.csv", delimiter=";"
                )
            elif self.get_water_system() == WaterSystem.MEUSE_TIDAL:
                table = pd.read_csv(
                    PATH / "Sluitfunctie Europoortkering Maas 2017.csv", delimiter=";"
                )
            else:
                raise (
                    f"[ERROR] No closing levels for water system '{self.get_water_system().name}'."
                )

        # All columns to lower
        table.columns = table.columns.str.lower()

        # Rename
        table.rename(
            columns={
                "windrichting": "r",
                "afvoer": "q",
                "windsnelheid": "u",
                "zeewaterstand": "m",
            },
            inplace=True,
        )

        # Return table
        return table

    def get_closing_levels_table_eastern_scheldt(self):
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

    def get_closing_situations_eastern_scheldt(self) -> dict:
        """
        Read the closing situations from the database (ClosingSituationId : (Description : FailingLocks)).
        e.g. 1 : ("Reguliere sluiting", 0)

        Only works for the Eastern Scheldt.
        """
        # Check watersystem
        if self.get_water_system() != WaterSystem.EASTERN_SCHELDT:
            raise ValueError(
                "[ERROR] Function can only be called for the Eastern Scheldt"
            )

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

    def get_result_table_eastern_scheldt(
        self, hrdlocation: Union[str, Settings]
    ) -> pd.DataFrame:
        """
        Function to export the loadcombinations of a location to a pandas DataFrame

        Parameters
        ----------
        naam : str
            Locationname
        """
        # Obtain HRDLocationId
        hrdlocationid = self.get_hrdlocation_id(hrdlocation)
        with DatabaseSettings() as database:
            ivids = database.get_input_variable_ids()
            rvids = database.get_result_variable_ids()

        # First collect the dataids. Also replace wind direction ids with real ids
        SQL = """
        SELECT D.HydraulicLoadId, D.ClosingSituationId, W.Direction AS "Wind direction"
        FROM HydroDynamicData D
        INNER JOIN HRDWindDirections W ON D.HRDWindDirectionId=W.HRDWindDirectionId;"""
        dataids = pd.read_sql(SQL, self.con, index_col="HydraulicLoadId")
        dataids.rename(
            columns={"Wind direction": "r", "ClosingSituationId": "k"}, inplace=True
        )

        # Collect the result data. Replace HRDResultColumnId with variable id's
        SQL = """
        SELECT RD.HydraulicLoadId, RV.ResultVariableId, RD.Value
        FROM HydroDynamicResultData RD
        INNER JOIN HRDResultVariables RV ON RD.HRDResultColumnId = RV.HRDResultColumnId
        WHERE HRDLocationId = {};""".format(hrdlocationid)
        resultdata = pd.read_sql(
            SQL, self.con, index_col=["HydraulicLoadId", "ResultVariableId"]
        ).unstack()

        # Reduce columnindex to single level index (without 'Value')
        resultdata.columns = [
            rvids[rid] for rid in resultdata.columns.get_level_values(1)
        ]

        # Create dictionary for mapping HRDInputColumnId to InputVariableId
        SQL = """
        SELECT ID.HydraulicLoadId, IV.InputVariableId, ID.Value
        FROM HydroDynamicInputData ID
        INNER JOIN HRDInputVariables IV ON ID.HRDInputColumnId = IV.HRDInputColumnId"""
        inputdata = pd.read_sql(
            SQL, self.con, index_col=["HydraulicLoadId", "InputVariableId"]
        ).unstack()

        # Reduce columnindex to single level index (without 'Value')
        inputdata.columns = [
            ivids[ivid] for ivid in inputdata.columns.get_level_values(1)
        ]

        # Join data and sort values
        resultaat = (
            dataids.join(inputdata).join(resultdata).sort_values(by=["r", "u", "m"])
        )

        # In the WBI2023 the water levels and waves are in the same table, but have different input variables
        # Split the water levels and wave results
        idx = pd.isnull(resultaat["hs"])

        waterlevels = resultaat.loc[idx].dropna(how="all", axis=1)
        waveconditions = resultaat.loc[~idx].dropna(how="all", axis=1)

        # Replace m_os by m for the water level and m for h for the wave conditions
        waterlevels.rename(columns={"m_os": "m"}, inplace=True)
        waveconditions.rename(columns={"m": "h"}, inplace=True)

        return waterlevels, waveconditions
