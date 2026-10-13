import pandas as pd

from typing import Union

from .hrdatabase_wbi2017_generic import HRDReaderWBI2017
from ...location.settings.settings import Settings


class HRDReaderWBI2017Dunes(HRDReaderWBI2017):
    """
    HR database reader for WBI2017 dune databases.

    The dunes HRD (region 16) differs structurally from other HRDs:
    - No ClosingSituationId / wind directions in HydroDynamicData
    - Input axes are u-values (u_wl, u_stat, u_Hs, u_Tp) as standard-normal variates
    - Results are: water level (h), Hs, Tp, wave direction, tidal amplitude,
      wave directional spread, tide-surge phase difference
    """

    def get_input_variables(self) -> list:
        """
        Return the input variables

        Returns
        -------
        list
            List with input variables
        """
        # Query: use ColumnName directly from HRDInputVariables
        sql = "SELECT ColumnName FROM HRDInputVariables ORDER BY HRDInputColumnId"
        data = [row[0] for row in self.con.execute(sql).fetchall()]

        return data

    def get_result_variables(self) -> list:
        """
        Return the result variables

        Returns
        -------
        list
            List with result variables
        """
        # Query: use ColumnName directly from HRDResultVariables
        sql = "SELECT ColumnName FROM HRDResultVariables ORDER BY HRDResultColumnId"
        data = [row[0] for row in self.con.execute(sql).fetchall()]

        return data

    def get_result_table(self, hrdlocation: Union[str, Settings]) -> pd.DataFrame:
        """
        Read the load combinations of a dunes HRD location into a pandas DataFrame.

        Parameters
        ----------
        hrdlocation : Union[str, Settings]
            HRDLocation name or Settings object

        Returns
        -------
        pd.DataFrame
            DataFrame with all load combinations indexed by HydroDynamicDataId,
            columns = input u-values + result variables,
            sorted by u_wl > u_stat > u_Hs > u_Tp > remaining input axes.
        """
        # Obtain HRDLocationId
        hrdlocationid = self.get_hrdlocation_id(hrdlocation)

        # --- HydroDynamicData: one row per (u_wl, u_stat, u_Hs, u_Tp) combination ---
        # For dunes there is no ClosingSituationId or HRDWindDirectionId column;
        # the only foreign key besides HRDLocationId is HydroDynamicDataId itself.
        query = (
            f"SELECT * FROM HydroDynamicData WHERE HRDLocationId = {hrdlocationid}"
        )
        hydrodynamicdata = pd.read_sql(query, self.con, index_col="HydroDynamicDataId")

        hydrodynamicdataids = ",".join(
            hydrodynamicdata.index.values.astype(str).tolist()
        )

        # --- HydroDynamicInputData: u-value axes (u_wl, u_stat, u_Hs, u_Tp) ---
        # ColumnName values: 'Sea water level (u)', 'Uncertainty water level (u)',
        #                    'Wave height (u)', 'Wave period (u)'
        query = """
        SELECT ID.HydroDynamicDataId, IV.ColumnName, ID.Value
        FROM HydroDynamicInputData ID
        INNER JOIN HRDInputVariables IV ON ID.HRDInputColumnId = IV.HRDInputColumnId
        WHERE ID.HydroDynamicDataId IN ({ids});
        """.format(ids=hydrodynamicdataids)
        hydrodynamicinputdata = (
            pd.read_sql(query, self.con, index_col=["HydroDynamicDataId", "ColumnName"])
            .unstack()
        )
        hydrodynamicinputdata.columns = hydrodynamicinputdata.columns.get_level_values(1)
        ivcols = hydrodynamicinputdata.columns.tolist()

        # --- HydroDynamicResultData: h, Hs, Tp, wave direction, tidal amplitude, etc. ---
        query = """
        SELECT RD.HydroDynamicDataId, RV.ColumnName, RD.Value
        FROM HydroDynamicResultData RD
        INNER JOIN HRDResultVariables RV ON RD.HRDResultColumnId = RV.HRDResultColumnId
        WHERE RD.HydroDynamicDataId IN ({ids});
        """.format(ids=hydrodynamicdataids)
        hydrodynamicresultdata = (
            pd.read_sql(query, self.con, index_col=["HydroDynamicDataId", "ColumnName"])
            .unstack()
        )
        hydrodynamicresultdata.columns = hydrodynamicresultdata.columns.get_level_values(1)

        # --- Merge and clean up ---
        # Drop HRDLocationId; no wind-direction or closing-situation column to replace
        results = (
            hydrodynamicdata
            .drop(columns=["HRDLocationId"], errors="ignore")
            .join(hydrodynamicinputdata)
            .join(hydrodynamicresultdata)
        )

        # Sort by u_wl > u_stat > u_Hs > u_Tp, then any remaining input axes.
        # Matching on ColumnName substrings from HRDInputVariables.
        priority_patterns = ["sea water", "uncertainty", "wave height", "wave period"]
        sort_cols = []
        for pattern in priority_patterns:
            match = next((c for c in ivcols if pattern in c.lower()), None)
            if match:
                sort_cols.append(match)
        sort_cols += [c for c in ivcols if c not in sort_cols]
        results.sort_values(by=sort_cols, inplace=True)

        return results
