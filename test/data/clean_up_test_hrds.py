import numpy as np
import pandas as pd
import sqlite3 as sql

from pathlib import Path


# Load results
path = Path(__file__).parent
df = pd.read_excel(path / "hydranl_bor_results.xlsx")

# Per database
for hrd, deeldf in df.groupby(by = "HRDatabase"):

    # Select WS
    ws = np.unique(deeldf['WaterSystem'])
    assert len(ws) == 1
    ws = ws[0]

    # Path
    hrd_path = path / ws / hrd
    con = sql.Connection(hrd_path)
    cur = con.cursor()

    # Select HRDLoc to preserve
    hrdloc = np.unique(deeldf['HRLocation'])
    placeholders = ",".join("?" * len(hrdloc))
    query = f"""
        SELECT HRDLocationId
        FROM HRDLocations
        WHERE Name IN ({placeholders})
    """
    cur.execute(query, tuple(hrdloc))
    results = cur.fetchall()
    hrd_ids = [row[0] for row in results]

    # Delete all hrdlocations
    placeholders = ",".join("?" * len(hrd_ids))
    delete_query = f"""
        DELETE FROM HRDLocations
        WHERE HRDLocationId NOT IN ({placeholders})
    """
    cur.execute(delete_query, tuple(hrd_ids))

    # Delete all model uncertainties
    placeholders = ",".join("?" * len(hrd_ids))
    delete_query = f"""
        DELETE FROM UncertaintyModelFactor
        WHERE HRDLocationId NOT IN ({placeholders})
    """
    cur.execute(delete_query, tuple(hrd_ids))

    # Select loading to save
    placeholders = ",".join("?" * len(hrd_ids))
    select_query = f"""
        SELECT HydroDynamicDataId
        FROM HydroDynamicData
        WHERE HRDLocationId IN ({placeholders})
    """
    cur.execute(select_query, tuple(hrd_ids))
    hdd_ids = [row[0] for row in cur.fetchall()]
    con.commit()

    # Empty other loading
    ph_hdd = ",".join("?" * len(hdd_ids))

    # Delete from HydroDynamicData
    cur.execute(f"""
        DELETE FROM HydroDynamicData
        WHERE HydroDynamicDataId NOT IN ({ph_hdd})
    """, tuple(hdd_ids))
    con.commit()

    # Delete from HydroDynamicInputData
    cur.execute(f"""
        DELETE FROM HydroDynamicInputData
        WHERE HydroDynamicDataId NOT IN ({ph_hdd})
    """, tuple(hdd_ids))
    con.commit()

    # Delete from HydroDynamicResultData
    cur.execute(f"""
        DELETE FROM HydroDynamicResultData
        WHERE HydroDynamicDataId NOT IN ({ph_hdd})
    """, tuple(hdd_ids))

    # Commit
    con.commit()
    cur.execute("VACUUM")
    con.commit()
    cur.close()
    con.close()
