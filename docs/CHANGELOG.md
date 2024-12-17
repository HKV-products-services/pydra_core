# Migration to git

1. np.product (depreciated v1.25)
    loading.py line 288-292:

    ```py
                    reshaped = tmparr.reshape((tmparr.shape[0], np.prod(tmparr.shape[1:])))
            
            # Otherwise, only reshape the array
            else:
                reshaped = arr.reshape((arr.shape[0], np.prod(arr.shape[1:])))
    ```

1. ruff error:
    `Exp is not defined`

    ```py
        # TODO: Check if this is correct?
        # Save
        if _ip:
            exp = exp + exceedance_probability * _p
        else:
            exp = exceedance_probability * _p
    ```

1. ruff errors (zie eerste paar commits)
    1. src\pydra_core\location\model\statistics\stochastics\model_uncertainty.py:351:16: E713 [*] Test for membership should be `not in`
    1. `== None` -> `is None`
1. Locatie paden werken niet in linux door `\\` & documentatie is nu in linux:

    ```py
    FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/pydra_core/pydra_core/pydra_core/src/pydra_core/data/statistics/Zeewaterstand\\Vlissingen\\CondPovVlissingen_12u_zichtjaar2017_metOnzHeid.txt'
    ```

    Hier kunnen we nog in de data be standen wat aanpassen, voor nu vervangen door check op `\\`
1. In `pydra_condig\io\database_hr.py`:
    'continuing without correlation'-> nu een warning, even kijken hoe we dat ook beter afvangen.