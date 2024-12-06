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
    Exp is not defined?

    ```py
        # TODO: Check if this is correct?
        # Save
        if _ip:
            exp = exp + exceedance_probability * _p
        else:
            exp = exceedance_probability * _p
    ```

1. ruff error:

    ```py
    src\pydra_core\hrdatabase\hrdatabase.py:142:20: E713 [*] Test for membership should be `not in`
        |
    140 |         if self.check_location(hrdlocation):
    141 |             # If the location is not cached, create the location
    142 |             if not hrdlocation in self.locations:
        |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ E713
    143 |                 # Default settings and create location
    144 |                 settings = self.get_settings(hrdlocation)
        |
        = help: Convert to `not in`

    ```
1. src\pydra_core\location\model\statistics\stochastics\model_uncertainty.py:351:16: E713 [*] Test for membership should be `not in`
1. `== None` -> `is None`
1. 