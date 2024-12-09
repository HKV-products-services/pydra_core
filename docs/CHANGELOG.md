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
