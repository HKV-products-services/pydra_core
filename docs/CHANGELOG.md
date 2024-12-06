# Migration to git

1. np.product (depreciated v1.25)
    loading.py line 288-292:
    ```py
                    reshaped = tmparr.reshape((tmparr.shape[0], np.prod(tmparr.shape[1:])))
            
            # Otherwise, only reshape the array
            else:
                reshaped = arr.reshape((arr.shape[0], np.prod(arr.shape[1:])))
    ```

