# Pydra Core

This is a stable version of the Pydra, containing the core functionality of the Pydra package.
Pydra was started as an experimental Python version of Hydra-NL
Pydra is developed by HKV together with Rijkswaterstaat and Deltares and is published under de GNU GPL-3 license.
Certain submodules have their own licensing.

Currently (alpha), four dll's are excluded:
    - `CombOverloopOverslag64.dll`
    - `dllDikesOvertopping.dll`
    - `DynamicLib-DaF.dll`
    - `feedbackDLL.dll`

install these `$python_dir$\pydra_core\location\profile\lib`, where `python_dir` can be something like `C:\Users\<username>\miniforge3\envs\<env_name>\Lib\site-packages`.

For questions about how to use this package contact `n.vandervegt@hkv.nl`.
