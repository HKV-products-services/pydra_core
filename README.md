# Pydra Core

This is a stable version of the Pydra, containing the core functionality of the Pydra package.
Pydra was started as an experimental Python version of Hydra-NL
Pydra is developed by HKV together with Rijkswaterstaat and Deltares and is published under de GNU GPL-3 license.
Certain submodules have their own licensing.

## Getting started

### Using install (in future)

run `pip install pydra_core`

### developing with pixi

To manage the environment we use Pixi.

#### windows

```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

#### Linux/Mac

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

#### installing

With the `Pixi` command in powershell install the python environment:

```bash
 cd ..../pydra_core
 pixi install
```

The `pixi.lock` file loads the correct packages and downloads to the `.pixi` file, you can use this environment in developing and resting.

Run `pixi run quarto-render` to create documentation locally, this will be in the `docs\docs` file.

## excluded dlls

Currently (alpha), four dll's are excluded:

- `CombOverloopOverslag64.dll`
- `dllDikesOvertopping.dll`
- `DynamicLib-DaF.dll`
- `feedbackDLL.dll`

install these `$python_dir$\pydra_core\location\profile\lib`, where `python_dir` can be something like `C:\Users\<username>\miniforge3\envs\<env_name>\Lib\site-packages`.

For questions about how to use this package contact `n.vandervegt@hkv.nl`.
