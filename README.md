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

## Certain submodules have their own licensing

The files `CombOverloopOverslag64.dll` and `DynamicLib-DaF.dll` are obtained from [Hydra-NL v2.8.2](https://iplo.nl/thema/water/applicaties-modellen/waterveiligheidsmodellen/hydra-nl/) which is freely available through the dutch government website.

The `dllDikesOvertopping.dll` and `feedbackDLL.dll` are part of [DiKErnel](https://github.com/Deltares/DiKErnel) which is made by [Deltares](https://www.deltares.nl/en) and published under the
 [GNU AFFERO GPL v3](https://github.com/Deltares/DiKErnel/blob/master/Licenses/Deltares/DikesOvertopping.LICENSE) license.
These dll files are only included to make use of this package easier.

For questions about how to use this package contact `n.vandervegt@hkv.nl`.
