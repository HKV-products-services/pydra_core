# Pydra Core

This is a stable version of the Pydra, containing the core functionality of the Pydra package.
Pydra was started as an experimental Python version of Hydra-NL
Pydra is developed by HKV together with Rijkswaterstaat and Deltares and is published under de GNU GPL-3 license.

## getting started

To download the package run `pip install pydra-core`

```py
import pydra_core

profile = pydra_core.Profile("Borselle")
profile.set_dike_crest_level(10.75)
profile.set_dike_orientation(225)
profile.set_dike_geometry([-30, 30], [-10, 10])
profile.draw_profile()
```

## Certain submodules have their own licensing

> The files `CombOverloopOverslag64.dll` and `DynamicLib-DaF.dll` are obtained from [Hydra-NL v2.8.2](https://iplo.nl/thema/water/applicaties-modellen/waterveiligheidsmodellen/hydra-nl/) which is freely available through the dutch government and have been published with permission.
>
> The `dllDikesOvertopping.dll` and `feedbackDLL.dll` are part of [DiKErnel](https://github.com/Deltares/DiKErnel) which is made by [Deltares](https://www.deltares.nl/en) and published under the
> [GNU AFFERO GPL v3](https://github.com/Deltares/DiKErnel/blob/master/Licenses/Deltares/DikesOvertopping.LICENSE) license.
> These dll files are only included to make use of this package easier.
