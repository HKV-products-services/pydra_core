# Pydra Core

Pydra Core is a Python package containing (part of) the functionalities from Hydra-NL.
Pydra was started as an experimental Python version of Hydra-NL, developed by HKV together with Rijkswaterstaat WVL.
Pydra is maintained by HKV. For questions about how to use this package, contact `n.vandervegt@hkv.nl`.

Currently the following Hydra-NL functionalities are included:
* Exceedance Frequency Lines (Based on marginal statistics)
* HBN (Probabilistically Determined Required Crest Height)
* Profiles (Run-up, Average overtopping discharge, HBN)

For the following water systems:
* Non-Tidal Rivers (Bovenrivieren 01 Rijn, 02 Maas, 18 Maasvallei)
* Tidal Rivers (Benedenrivieren 03 Rijn, 04 Maas)
* Coast (09/10 Waddenzee Oost en West; 11/12/13 Hollandse Kust Noord, Midden en Zuid; 15 Westerschelde)
* Eastern Scheldt 'WBI2023' databases (14 Oosterschelde)
* Lakes (07 IJsselmeer; 08 Markermeer)
* Vecht-IJssel Delta (05 IJssel Delta; 06 Vecht Delta)

Pydra Core is published under de GNU GPL-3 license. Certain submodules have their own licensing.

## Getting started

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
> The `dllDikesOvertopping.dll`, `feedbackDLL.dll`, and `libiomp5md.dll` are part of [DiKErnel](https://github.com/Deltares/DiKErnel) which is made by [Deltares](https://www.deltares.nl/en) and published under the
> [GNU AFFERO GPL v3](https://github.com/Deltares/DiKErnel/blob/master/Licenses/Deltares/DikesOvertopping.LICENSE) license.
> These dll files are only included to make use of this package easier.
