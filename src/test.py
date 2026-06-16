import pydra_core as pydra

hrd = pydra.HRDatabase("test/data/01_RHINE_NON_TIDAL/WBI2017_Bovenrijn_213_v04.sqlite")
loc = hrd.create_location(hrd.get_location_names()[0])

efl = pydra.ExceedanceFrequencyLine("h", False).calculate(loc)

print(efl.interpolate_level(7.0))

print(1)
