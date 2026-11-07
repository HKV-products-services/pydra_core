"""
Fill the BOI2023 scenarios in 'calculation_settings.sqlite' and the folder 'data/statistics/BOI2023' from an
installation of Hydra-NL. Run it again for a new version of Hydra-NL:

    python tools/import_hydranl_statistics.py C:/Software/Hydra-NL_v292

Where the values come from:

* The water systems of pydra_core are coupled to the water systems in 'data/ini/Watersystemen.ini' with the table
  'HYDRA_WATER_SYSTEMS'; the river, lake and sea indices there are checked against those of pydra_core.
* Files of the fresh water systems come from the scenario file ('data/Scenarios/<area>/<scenario>.sce') of the area,
  data block 0, with the river or lake index. Files of the sea come from 'BerParInfoZout.ini'; every one of them must
  also be named in the scenario file, so a scenario cannot change a file unnoticed.
* Numbers come from 'BerParInfoZoet.ini' and 'BerParInfoZout.ini' with the indices of pydra_core ('COUPLING'), the
  rise of the sea and lake levels from the scenario file. A number that Hydra-NL does not index is taken over from
  WBI2017; a file is never taken over.
* The prediction error of the Europoort barrier per zone comes from the section 'KERINGEN'; zone 8 is the seaward
  side of the barriers ('buitensvk').
* A statistics file without numbers (Olst 2075 and 2125 in Hydra-NL 2.9.2) is reported and not used: the setting is
  left out for that scenario.
"""

import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import fiona

DATA = Path(__file__).resolve().parents[1] / "src" / "pydra_core" / "data"
DATABASE = DATA / "settings" / "calculation_settings.sqlite"
FOLDER = "BOI2023"
SCENARIOS = ("BOI2023_2035", "BOI2023_2075_Mn", "BOI2023_2125_Mn")

# Water system of pydra_core -> water system in 'Watersystemen.ini'. Europoort is the seaward side of the storm surge
# barriers ('BuitenSVK'); 'Meuse valley non tidal' has the same statistics as the upper Meuse in WBI2017; the dunes
# have no statistics.
HYDRA_WATER_SYSTEMS = {
    1: "BOVENRIJNLOBITH",
    2: "BOVENMAASBORGHAREN",
    3: "BENEDENRIJNLOBITH",
    4: "BENEDENMAASBORGHAREN",
    5: "IJSSELDELTA",
    6: "VECHTDELTA",
    7: "IJSSELMEER",
    8: "MARKERMEER",
    9: "WADDENZEE",
    10: "WADDENZEE",
    11: "HOLLANDSE_KUST",
    12: "HOLLANDSE_KUST",
    13: "HOLLANDSE_KUST",
    14: "OOSTERSCHELDE",
    15: "WESTERSCHELDE",
    16: None,
    17: "BUITENSVKLOBITH",
    18: "BOVENMAASBORGHAREN",
    19: "GREVELINGENMEER",
    20: "VELUWEMEER",
    21: "VOLKERAKZOOMMEER",
    22: "HOLLANDSCHEIJSSEL",
}

# The index that BOI2023 changed: the discharge of the Meuse tidal is measured at Borgharen (river 4) instead of
# Lith (river 3). Any other difference with the indices of WBI2017 is an error.
STATION_CHANGES = {4: ("river", 3, 4)}

# Part of the Wadden Sea and the coast in the keys of 'BerParInfoZout.ini' ('SIGMAFUNCTIE.9.oost', 'OVKANSU.10.noord')
SEA_PART = {9: "oost", 10: "west", 11: "noord", 12: "midden", 13: "zuid"}

# The closing function of the Europoort barrier per river index
CLOSING_FUNCTION_RIVER = {2: "Rijn", 4: "Maas"}


@dataclass(frozen=True)
class Coupling:
    """
    A number of pydra_core and its key in Hydra-NL: 'FAMILY.<before>.<index>[.<region>]'. 'index' is the column of
    pydra_core that gives the index ('river', 'lake', 'sea' or '' for none), 'factor' converts the unit of Hydra-NL to
    that of pydra_core, 'region' allows a key per region to override the general one.
    """

    setting: str
    ini: str
    family: str
    index: str = ""
    before: tuple[int, ...] = ()
    factor: float = 1.0
    region: bool = False


# The coupling of HyBPo ('hydra_instellingen.py'), confirmed there by the WBI2017 values of pydra_core coming out of
# Hydra-NL 2.8.2. 'IFHM' and 'FU' are percentages: the Volkerak-Zoommeer has 'IFHM = 10' and 'ifh_a = 0.1'.
# 'm_step' of the sea is not coupled: Hydra-NL has 'MSTAP = 0.1' in WBI2017 and BOI2023 alike, pydra_core 0.01; the
# coarser grid is a choice per quantity of the caller, not a change of BOI2023.
COUPLING = (
    Coupling("base_duration", "zoet", "BASISDUUR", "lake", (0,), 24.0),
    Coupling("periods_base_duration", "zoet", "NGEGEVENSBLOK", "lake", (0,)),
    Coupling("block_duration", "zoet", "BLOKDUUR", "", (0,)),
    Coupling("periods_block_duration", "zout", "NPERIODENWHJ"),
    Coupling("a_min", "zoet", "MMIN", "lake"),
    Coupling("a_max", "zoet", "MMAX", "lake"),
    Coupling("a_step", "zoet", "MSTAP", "lake"),
    Coupling("a_limit", "zoet", "MAFTOP", "lake"),
    Coupling("q_min", "zoet", "QMIN", "river", region=True),
    Coupling("q_max", "zoet", "QMAX", "river", region=True),
    Coupling("q_step", "zoet", "QSTAP", "river", region=True),
    Coupling("q_limit", "zoet", "QAFTOP", "river", region=True),
    Coupling("m_max", "zout", "MMAX", "sea"),
    Coupling("u_step", "zout", "USTAP"),
    Coupling("ifh_a", "zoet", "IFHM", "lake", (0,), 0.01),
    Coupling("ifb_a", "zoet", "IFBM", "lake", (0,), 0.01),
    Coupling("ifh_q", "zoet", "IFHQ", "river", (0,), 0.01),
    Coupling("ifb_q", "zoet", "IFBQ", "river", (0,), 0.01),
    Coupling("sigma_aq", "zoet", "SIGMA", "river", (0,)),
    Coupling("transition_lake_wave_shape", "zoet", "VERSCHUIVING", "river", (0,)),
    Coupling("fu", "zoet", "FU", "", (0,), 0.01),
    Coupling("waveshape_pw", "zoet", "PW", "", (0,)),
    Coupling("waveshape_time_step", "zoet", "DISCRSTAPAFVMP"),
    Coupling("transitional_wind", "zoet", "TRANSWIND", "", (0,)),
    Coupling("failure_probability_ramspol", "zoet", "FAALKANSKERING"),
    Coupling("europoort_barrier", "zoet", "KERING"),
    Coupling("europoort_barrier_distribution", "zoet", "VERDELING"),
    Coupling("europoort_barrier_mu", "zoet", "MU"),
    Coupling("europoort_barrier_sigma", "zoet", "SIGMA"),
    Coupling("europoort_barrier_alfa", "zoet", "ALFA"),
    Coupling("hollandsche_ijssel_barrier_alfa", "zoet", "ALFA_HOIJK"),
)


def read_ini(path: Path) -> dict[str, str]:
    """
    Read an ini file of Hydra-NL as one flat dictionary. The families have unique names over the sections, and
    'configparser' does not accept the repeated keys.
    """
    values = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if line and not line.startswith((";", "[")) and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def read_sce(path: Path) -> dict[tuple[str, str], str]:
    """
    Read a scenario file of Hydra-NL by section and key. The same key occurs in more than one section (the wind of
    'Windstatistiek_LAND' and of 'Windenzeestatistiek_LAND').
    """
    values = {}
    section = ""
    for line in path.read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line.strip("[]")
        elif line and not line.startswith(";") and "=" in line:
            key, _, value = line.partition("=")
            values[(section, key.strip())] = value.strip()
    return values


def find_in_sce(sce: dict, key: str, section: str | None = None) -> str:
    """
    Return the value of a key in a scenario file, in one section or in any; the sections that name it must agree.
    """
    found = {value for (sec, k), value in sce.items() if k == key and (section is None or sec == section)}
    if len(found) != 1:
        raise ValueError(f"Key '{key}' (section '{section}') gives {sorted(found)} in the scenario file.")
    return found.pop()


def read_water_systems(hydra: Path) -> dict[str, dict]:
    """
    Read 'Watersystemen.ini': per water system its item number and the columns 'winds', 'wndze', 'Rc' and 'Mc'.
    """
    water_systems = {}
    for line in (hydra / "data" / "ini" / "Watersystemen.ini").read_text(encoding="latin-1").splitlines():
        match = re.match(r"item\.(\d+)=(.+)", line.strip())
        if not match or "|" not in match.group(2):
            continue
        columns = [column.strip() for column in match.group(2).split("|")]
        water_systems[columns[1]] = {"item": int(match.group(1)), "winds": columns[6], "wndze": columns[7], "Rc": columns[17], "Mc": columns[19]}
    return water_systems


def has_numbers(path: Path) -> bool:
    """
    Determine whether a statistics file holds numbers and not only comments: Hydra-NL 2.9.2 names an Olst file for
    2075 that only says it is a dummy.
    """
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("*", ";", "#")):
            continue
        try:
            float(line.split()[0].replace(",", "."))
            return True
        except ValueError:
            continue
    return False


def package_path(hydra_value: str) -> str:
    """
    Turn a file of Hydra-NL ('.\\data\\invoer\\Afvoer\\...%0.txt') into a path in 'data/statistics', with the
    statistical uncertainty ('%0' is '_metOnzHeid' or empty).
    """
    relative = hydra_value.replace("\\", "/").removeprefix("./").removeprefix("data/invoer/").replace("%0", "_metOnzHeid")
    if "%" in relative or not relative:
        raise ValueError(f"Cannot read the file '{hydra_value}'.")
    return f"{FOLDER}/{relative}"


def ini_number(ini: dict, coupling: Coupling, indices: dict[str, int]) -> float | None:
    """
    Return a number from the ini of Hydra-NL for the indices of a water system, None if Hydra-NL has no key for it.
    """
    if coupling.index and coupling.index not in indices:
        return None
    parts = [coupling.family, *map(str, coupling.before)] + ([str(indices[coupling.index])] if coupling.index else [])
    keys = [".".join(parts + [str(indices["region"])])] if coupling.region and "region" in indices else []
    keys.append(".".join(parts))
    for key in keys:
        if key in ini and ini[key] != "":
            return float(ini[key].replace(",", ".")) * coupling.factor
    return None


def main(hydra: Path) -> None:
    zoet = read_ini(hydra / "data" / "rekeninstellingen" / "BerParInfoZoet.ini")
    zout = read_ini(hydra / "data" / "rekeninstellingen" / "BerParInfoZout.ini")
    hydra_water_systems = read_water_systems(hydra)
    invoer = hydra / "data" / "invoer"
    reports = []

    # The WBI2017 rows are the full set of settings per water system
    con = sqlite3.connect(DATABASE)
    wbi2017 = {}
    for water_system, name, value in con.execute("SELECT WaterSystem, SettingName, SettingValue FROM CalculationSettings WHERE Scenario = 'WBI2017' ORDER BY SettingId"):
        wbi2017.setdefault(water_system, {})[name] = value
    if set(wbi2017) != set(HYDRA_WATER_SYSTEMS):
        raise ValueError(f"The water systems of WBI2017 {sorted(wbi2017)} differ from 'HYDRA_WATER_SYSTEMS'.")

    # Check the coupling against the indices in 'Watersystemen.ini'
    indices = {}
    for water_system, hydra_name in HYDRA_WATER_SYSTEMS.items():
        indices[water_system] = {name: int(float(wbi2017[water_system][name])) for name in ("river", "lake", "sea", "region") if name in wbi2017[water_system]}
        if hydra_name is None:
            continue
        columns = hydra_water_systems[hydra_name]
        for name, column in (("river", "Rc"), ("lake", "Mc"), ("sea", "Mc")):
            if name in indices[water_system] and str(indices[water_system][name]) != columns[column]:
                if STATION_CHANGES.get(water_system) != (name, indices[water_system][name], int(columns[column])):
                    raise ValueError(f"Water system {water_system}: '{name}' {indices[water_system][name]} differs from '{column}' {columns[column]} of {hydra_name}.")
                reports.append(f"Water system {water_system} ({hydra_name}): '{name}' {indices[water_system][name]} in WBI2017, {columns[column]} in Hydra-NL.")
                indices[water_system][name] = int(columns[column])

    # Without a river index in pydra_core the numbers of the river are looked up with that of Hydra-NL (the
    # Hollandsche IJssel and the Rhine); otherwise they would keep their WBI2017 value unnoticed
    lookup = {water_system: dict(index) for water_system, index in indices.items()}
    for water_system, hydra_name in HYDRA_WATER_SYSTEMS.items():
        if hydra_name is not None and "river" not in lookup[water_system] and hydra_water_systems[hydra_name]["Rc"] != "-":
            lookup[water_system]["river"] = int(hydra_water_systems[hydra_name]["Rc"])
            reports.append(f"Water system {water_system} ({hydra_name}): no river index in pydra_core, river {lookup[water_system]['river']} of Hydra-NL used.")

    # The prediction error of the Europoort barrier per zone of 'lower_river_settings.shp'
    with fiona.open(DATA / "settings" / "lower_river_settings.shp") as shp:
        zones = sorted(feature["properties"]["number"] for feature in shp)
    general = tuple(float(zoet[key]) for key in ("MU", "SIGMA", "ALFA"))
    seaward = tuple(float(zoet[f"{key}.8.buitensvk"]) for key in ("MU", "SIGMA", "ALFA"))
    if seaward != (0.5, 0.001, 0.0):
        raise ValueError(f"Zone 8 of the Europoort barrier is {seaward} instead of (0.5, 0.001, 0).")
    epsilon_rows = [(scenario, zone, *(seaward if zone == 8 else general)) for scenario in SCENARIOS for zone in zones]

    # Settings per scenario and water system
    setting_rows = []
    subsystem_rows = {"CoastSubSystems": [], "WaddenSeaSubSystems": [], "WesternScheldtSubSystems": []}
    from_wbi2017 = {}
    used_files = set()
    for scenario in SCENARIOS:
        for water_system, hydra_name in HYDRA_WATER_SYSTEMS.items():
            # Without statistics the settings are numbers that Hydra-NL does not index
            if hydra_name is None:
                if any("/" in value for value in wbi2017[water_system].values()):
                    raise ValueError(f"Water system {water_system} has files but no water system in Hydra-NL.")
                setting_rows += [(scenario, water_system, name, value) for name, value in wbi2017[water_system].items()]
                for name in wbi2017[water_system]:
                    from_wbi2017.setdefault(name, set()).add(water_system)
                continue

            area = next(p for p in (hydra / "data" / "Scenarios").iterdir() if p.name.upper() == hydra_name)
            sce = read_sce(area / f"{scenario}.sce")
            columns = hydra_water_systems[hydra_name]
            index = indices[water_system]
            files = {}
            numbers = {}

            # Sea: files from 'BerParInfoZout.ini', the rise from the scenario file
            if "sea" in index:
                sea = index["sea"]
                part = SEA_PART.get(water_system)
                for setting, family in (("wind_speed_probability", "OVKANSU"), ("sigma_function", "SIGMAFUNCTIE"), ("wind_direction_probability", "KANSENRICHTING"), ("sea_level_probability", "OVKANSM")):
                    key = next((k for k in (f"{family}.{sea}.{part}", f"{family}.{sea}") if zout.get(k)), None)
                    if key:
                        files[setting] = zout[key]
                for setting, key in (("barrier_scenario_probability", "KANSENFAALSCENARIO"), ("barrier_closing_probability", "KANSSLUITINGEN"), ("storm_surge_duration_probability", "KANSENSTORMDUUR"), ("phase_surge_tide_probability", "KANSENFASEVERSCHIL")):
                    if setting in wbi2017[water_system]:
                        files[setting] = zout[key]
                numbers["sea_level_rise"] = float(find_in_sce(sce, f"ZWS_STIJGING.{sea}"))

                # The three sea level points per subsystem, in the order of Hydra-NL
                table = {9: "WaddenSeaSubSystems", 10: "CoastSubSystems", 11: "WesternScheldtSubSystems"}.get(sea)
                if table and not any(row[0] == scenario for row in subsystem_rows[table]):
                    # The Western Scheldt has one subsystem without a number, the others 'OVKANSM_HOEKPUNT1.<sea>.<n>'
                    if f"OVKANSM_HOEKPUNT1.{sea}" in zout:
                        suffixes = [f"{sea}"]
                    else:
                        suffixes = [key.removeprefix("OVKANSM_HOEKPUNT1.") for key in zout if key.startswith(f"OVKANSM_HOEKPUNT1.{sea}.")]
                        if suffixes != [f"{sea}.{n}" for n in range(len(suffixes))]:
                            raise ValueError(f"The subsystems of sea {sea} are not numbered 0, 1, ...: {suffixes}")
                    for subsystem_id, suffix in enumerate(suffixes, start=1):
                        points = [package_path(zout[f"OVKANSM_HOEKPUNT{point}.{suffix}"]) for point in (1, 2, 3)]
                        subsystem_rows[table].append((scenario, subsystem_id, *points))
                        used_files.update(points)

                # A scenario file must name every file of the sea, so a scenario cannot change one unnoticed
                named = {package_path(value) for value in sce.values() if value.lower().endswith(".txt")}
                points = {point for row in subsystem_rows.get(table, []) if row[0] == scenario for point in row[2:]}
                missing = {package_path(value) for value in files.values()} - named | (points - named)
                if missing:
                    raise ValueError(f"{scenario} {hydra_name}: the scenario file does not name {sorted(missing)}.")

            # Fresh water: files from the scenario file, data block 0
            else:
                river, lake = columns["Rc"], columns["Mc"]
                wind_section = "Windstatistiek_LAND" if columns["winds"] == "LAND" else "Windenzeestatistiek_LAND"
                files["wind_direction_probability"] = find_in_sce(sce, "File_MomKansWindrichting.0", wind_section)
                files["wind_speed_probability"] = find_in_sce(sce, "File_Ovkanswindsnelheid.0", wind_section)
                if "discharge_probability" in wbi2017[water_system]:
                    files["discharge_probability"] = find_in_sce(sce, f"File_OvkansPiekafvoer.0.{river}")
                    files["top_duration_q"] = find_in_sce(sce, f"File_TopDurenAfvoertrap.0.{river}")
                if "lake_level_probability" in wbi2017[water_system]:
                    files["lake_level_probability"] = find_in_sce(sce, f"File_OvkansPM.0.{lake}")
                    files["top_duration_a"] = find_in_sce(sce, f"File_tdmeerpeiltrap.0.{lake}")
                    numbers["lake_level_rise"] = float(find_in_sce(sce, f"Val_Meerpeilstijging.0.{lake}"))
                if "sea_level_probability" in wbi2017[water_system]:
                    files["sea_level_probability"] = find_in_sce(sce, "File_Kans12.0", "Windenzeestatistiek_LAND")
                    files["wind_sea_level_probability"] = find_in_sce(sce, "File_West12.0", "Windenzeestatistiek_LAND")
                    numbers["sea_level_rise"] = float(find_in_sce(sce, "Val_Zeespiegelstijging.0", "Windenzeestatistiek_LAND"))
                if "europoort_barrier_closing_function" in wbi2017[water_system]:
                    name = CLOSING_FUNCTION_RIVER[int(river)]
                    (csv,) = (hydra / "data" / "ini").glob(f"Sluitfunctie_EPK_{name}_*.csv")
                    files["europoort_barrier_closing_function"] = f".\\data\\invoer\\Sluitpeilen\\{csv.name}"

            # Numbers from the ini; the shift from Hoek van Holland to Maasmond does not exist in BOI2023
            for coupling in COUPLING:
                if coupling.setting in wbi2017[water_system]:
                    number = ini_number(zoet if coupling.ini == "zoet" else zout, coupling, lookup[water_system])
                    if number is not None:
                        numbers[coupling.setting] = number
            if "sea_level_translation" in wbi2017[water_system]:
                numbers["sea_level_translation"] = 0.0
            numbers.update({name: float(value) for name, value in index.items() if value != int(float(wbi2017[water_system][name]))})

            # One row per setting of WBI2017; a file is never taken over
            for name, wbi2017_value in wbi2017[water_system].items():
                if name in files:
                    path = package_path(files[name])
                    source = DATA / "statistics" / path if name == "europoort_barrier_closing_function" else invoer / path.removeprefix(f"{FOLDER}/")
                    if name != "europoort_barrier_closing_function" and not has_numbers(source):
                        reports.append(f"{scenario} water system {water_system} ({hydra_name}): '{name}' is {source.name} without numbers; left out.")
                        continue
                    setting_rows.append((scenario, water_system, name, path))
                    used_files.add(path)
                elif name in numbers:
                    setting_rows.append((scenario, water_system, name, repr(float(numbers[name]))))
                elif "/" in wbi2017_value:
                    raise ValueError(f"{scenario} water system {water_system}: no file for '{name}' in Hydra-NL.")
                else:
                    setting_rows.append((scenario, water_system, name, wbi2017_value))
                    from_wbi2017.setdefault(name, set()).add(water_system)

    # The folder BOI2023: every BOI2023 file of 'data/invoer' and the closing functions of 'data/ini'
    target = DATA / "statistics" / FOLDER
    if target.exists():
        shutil.rmtree(target)
    for source in sorted(invoer.rglob("*")):
        if source.is_file() and "BOI2023" in source.name:
            if not has_numbers(source):
                reports.append(f"{source.relative_to(invoer)} has no numbers; not copied.")
                continue
            destination = target / source.relative_to(invoer)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    (target / "Sluitpeilen").mkdir(parents=True)
    for source in sorted((hydra / "data" / "ini").glob("*.csv")):
        shutil.copyfile(source, target / "Sluitpeilen" / source.name)

    # Every file in the settings, with and without statistical uncertainty, must be in the folder
    for path in sorted(used_files):
        for variant in {path, path.replace("_metOnzHeid", "")}:
            if not (DATA / "statistics" / variant).is_file():
                source = invoer / variant.removeprefix(f"{FOLDER}/")
                if not source.is_file():
                    raise FileNotFoundError(f"{variant} is not in Hydra-NL.")
                (DATA / "statistics" / variant).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, DATA / "statistics" / variant)

    # Write the rows of the scenarios, replacing those of an earlier run
    with con:
        for table in ["CalculationSettings", "CoastSubSystems", "WaddenSeaSubSystems", "WesternScheldtSubSystems", "EuropoortBarrierZones"]:
            con.execute(f"DELETE FROM {table} WHERE Scenario IN ({','.join('?' * len(SCENARIOS))})", SCENARIOS)
        next_id = con.execute("SELECT MAX(SettingId) FROM CalculationSettings").fetchone()[0] + 1
        con.executemany(
            "INSERT INTO CalculationSettings (SettingId, Scenario, WaterSystem, SettingName, SettingValue) VALUES (?, ?, ?, ?, ?)",
            [(next_id + n, *row) for n, row in enumerate(setting_rows)],
        )
        for table, rows in subsystem_rows.items():
            con.executemany(f"INSERT INTO {table} (Scenario, SubSystemId, Point1, Point2, Point3) VALUES (?, ?, ?, ?, ?)", rows)
        con.executemany("INSERT INTO EuropoortBarrierZones (Scenario, Zone, Mu, Sigma, Alfa) VALUES (?, ?, ?, ?, ?)", epsilon_rows)
    con.execute("VACUUM")
    con.close()

    print(f"{len(setting_rows)} settings in {len(SCENARIOS)} scenarios, {sum(len(rows) for rows in subsystem_rows.values())} subsystems, {len(epsilon_rows)} zones.")
    print("Taken over from WBI2017 (not indexed by Hydra-NL), per setting the water systems:")
    for name, water_systems in sorted(from_wbi2017.items()):
        print(f"  {name}: {sorted(water_systems)}")
    for report in reports:
        print(f"NOTE: {report}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
