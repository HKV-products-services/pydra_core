import logging
import os
import re
import shutil
import subprocess
from typing import Optional
from pathlib import Path

import netCDF4
import numpy as np

from ..location.location import Location

log = logging.getLogger(__name__)

XBEACH_PATH = Path(__file__).parent.parent / "location" / "profile" / "lib" / "XBeach_v1.24.5956_BOI_phase3"
_D50_BND = Path(__file__).parent.parent / "data" / "settings" / "D50.bnd"


def _parse_kv(name: str) -> Optional[int]:
    """Extract kustvaknummer from a location name of the form <word>_<Kv>_<Nr>."""
    m = re.match(r"^[A-Za-z][A-Za-z0-9\-]*_(\d+)_\d+$", name.strip())
    return int(m.group(1)) if m else None


def _get_d50(kv: int, nr: float) -> float:
    """Return D50 [m] interpolated from D50.bnd for the given kustvaknummer and metrering (dam)."""
    rows = []
    with open(_D50_BND) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("Kv"):
                continue
            parts = line.split()
            if len(parts) < 3 or not parts[2]:
                continue
            if int(parts[0]) == kv:
                rows.append((float(parts[1]), float(parts[2])))
    if not rows:
        raise ValueError(f"No D50 data for kustvaknummer {kv} in {_D50_BND}")
    rows.sort()
    nrs  = np.array([r[0] for r in rows])
    d50s = np.array([r[1] for r in rows])
    return float(np.interp(nr, nrs, d50s))

T_M2 = 12.4206  # M2 tidal period [hours]


def _m2_tide(t_h: np.ndarray, tidal_amp: float, fase: float) -> np.ndarray:
    """
    Synthetic M2 tidal signal scaled to ``tidal_amp`` (mean tidal amplitude
    from HRD).  The tidal peak aligns with ``t = fase`` hours, where t = 0 is
    the storm-surge peak.
    """
    return tidal_amp * np.cos(2.0 * np.pi / T_M2 * (t_h - fase))


def _water_level_evolution(
    t_h: np.ndarray,
    surge_height: float,
    opzetduur: float,
    opzettopduur: float,
    tide_arr: np.ndarray,
) -> np.ndarray:
    """
    Combine tidal time series with a trapezoidal storm-surge shape centred at
    t = 0 (storm-surge peak).  Returns the total water level on ``t_h``.
    """
    tpoints = [
        t_h[0], -opzetduur / 2, -opzettopduur / 2, 0.0,
        opzettopduur / 2,  opzetduur / 2, t_h[-1],
    ]
    spoints = [0.0, 0.0, surge_height - 0.1, surge_height, surge_height - 0.1, 0.0, 0.0]
    surge = np.interp(t_h, tpoints, spoints)
    return tide_arr + surge





class XbeachComputation:
    """
    Run a 1D XBeach dune-erosion simulation driven by FORM hydraulic loads.

    Parameters
    ----------
    hydraulic_loads : list
        Output of HydraulicLoadsDunes.calculate_location — a list of dicts
        containing rp, hs, tp, tidal_amp, jarkus_raai, …
    storm_duration_hours : int
        Total storm duration in hours.  Water level follows a half-cosine that
        peaks at ``rp`` at the midpoint; waves are constant at (hs, tp).
        Default: 24 h.
    xbeach_path : str, optional
        Path to the XBeach template folder.  Defaults to the BOI-2023 build at
        ``XBEACH_PATH``.
    """

    def __init__(
        self,
        hydraulic_loads: list,
        xbeach_path: Optional[str] = None,
        surge_duration: float = 44.0,
        surge_top_duration: float = 2.0,
    ):
        self.hydraulic_loads    = hydraulic_loads
        self.xbeach_path        = xbeach_path or XBEACH_PATH
        self.surge_duration     = surge_duration
        self.surge_top_duration = surge_top_duration

    # ─────────────────────────────────────────────────────────────────────────

    def calculate_1D(
        self,
        location: Location,
        save: bool = False,
        make_plot: bool = False,
    ) -> dict:
        """
        Write XBeach boundary conditions, run the model, and return results.

        Parameters
        ----------
        location : Location
            Used for output-path derivation when save=True.
        save : bool
            Copy the netCDF output next to the .sqlite database when True.
        make_plot : bool
            Save a PDF profile plot when True.

        Returns
        -------
        dict
            resthoogte  – maximum residual dune height after storm [m+NAP]
            xb_path     – XBeach template folder used
            nc_path     – path to xboutput.nc inside the template folder
        """
        hl        = self.hydraulic_loads[0]
        rp        = float(hl["rp"])
        hs        = float(hl["hs"])
        tp        = float(hl["tp"])
        raai      = int(hl.get("jarkus_raai", 0))
        xb_path   = self.xbeach_path

        # ── Per-location run folder ───────────────────────────────────────────
        loc_name = getattr(location.settings, "location", None) or f"raai_{raai}"
        db_dir   = os.path.dirname(getattr(location.settings, "database_path", None) or "") or "."
        run_dir  = os.path.join(db_dir, str(loc_name))
        os.makedirs(run_dir, exist_ok=True)
        for _src in Path(xb_path).iterdir():
            if _src.is_file():
                shutil.copy2(str(_src), run_dir)
        xb_path  = run_dir

        # ── Storm time series ─────────────────────────────────────────────────
        fase      = float(hl.get("fase",      3.5))
        spread    = float(hl.get("spread",    6.0))
        tidal_amp = float(hl.get("getij",     1.0))

        # ── Full surge window (centred on t = 0) ─────────────────────────────
        # Start at the tidal peak (t = fase mod T_M2) that falls just before
        # the surge onset at -surge_half, using fase directly from the HRD.
        import math
        surge_half = self.surge_duration / 2.0
        n_periods  = math.ceil((fase + surge_half) / T_M2) - 1
        n_start    = int(np.floor(fase - n_periods * T_M2))
        t_full     = np.arange(float(n_start), surge_half + 2.0, 1.0)
        tide_full  = _m2_tide(t_full, tidal_amp, fase)

        # Find surge_height such that max(tide + surge) == rp exactly
        surge_height = rp
        for _ in range(30):
            wl_full = _water_level_evolution(
                t_full, surge_height, self.surge_duration, self.surge_top_duration, tide_full
            )
            error = rp - float(np.max(wl_full))
            surge_height += error
            if abs(error) < 1e-6:
                break

        # ── Trim end: cut off once wl has dropped 2 m below rp after the peak ──
        i_peak = int(np.argmax(wl_full))
        active_tail = np.asarray(wl_full[i_peak:] >= (rp - 2.0))
        tail_len = int(np.argmin(active_tail)) if not active_tail.all() else int(active_tail.size)
        i1 = min(int(wl_full.size) - 1, i_peak + tail_len)
        t_h = t_full[: i1 + 1]
        wl  = wl_full[: i1 + 1]
        t_s = (t_h - t_h[0]) * 3600.0

        log.info(
            "M2 tide: tidal_amp = %.3f m, tidal_phase = %.2f h, surge_height = %.3f m, "
            "max(wl) = %.3f m (rp %.3f m), tstop = %.0f s (%.1f h)",
            tidal_amp, fase, surge_height, float(wl.max()), rp,
            float(t_s[-1]), float(t_s[-1]) / 3600.0,
        )

        # ── tide.txt  [time_s  wl_left  zs_min]  at 30-minute resolution ───────
        # t_s[-1] is always a multiple of 3600 → also divisible by 1800
        t_tide_s  = np.arange(0.0, float(t_s[-1]) + 1.0, 1800.0)
        t_tide_h  = t_tide_s / 3600.0 + t_h[0]          # storm-centred time
        wl_fine   = _water_level_evolution(
            t_tide_h, surge_height, self.surge_duration,
            self.surge_top_duration, _m2_tide(t_tide_h, tidal_amp, fase),
        )
        zs_min    = float(np.min(wl_fine))
        tide_data = np.column_stack([t_tide_s, wl_fine, np.full(len(t_tide_s), zs_min)])
        np.savetxt(os.path.join(xb_path, "tide.txt"), tide_data, fmt=["%.0f", "%.6f", "%.6f"])

        # ── waves FILELIST + individual .bnd files ────────────────────────────
        # n_seg == i1: the wave list is trimmed to exactly the same number of
        # 1-hour intervals as the water-level simulation.
        n_seg      = i1
        t_wave_h   = t_full[:n_seg]       # start-of-interval hours (trimmed)

        # wl for FILELIST: wl_piek * cos( 2π t / 110 )
        wl_lst     = rp * np.cos(2 * np.pi * (t_wave_h) / 110.0)

        # Hs(t) = Hs_peak * cos²( π (t - tmax) / Dwaves ),  tmax = 0 (surge peak)
        Dwaves     = 1.25 * self.surge_duration   # = 55 h when surge_duration = 44 h
        cos2       = np.cos(np.pi * t_wave_h / Dwaves) ** 2
        cos2[np.abs(t_wave_h) > Dwaves / 2] = 0.0
        Hm0_series = np.maximum(0.3, hs * cos2)

        g          = 9.81
        speak      = (2 * np.pi / g) * (hs / tp ** 2)
        Tp_series  = np.maximum(1.0, np.sqrt(2 * np.pi * Hm0_series / (g * speak)))
        gammajsp   = 3.3

        last_bnd = ""
        for i in range(n_seg):
            last_bnd = (
                f"Hm0      = {float(Hm0_series[i]):.10f}\n"
                f"fp       = {1.0 / float(Tp_series[i]):.10f}\n"
                f"mainang  = 270.0\n"
                f"gammajsp = {gammajsp}\n"
                f"s        = {spread:.1f}\n"
            )
            with open(os.path.join(xb_path, f"waves{i + 1}.bnd"), "w") as f:
                f.write(last_bnd)
        with open(os.path.join(xb_path, f"waves{n_seg + 1}.bnd"), "w") as f:
            f.write(last_bnd)

        with open(os.path.join(xb_path, "waves.lst"), "w") as f:
            f.write("FILELIST\n")
            for i in range(n_seg):
                f.write(f"3600 {float(wl_lst[i]):.7f} waves{i + 1}.bnd\n")
            f.write(f"1 {float(wl_lst[-1]):.7f} waves{n_seg + 1}.bnd\n")

        # ── Profile files (x.grd, y.grd, bed.dep) ─────────────────────────────
        profile  = getattr(location, "profile", None)
        alfa_deg = 0
        nx_val   = 10

        if profile is not None and getattr(profile, "dune_x", None) is not None:
            x_pts  = np.asarray(profile.dune_x, dtype=float)
            z_pts  = np.asarray(profile.dune_y, dtype=float)
            np.savetxt(os.path.join(xb_path, "x.grd"), x_pts.reshape(1, -1), fmt="%.4f")
            np.savetxt(os.path.join(xb_path, "y.grd"), np.zeros((1, len(x_pts))), fmt="%.4f")
            np.savetxt(os.path.join(xb_path, "bed.dep"), z_pts.reshape(1, -1), fmt="%.4f")
            nx_val   = len(x_pts) - 1
            alfa_deg = int(getattr(profile, "dune_orientation", 0) or 0)

        # ── params.txt from params_params.txt template ────────────────────────
        template_path = os.path.join(xb_path, "params_template.txt")
        params_path   = os.path.join(xb_path, "params.txt")
        with open(template_path, "r") as f:
            params_text = f.read()

        kv = _parse_kv(str(loc_name))
        if kv is None:
            raise ValueError(
                f"Cannot extract kustvaknummer from location name '{loc_name}'. "
                "Expected format: <name>_<Kv>_<Nr>"
            )
        nr = float(hl.get("bnd_nr", raai // 10))
        D50 = _get_d50(kv, nr)
        log.info("D50 = %.6f m  (Kv=%d, Nr=%.1f)", D50, kv, nr)

        params_text = params_text.replace("$alfa$",   str(alfa_deg))
        params_text = params_text.replace("$nx$",     str(nx_val))
        params_text = params_text.replace("$ny$",     "0")
        params_text = params_text.replace("$d50$",    f"{D50:.6f}")
        params_text = params_text.replace("$d90$",    f"{1.5 * D50:.6f}")
        params_text = params_text.replace("$tstop$",  str(int(t_s[-1])))
        params_text = params_text.replace("$tint$",   "3600")
        params_text = params_text.replace("$tstart$", "0")

        with open(params_path, "w") as f:
            f.write(params_text)

        # ── Run XBeach ────────────────────────────────────────────────────────
        exe = os.path.join(xb_path, "xbeach.exe")
        log.info("Running XBeach for raai %d  (tstop = %d s) …", raai, int(t_s[-1]))
        subprocess.call([exe], cwd=xb_path)
        log.info("XBeach simulation finished.")

        # ── Read output ───────────────────────────────────────────────────────
        nc_path = os.path.join(xb_path, "xboutput.nc")
        ds      = netCDF4.Dataset(nc_path)
        zb_end  = np.array(ds["zb"][-1, 0, :])
        resthoogte = float(np.max(zb_end))
        log.info("Resthoogte = %.2f m+NAP", resthoogte)

        result = dict(
            resthoogte = resthoogte,
            xb_path    = xb_path,
            nc_path    = nc_path,
        )

        # ── Save output next to the database ──────────────────────────────────
        out_dir = "."
        t_val   = int(hl["return_period"]) if "return_period" in hl else 0
        t_str   = f"{t_val}" if t_val else ""
        stem    = f"Pydra_XBeach_T{t_str}_r{raai}"
        if save:
            db_path = location.settings.database_path or ""
            out_dir = os.path.dirname(db_path) or "."
            db_stem = os.path.splitext(os.path.basename(db_path))[0]
            stem    = f"Pydra_XBeach_{db_stem}_T{t_str}_r{raai}"
            nc_dest = os.path.join(out_dir, f"{stem}.nc")
            shutil.copy2(nc_path, nc_dest)
            log.info("XBeach output saved → %s", nc_dest)
            # result["nc_saved"] = nc_dest

        # ── Plot ──────────────────────────────────────────────────────────────
        if make_plot:
            self._plot(ds, hl, resthoogte, out_dir, stem, xb_path)

        ds.close()
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _plot(self, ds, hl, resthoogte: float, out_dir: str, stem: str, xb_path: str) -> None:
        import matplotlib.pyplot as plt

        # ── Read boundary condition files ─────────────────────────────────────
        import math
        tide_data  = np.loadtxt(os.path.join(xb_path, "tide.txt"))   # [t_s, wl, zs_min]

        tidal_amp  = float(hl.get("getij",     1.0))
        fase       = float(hl.get("fase",      3.5))
        surge_half = self.surge_duration / 2.0
        n_periods  = math.ceil((fase + surge_half) / T_M2) - 1
        n_start    = int(np.floor(fase - n_periods * T_M2))

        t_tide_h   = tide_data[:, 0] / 3600.0 + n_start   # storm-centred [h]
        wl         = tide_data[:, 1]

        # Decompose into tide and trapezoidal surge (t=0 = surge peak)
        tide_line  = tidal_amp * np.cos(2.0 * np.pi / T_M2 * (t_tide_h - fase))
        surge_line = wl - tide_line

        # Read FILELIST waves.lst + individual .bnd files (skip trailing 1-second entry)
        with open(os.path.join(xb_path, "waves.lst")) as wf:
            wave_rows = [l.split() for l in wf if not l.startswith("FILELIST") and len(l.split()) >= 3]
        Hm0_series, Tp_series, dt_s_list = [], [], []
        for row in wave_rows:
            dur_s = int(row[0])
            if dur_s < 3600:
                continue
            bnd_params: dict = {}
            with open(os.path.join(xb_path, row[2])) as bf:
                for bline in bf:
                    if "=" in bline:
                        k, v = bline.split("=", 1)
                        bnd_params[k.strip()] = float(v.strip())
            Hm0_series.append(bnd_params.get("Hm0", 0.0))
            fp = bnd_params.get("fp", 1.0)
            Tp_series.append(1.0 / fp if fp > 0 else 1.0)
            dt_s_list.append(dur_s)
        Hm0_series = np.array(Hm0_series)
        Tp_series  = np.array(Tp_series)
        t_wave_h   = np.cumsum([0.0] + [d / 3600.0 for d in dt_s_list[:-1]]) + t_tide_h[0]

        # ── Read profile ──────────────────────────────────────────────────────
        zb_ini = np.array(ds["zb"][0,  0, :])
        zb_end = np.array(ds["zb"][-1, 0, :])
        x      = np.loadtxt(os.path.join(xb_path, "x.grd")).flatten()[:len(zb_ini)]

        # ── Layout: profile (top) + boundary conditions (bottom, 3 panels) ───
        fig = plt.figure(figsize=(14, 9))
        gs  = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.42, wspace=0.38)

        ax_prof = fig.add_subplot(gs[0, :])   # full-width profile panel
        ax_wl   = fig.add_subplot(gs[1, 0])
        ax_hm0  = fig.add_subplot(gs[1, 1])
        ax_tp   = fig.add_subplot(gs[1, 2])

        # ── Profile panel ─────────────────────────────────────────────────────
        ax_prof.plot(x, zb_ini, color="#2B4590", lw=1.5, label="Initieel profiel")
        ax_prof.fill_between(x, zb_ini, zb_end, where=(zb_end < zb_ini).tolist(),
                             color="#E63946", alpha=0.25, label="Erosie")
        ax_prof.plot(x, zb_end, color="#E63946", lw=1.5,
                     label=f"Eindprofiel  (resthoogte = {resthoogte:.2f} m+NAP)")
        ax_prof.axhline(hl["rp"], color="#F4A261", lw=1.2, ls="--",
                        label=f"Rekenpeil  {hl['rp']:.2f} m+NAP")
        ax_prof.set_xlabel("Dwarsraai coördinaat [m]")
        ax_prof.set_ylabel("Hoogte [m+NAP]")
        ax_prof.set_title(
            f"XBeach duinerosie  –  JARKUS raai {hl.get('jarkus_raai', '–')}  "
            f"(Hm0 = {hl['hs']:.2f} m,  Tp = {hl['tp']:.1f} s,  "
            f"Rekenpeil = {hl['rp']:.2f} m+NAP)"
        )
        ax_prof.legend(fontsize=9)
        ax_prof.grid(True, alpha=0.35)

        # ── Water level ───────────────────────────────────────────────────────
        ax_wl.plot(t_tide_h, tide_line,  color="#43AA8B", lw=1.2, ls="--", label="Getij")
        ax_wl.plot(t_tide_h, surge_line, color="#E63946", lw=1.2, ls="--", label="Opzet")
        ax_wl.plot(t_tide_h, wl, color="#2B4590", lw=1.6, label="Waterstand")
        ax_wl.axhline(hl["rp"], color="#F4A261", lw=1.0, ls="--",
                      label=f"Rekenpeil {hl['rp']:.2f} m")
        ax_wl.axvline(0.0, color="#6C757D", lw=0.8, ls=":", alpha=0.7)
        ax_wl.set_xlabel("Tijd t.o.v. stormopzet piek [h]")
        ax_wl.set_ylabel("Waterstand [m+NAP]")
        ax_wl.set_title("Waterstand")
        ax_wl.legend(fontsize=8)
        ax_wl.grid(True, alpha=0.3)

        # ── Wave height ───────────────────────────────────────────────────────
        ax_hm0.plot(t_wave_h, Hm0_series, color="#E63946", lw=1.4)
        ax_hm0.scatter(t_wave_h, Hm0_series, color="#E63946", s=20, zorder=3)
        ax_hm0.axhline(hl["hs"], color="#F4A261", lw=1.0, ls="--",
                       label=f"Rekenwaarde {hl['hs']:.2f} m")
        ax_hm0.axvline(0.0, color="#6C757D", lw=0.8, ls=":", alpha=0.7)
        ax_hm0.set_xlabel("Tijd t.o.v. stormopzet piek [h]")
        ax_hm0.set_ylabel("Hm0 [m]")
        ax_hm0.set_title("Significante golfhoogte")
        ax_hm0.legend(fontsize=8)
        ax_hm0.grid(True, alpha=0.3)

        # ── Wave period ───────────────────────────────────────────────────────
        ax_tp.plot(t_wave_h, Tp_series, color="#43AA8B", lw=1.4)
        ax_tp.scatter(t_wave_h, Tp_series, color="#43AA8B", s=20, zorder=3)
        ax_tp.axhline(hl["tp"], color="#F4A261", lw=1.0, ls="--",
                      label=f"Rekenwaarde {hl['tp']:.1f} s")
        ax_tp.axvline(0.0, color="#6C757D", lw=0.8, ls=":", alpha=0.7)
        ax_tp.set_xlabel("Tijd t.o.v. stormopzet piek [h]")
        ax_tp.set_ylabel("Tp [s]")
        ax_tp.set_title("Piekperiode")
        ax_tp.legend(fontsize=8)
        ax_tp.grid(True, alpha=0.3)

        pdf_path = os.path.join(out_dir, f"{stem}.pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("XBeach profile plot saved → %s", pdf_path)
