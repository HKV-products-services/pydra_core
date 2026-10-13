from scipy.stats import norm

from .calculation import Calculation
from ..hrdatabase.hrdatabase import HRDatabase
from ..location.location import Location

import logging
import os
import re
from typing import Optional, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── JARKUS raai parser ────────────────────────────────────────────────────────

def _parse_raai(name: str) -> Optional[int]:
    """
    Extract the JARKUS raai number (in metres) from a location name.

    Supported formats
    -----------------
    1. HRD database names  –  contain a ``jr<digits>`` token; digits in metres:
        ``001-01_0001_SCHR_02_jr001000``  →  1000
        ``vk2110_0002_VRNE_11_jr016000``  →  16000

    2. Human-readable names  –  ``<location>_<kustvak>_<raai_dam>``; trailing
       number is in decametres (dam) and is multiplied by 10:
        ``Schiermonnikoog_2_100``  →  1000
        ``SCHR_2_140``            →  1400

    Returns ``None`` if no raai number can be parsed.
    """
    name = name.strip()
    # Strategy 1: explicit jr<digits> token → metres
    m = re.search(r"jr(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Strategy 2: <word>_<kustvak>_<raai_dam>  → decametres × 10
    m = re.match(r"^[A-Za-z][A-Za-z0-9\-]*_\d+_(\d+)$", name)
    if m:
        return int(m.group(1)) * 10
    return None


# ── pydra object helpers ──────────────────────────────────────────────────────

def _find_col(names: list, target: str) -> int:
    """
    Return the 0-based index of *target* in *names* (case-insensitive
    substring match).  Tolerates minor wording differences across pydra
    versions.
    """
    target_l = target.lower()
    for i, n in enumerate(names):
        if target_l in n.lower():
            return i
    raise KeyError(
        f"Variable '{target}' not found in loading model variables: {names}"
    )


def _location_meta(location) -> tuple:
    """
    Extract (jarkus_raai, name, x, y, delta_h) from a pydra location or
    settings object.  Tries several attribute naming conventions.
    """
    name = (
        getattr(location, "name", None)
        or getattr(location, "Name", None)
        or getattr(location, "location_name", None)
        or getattr(location, "location", None)   # Settings.location
        or ""
    )
    x = float(
        getattr(location, "x_coordinate", None)
        or getattr(location, "XCoordinate", None)
        or getattr(location, "x", 0.0)
    )
    y = float(
        getattr(location, "y_coordinate", None)
        or getattr(location, "YCoordinate", None)
        or getattr(location, "y", 0.0)
    )
    def _f(val) -> float:
        return float(val) if val is not None else 0.0

    # Both values are additive: database correction + scenario sea-level rise
    dh = (
        _f(getattr(location, "water_level_correction", None))
        + _f(getattr(location, "sea_level_rise", None))
    )

    raai = _parse_raai(str(name))
    if raai is None:
        # Try nested settings sub-object
        nested = getattr(location, "settings", None)
        if nested is not None:
            loc_name = (
                getattr(nested, "location", None)
                or getattr(nested, "name", None)
                or ""
            )
            raai = _parse_raai(str(loc_name))
            if not name:
                name = str(loc_name)

    if raai is None:
        raise ValueError(
            f"Cannot extract JARKUS raai from location name '{name}'. "
            "Check _parse_raai() or ensure the location name contains a "
            "jr<digits> token or a <word>_<kustvak>_<raai_dam> pattern."
        )

    return raai, str(name), x, y, dh


def _extract_grid_from_loading(loading_model) -> tuple:
    """
    Build (h_fn, Hs_fn, Tp_fn, tidal_fn, u_wl_vals, u_stat_vals) as 2-D
    RegularGridInterpolators over (u_wl, u_stat) at u_hs = 0, u_tp = 0,
    directly from a pydra LoadingModel object.

    Expected loading model attributes
    ----------------------------------
    loading_model.input_variables   – list of str, one label per input axis
    loading_model.result_variables  – list of str, one label per result column
    loading_model.h                 – ndarray, shape (n_wl, n_hs, n_tp, n_stat)
                                      (same for Hs / Tp / tidal arrays)

    Axis order (matches HRD HRDInputColumnId order seen in screenshots):
    axis 0  Sea water level (u_wl)
    axis 1  Wave height     (u_hs)   → sliced at index of 0.0
    axis 2  Wave period     (u_tp)   → sliced at index of 0.0
    axis 3  Statistical uncertainty (u_stat)
    """
    in_vars  = list(loading_model.input_variables)
    res_vars = list(loading_model.result_variables)

    # Locate variable columns by name
    col_wl   = _find_col(in_vars, "sea water level")
    col_hs   = _find_col(in_vars, "wave height")
    col_tp   = _find_col(in_vars, "wave period")
    col_stat = _find_col(in_vars, "uncertainty water level")

    col_h      = _find_col(res_vars, "h")
    col_Hs     = _find_col(res_vars, "significant wave height")
    col_Tp     = _find_col(res_vars, "peak wave period")
    col_tidal  = _find_col(res_vars, "mean tidal amplitude")
    try:
        col_fase   = _find_col(res_vars, "phase difference")
    except KeyError:
        col_fase = None
    try:
        col_spread = _find_col(res_vars, "directional spread")
    except KeyError:
        col_spread = None

    # ── retrieve axis arrays ──────────────────────────────────────────────────
    def _get_axis(col_idx: int, attr_candidates: list) -> np.ndarray:
        """Try the direct column name first, then named candidates."""
        direct_name = in_vars[col_idx]
        val = getattr(loading_model, direct_name, None)
        if val is not None:
            return np.asarray(val).ravel()
        for attr in attr_candidates:
            val = getattr(loading_model, attr, None)
            if val is not None:
                return np.asarray(val).ravel()
        raise AttributeError(
            f"Cannot find axis data for input column {col_idx} ('{direct_name}'). "
            f"Tried attributes: {[direct_name] + attr_candidates}. "
            "Add the correct pydra attribute name to attr_candidates."
        )

    u_wl_vals   = _get_axis(col_wl,   ["sea_water_level_u",          "u_wl",   "uwl"])
    u_hs_axis   = _get_axis(col_hs,   ["wave_height_u",               "u_hs",   "uhs"])
    u_tp_axis   = _get_axis(col_tp,   ["wave_period_u",               "u_tp",   "utp"])
    u_stat_vals = _get_axis(col_stat, ["uncertainty_water_level_u",   "u_stat", "ustat"])

    # Find indices of 0.0 on the u_hs and u_tp axes
    idx_hs0 = int(np.argmin(np.abs(u_hs_axis - 0.0)))
    idx_tp0 = int(np.argmin(np.abs(u_tp_axis - 0.0)))
    if abs(u_hs_axis[idx_hs0]) > 1e-9 or abs(u_tp_axis[idx_tp0]) > 1e-9:
        log.warning(
            "Closest u_hs/u_tp to 0 are %.4f / %.4f; "
            "slice may not be exactly at (u_hs=0, u_tp=0).",
            u_hs_axis[idx_hs0], u_tp_axis[idx_tp0],
        )

    # ── retrieve 4-D result arrays and slice to 2-D ───────────────────────────
    def _get_result_array(col_idx: int, attr_candidates: list) -> np.ndarray:
        direct_name = res_vars[col_idx]
        val = getattr(loading_model, direct_name, None)
        if val is not None:
            return np.asarray(val)
        for attr in attr_candidates:
            val = getattr(loading_model, attr, None)
            if val is not None:
                return np.asarray(val)
        # Generic fallback: loading_model.result_data columns reshaped to 4-D
        rd = getattr(loading_model, "result_data", None)
        if rd is not None:
            shape4d = (
                len(u_wl_vals), len(u_hs_axis),
                len(u_tp_axis), len(u_stat_vals),
            )
            return np.asarray(rd)[:, col_idx].reshape(shape4d)
        raise AttributeError(
            f"Cannot find result data for column {col_idx} ('{direct_name}'). "
            f"Tried: {[direct_name] + attr_candidates}. Adapt _get_result_array()."
        )

    def _slice2d(arr4d: np.ndarray) -> np.ndarray:
        """(u_wl, u_hs, u_tp, u_stat)  →  (u_wl, u_stat) at u_hs=0, u_tp=0."""
        return arr4d[:, idx_hs0, idx_tp0, :]

    h_2d     = _slice2d(_get_result_array(col_h,     ["h"]))
    hs_2d    = _slice2d(_get_result_array(col_Hs,    ["significant_wave_height", "hs", "Hs"]))
    tp_2d    = _slice2d(_get_result_array(col_Tp,    ["peak_wave_period",        "tp", "Tp"]))
    tidal_2d = _slice2d(_get_result_array(col_tidal, ["mean_tidal_amplitude", "tidal_amplitude", "tidal"]))
    fase_2d   = _slice2d(_get_result_array(col_fase,   ["phase_difference", "fase"])) if col_fase   is not None else np.full_like(h_2d, 3.5)
    spread_2d = _slice2d(_get_result_array(col_spread, ["directional_spread", "spread"])) if col_spread is not None else np.full_like(h_2d, 20.0)

    def _rgi(data):
        return RegularGridInterpolator(ax, data, method="linear", bounds_error=False, fill_value=None)  # type: ignore[arg-type]

    ax = (list(u_wl_vals), list(u_stat_vals))
    return (
        _rgi(h_2d), _rgi(hs_2d), _rgi(tp_2d), _rgi(tidal_2d),
        _rgi(fase_2d), _rgi(spread_2d),
        list(u_wl_vals), list(u_stat_vals),
    )


# ── FORM design-point ─────────────────────────────────────────────────────────

def _blend(fn_i, fn_i1, frac: float, uwl: float, ust: float) -> float:
    pt = [[uwl, ust]]
    vi  = fn_i(pt)[0]
    vi1 = fn_i1(pt)[0]
    if vi is None or np.isnan(vi) or vi1 is None or np.isnan(vi1):
        return float("nan")
    return (1.0 - frac) * float(vi) + frac * float(vi1)


def _design_point(
    h_fn_i,
    h_fn_i1,
    frac: float,
    beta: float,
    u_stat_max: float,
    n_coarse: int = 10_000,
) -> tuple:
    """
    Find θ* = argmax h(β·cosθ, β·sinθ)  for θ ∈ [0, θ_max]
    where θ_max = arcsin(min(u_stat_max, β) / β).

    Returns (u_wl_star, u_stat_star, theta_scan, h_scan).
    """
    theta_max = np.arcsin(min(u_stat_max, beta) / beta) if u_stat_max > 0.0 else 0.0
    thetas = np.linspace(0.0, theta_max, n_coarse)
    h_scan = np.array([
        _blend(h_fn_i, h_fn_i1, frac,
               beta * np.cos(t), beta * np.sin(t))
        for t in thetas
    ])

    valid = np.isfinite(h_scan)
    if not valid.any():
        return beta, 0.0, thetas, h_scan

    best_idx   = int(np.argmax(np.where(valid, h_scan, -np.inf)))
    best_theta = thetas[best_idx]

    if theta_max > 0.0:
        dt = theta_max / n_coarse
        res = minimize_scalar(
            lambda t: -_blend(h_fn_i, h_fn_i1, frac,
                              beta * np.cos(t), beta * np.sin(t)),
            bounds=(max(0.0, best_theta - dt), min(theta_max, best_theta + dt)),
            method="bounded",
        )
        best_theta = float(res.x)  # type: ignore[union-attr]

    return (
        beta * np.cos(best_theta),
        beta * np.sin(best_theta),
        thetas,
        h_scan,
    )


# ── .bnd writer ───────────────────────────────────────────────────────────────

def _write_bnd(rows: list, output_path: str, return_period: float) -> None:
    prob = 1.0 / return_period
    lines = [
        "Kv\tNr\tRp\tHs\tTp\tGetij\tdt\t_BOI2023_Waarde",
        "* KustvakID\tRaaiID\tRekenpeil\tRekenwaarde significante golfhoogte"
        "\tRekenwaarde piekperiode\tGem. getij amplitude"
        "\tFaseverschuilling getij\tP_afslag (doelkans)",
        "* [-]\t[dam]\t[m+NAP]\t[m]\t[s]\t[m]\t[hr]\t[1/jaar]",
    ]
    for row in rows:
        lines.append(
            f"2\t{row['bnd_nr']}\t{row['rp']:.2f}\t{row['hs']:.2f}"
            f"\t{row['tp']:.2f}\t{row['getij']:.2f}\t{row['fase']:.2f}\t{prob}"
        )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ── FORM stability plot ───────────────────────────────────────────────────────

def _make_form_plot(
    results: list,
    beta: float,
    u_stat_max: float,
    return_period: float,
    output_pdf: str,
) -> None:
    """Save a multi-page PDF with one FORM stability panel per JARKUS raai."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MultipleLocator

    C_CIRCLE = "#2B4590"
    C_ARC    = "#E63946"
    C_DP     = "#F4A261"
    C_GRID   = "#6C757D"
    C_BAND   = "#F4A261"
    CMAP     = "YlOrRd"

    with PdfPages(output_pdf) as pdf:
        for r in results:
            u_stat_max  = r.get("_u_stat_max_eff", u_stat_max)
            raai        = r["jarkus_raai"]
            u_wl_star   = r["u_wl_star"]
            u_stat_star = r["u_stat_star"]
            rp          = r["rp"]
            thetas      = r["_thetas"]
            h_arc       = r["_h_arc"]
            u_wl_grid   = r["_u_wl_grid"]
            u_stat_grid = r["_u_stat_grid"]
            h_fn        = r["_h_fn"]

            theta_star     = np.arctan2(u_stat_star, u_wl_star)
            theta_star_deg = np.degrees(theta_star)
            theta_max_deg  = np.degrees(thetas[-1]) if len(thetas) > 1 else 0.0

            valid   = np.isfinite(h_arc)
            h_valid = h_arc[valid]
            t_valid = thetas[valid]

            fig, (ax_map, ax_arc) = plt.subplots(
                1, 2, figsize=(14, 6.5),
                gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.35}
            )
            fig.suptitle(
                f"FORM Design-Point Stability  –  JARKUS raai {raai}  "
                f"(T = {return_period:.0f} yr,  β = {beta:.3f},  "
                f"u_stat_max = {u_stat_max:.2f})",
                fontsize=11, fontweight="bold", y=0.995
            )

            # Left: 2-D h contour surface
            uw_dense = np.linspace(0, max(u_wl_grid),  300)
            us_dense = np.linspace(0, max(u_stat_grid), 200)
            UW, US   = np.meshgrid(uw_dense, us_dense, indexing="ij")
            H_field  = h_fn(np.stack([UW.ravel(), US.ravel()], axis=1)).reshape(UW.shape)
            h_min    = float(np.nanpercentile(H_field, 5))
            h_max    = float(np.nanmax(H_field))
            levels   = np.linspace(h_min, h_max, 18)
            cf = ax_map.contourf(UW, US, H_field, levels=levels, cmap=CMAP, alpha=0.82)
            ax_map.contour(UW, US, H_field, levels=levels,
                           colors="white", linewidths=0.3, alpha=0.35)
            cb = fig.colorbar(cf, ax=ax_map, shrink=0.82, pad=0.03)
            cb.set_label("h  [m+NAP]", fontsize=9)
            cb.ax.tick_params(labelsize=8)

            t_full = np.linspace(0, np.pi / 2, 500)
            ax_map.plot(beta * np.cos(t_full), beta * np.sin(t_full),
                        color=C_CIRCLE, lw=1.6, ls="--",
                        label=f"Reliability circle (β={beta:.3f})", zorder=4)

            if len(thetas) > 1:
                ax_map.plot(beta * np.cos(thetas), beta * np.sin(thetas),
                            color=C_ARC, lw=2.8,
                            label=f"Search arc (u_stat ≤ {u_stat_max:.2f})", zorder=5)
            else:
                ax_map.plot([beta], [0], "o", color=C_ARC, ms=8,
                            label="Search point (u_stat = 0)", zorder=5)

            ax_map.axvline(max(u_wl_grid),  color=C_GRID, lw=1.0, ls=":",
                           alpha=0.8, label=f"DB u_wl max ({max(u_wl_grid):.1f})")
            ax_map.axhline(max(u_stat_grid), color=C_GRID, lw=1.0, ls="-.",
                           alpha=0.8, label=f"DB u_stat max ({max(u_stat_grid):.1f})")
            if u_stat_max < max(u_stat_grid):
                ax_map.axhline(u_stat_max, color=C_ARC, lw=1.0, ls="-.",
                               alpha=0.6, label=f"u_stat_max cap ({u_stat_max:.2f})")

            ax_map.plot(u_wl_star, u_stat_star, "*",
                        color=C_DP, ms=16, mec="black", mew=0.7,
                        label=f"Design point  h={rp:.2f} m", zorder=6)
            ax_map.annotate(
                "", xy=(u_wl_star, u_stat_star), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_DP,
                                lw=1.5, mutation_scale=14),
                zorder=5
            )

            ax_map.set_xlabel("u_wl  (water level u-value)", fontsize=10)
            ax_map.set_ylabel("u_stat  (statistical uncertainty u-value)", fontsize=10)
            ax_map.set_xlim(0, min(beta * 1.12, max(u_wl_grid) + 0.3))
            ax_map.set_ylim(0, max(u_stat_grid) + 0.3)
            ax_map.xaxis.set_minor_locator(MultipleLocator(0.2))
            ax_map.yaxis.set_minor_locator(MultipleLocator(0.25))
            ax_map.tick_params(which="both", direction="in", labelsize=8)
            ax_map.set_aspect("equal", adjustable="box")
            ax_map.legend(loc="upper right", fontsize=7.5, framealpha=0.9,
                          edgecolor="#CCCCCC")

            alpha_wl   = np.cos(theta_star)
            alpha_stat = np.sin(theta_star)
            ax_map.text(
                0.02, 0.02,
                f"α_wl   = {alpha_wl:.3f}\n"
                f"α_stat = {alpha_stat:.3f}\n"
                f"u_wl*  = {u_wl_star:.3f}\n"
                f"u_st*  = {u_stat_star:.3f}",
                transform=ax_map.transAxes, fontsize=8.5,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", alpha=0.9)
            )

            # Right: h along the arc
            if valid.any() and len(t_valid) > 1:
                ax_arc.plot(np.degrees(t_valid), h_valid,
                            color=C_CIRCLE, lw=2.0, zorder=3, label="h on arc")
                band_mask = np.abs(thetas - theta_star) <= np.radians(1.0)
                if band_mask.any() and np.isfinite(h_arc[band_mask]).any():
                    h_band        = h_arc[band_mask]
                    t_band        = np.degrees(thetas[band_mask])
                    h_band_finite = h_band[np.isfinite(h_band)]
                    if len(h_band_finite) > 0:
                        ax_arc.axvspan(t_band[0], t_band[-1], alpha=0.18,
                                       color=C_BAND, label="±1° sensitivity band")
                        dh_band = h_band_finite.max() - h_band_finite.min()
                        ax_arc.annotate(
                            f"Δh = {dh_band*100:.1f} cm\nin ±1°",
                            xy=(theta_star_deg, rp),
                            xytext=(
                                theta_star_deg + max(theta_max_deg * 0.12, 1.5),
                                rp - (h_valid.max() - h_valid.min()) * 0.08,
                            ),
                            fontsize=8, color=C_DP,
                            arrowprops=dict(arrowstyle="->", color=C_DP,
                                            lw=1.0, connectionstyle="arc3,rad=0.2"),
                        )
            elif u_stat_max == 0.0 and len(thetas) == 1:
                ax_arc.plot([0], [h_arc[0]], "o", color=C_ARC, ms=10, zorder=4)

            ax_arc.axvline(theta_star_deg, color=C_DP, lw=1.8, ls="--", zorder=4,
                           label=f"Design point  θ* = {theta_star_deg:.1f}°")
            if u_stat_max > 0.0 and u_stat_max < max(u_stat_grid):
                ax_arc.axvline(theta_max_deg, color=C_ARC, lw=1.2, ls=":",
                               alpha=0.7, label=f"u_stat_max cap  θ = {theta_max_deg:.1f}°")

            ax_arc.axhline(rp, color=C_DP, lw=1.0, ls=":", alpha=0.6)
            ax_arc.text(
                theta_max_deg * 0.02 if theta_max_deg > 0 else 0.2,
                rp + (h_valid.max() - h_valid.min()) * 0.03
                    if valid.any() and len(h_valid) > 0 else 0.05,
                f"Rp = {rp:.2f} m",
                fontsize=8.5, color=C_DP, va="bottom",
            )

            ax_arc.set_xlabel("θ = arctan(u_stat / u_wl)  [degrees]", fontsize=10)
            ax_arc.set_ylabel("h  [m+NAP]", fontsize=10)
            ax_arc.set_title("h along the reliability arc", fontsize=10, pad=6)
            if valid.any() and len(h_valid) > 1:
                margin = (h_valid.max() - h_valid.min()) * 0.12
                ax_arc.set_ylim(h_valid.min() - margin, h_valid.max() + margin * 2)
            ax_arc.tick_params(labelsize=8, direction="in")
            ax_arc.xaxis.set_minor_locator(MultipleLocator(1))
            ax_arc.legend(fontsize=8, loc="lower left", framealpha=0.9,
                          edgecolor="#CCCCCC")

            ax_arc.text(
                0.98, 0.97,
                f"Rp  = {rp:.2f} m+NAP\n"
                f"Hs  = {r['hs']:.2f} m\n"
                f"Tp  = {r['tp']:.2f} s\n"
                f"Getij = {r['getij']:.4f} m",
                transform=ax_arc.transAxes, fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.45", fc="#F8F9FA",
                          ec="#AAAAAA", alpha=0.95),
            )

            fig.subplots_adjust(top=0.93, left=0.07, right=0.97, bottom=0.10)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    log.info("FORM stability plot saved → %s  (%d pages)", output_pdf, len(results))


# ── Main class ────────────────────────────────────────────────────────────────

class HydraulicLoadsDunes(Calculation):
    """
    Calculate the hydraulic loads for a dunes location
    """

    def __init__(self, return_period: float, model_uncertainty: bool = True):
        """
        Parameters
        ----------
        return_period: float
            The return period for which the hydraulic loads will be calculated.
        model_uncertainty: bool
            Enable or disable the use of model uncertainties. Default is True.
        """
        super().__init__()
        self.return_period = return_period
        self.model_uncertainty = model_uncertainty

    def calculate(
        self,
        input: Union[Location, HRDatabase],
        save: bool = False,
        n_coarse: int = 10_000,
        make_plot: bool = False,
        plot_output: Optional[str] = None,
    ):
        """
        Calculate hydraulic loads for a single Location or all locations in an
        HRDatabase.

        Parameters
        ----------
        input : Union[Location, HRDatabase]
            A single Location or an HRDatabase (calculates every location).
        save : bool
            Write a combined .bnd file next to the database.
        n_coarse : int
            Angular scan resolution (default 10 000).
        make_plot : bool
            Produce a FORM stability PDF.
        plot_output : str, optional
            Explicit PDF path; auto-generated when None.
        """
        if isinstance(input, Location):
            return self.calculate_location(
                input, save=save, n_coarse=n_coarse,
                make_plot=make_plot, plot_output=plot_output,
            )

        if isinstance(input, HRDatabase):
            all_results = []
            for loc_name in input:
                loc = input.create_location(loc_name)
                results = self.calculate_location(loc, save=False, n_coarse=n_coarse, make_plot=False)
                all_results.extend(results)

            output_path: Optional[str] = None
            if save:
                db_path = input.database_path or ""
                db_dir  = os.path.dirname(db_path) or "."
                db_stem = os.path.splitext(os.path.basename(db_path))[0]
                output_path = os.path.join(
                    db_dir, f"Pydra_HR_{db_stem}_T{int(self.return_period)}.bnd"
                )
                _write_bnd(all_results, output_path, self.return_period)
                log.info("Written %d rows → %s", len(all_results), output_path)

            if make_plot:
                beta = float(norm.ppf(1.0 - 1.0 / self.return_period))
                if not plot_output:
                    base    = output_path or f"Pydra_HR_T{int(self.return_period)}"
                    stem    = os.path.splitext(os.path.basename(base))[0]
                    out_dir = os.path.dirname(base) or "."
                    plot_output = os.path.join(out_dir, f"{stem}.pdf")
                u_stat_rep = max(
                    (r.get("_u_stat_max_eff", 0.0) for r in all_results), default=0.0
                )
                _make_form_plot(all_results, beta, u_stat_rep, self.return_period, plot_output)

            return all_results

        raise NotImplementedError("[ERROR] Input type not implemented")

    def calculate_location(
        self,
        location: Location,
        save: bool = False,
        n_coarse: int = 10_000,
        make_plot: bool = False,
        plot_output: Optional[str] = None,
    ):
        """
        Calculate the hydraulic loads for dunes of a specific location using
        the FORM method.

        Parameters
        ----------
        location : Location
            The Location object.
        save : bool
            Write output files next to the .sqlite database when True.
            The .bnd filename is derived as
            ``Pydra_HR_{db_stem}_T{return_period}.bnd``.
        n_coarse : int
            Angular scan resolution (default 10 000).
        make_plot : bool
            Produce a FORM stability PDF if True.
        plot_output : str, optional
            Explicit PDF path; auto-generated next to the .bnd file if None.

        Returns
        -------
        list
            One result dict per location with keys:
            jarkus_raai, bnd_nr, u_wl_star, u_stat_star,
            rp, hs, tp, getij, fase, spread
            (plus _-prefixed internal arrays used by the plot).
        """
        model   = location.get_model()
        loading = model.get_loading()

        # For dunes there is a single LoadingModel (no wind directions / closing situations)
        loading_model = next(iter(loading.model.values()))

        if self.model_uncertainty:
            col_stat   = _find_col(list(loading_model.input_variables), "uncertainty water level")
            u_stat_col = loading_model.input_variables[col_stat]
            u_stat_max = float(np.max(getattr(loading_model, u_stat_col)))
        else:
            u_stat_max = 0.0

        beta           = float(norm.ppf(1.0 - 1.0 / self.return_period))
        u_stat_max_eff = float(np.clip(u_stat_max, 0.0, beta))
        if u_stat_max_eff != u_stat_max:
            log.warning("u_stat_max=%.3f clamped to β=%.3f.", u_stat_max, beta)
        log.info("T = %g yr  →  β = %.6f  |  u_stat_max = %.2f",
                 self.return_period, beta, u_stat_max)

        raai, loc_name, _x, _y, dh = _location_meta(location.settings)
        log.info("Location: %s  (raai %d,  ΔH = %.4f m)", loc_name, raai, dh)
        h_fn, hs_fn, tp_fn, ti_fn, fase_fn, spread_fn, u_wl_vals, u_stat_vals = (
            _extract_grid_from_loading(loading_model)
        )

        u_wl_star, u_stat_star, thetas, h_arc = _design_point(
            h_fn, h_fn, 0.0, beta, u_stat_max_eff, n_coarse
        )

        pt = [[u_wl_star, u_stat_star]]
        rp        = float(h_fn(pt)[0]) + dh
        hs        = float(hs_fn(pt)[0])
        tp        = float(tp_fn(pt)[0])
        getij     = float(ti_fn(pt)[0])
        fase      = float(fase_fn(pt)[0])
        spread    = float(spread_fn(pt)[0])

        def _h_blended(pts, _f=h_fn):
            return _f(pts)

        results = [dict(
            jarkus_raai     = raai,
            bnd_nr          = raai // 10,
            return_period   = self.return_period,
            u_wl_star       = round(u_wl_star,4),
            u_stat_star     = round(u_stat_star,4),
            rp              = round(rp,3),
            hs              = round(hs,3),
            tp              = round(tp,3),
            getij           = round(getij,4),
            fase            = round(fase,2),
            spread          = round(spread,2),
            _thetas         = thetas,
            _h_arc          = h_arc,
            _u_wl_grid      = u_wl_vals,
            _u_stat_grid    = u_stat_vals,
            _h_fn           = _h_blended,
            _u_stat_max_eff = u_stat_max_eff,
        )]

        # Derive output path from the database location when save=True
        output_path: Optional[str] = None
        if save:
            db_path = location.settings.database_path or ""
            db_dir  = os.path.dirname(db_path) or "."
            db_stem = os.path.splitext(os.path.basename(db_path))[0]
            output_path = os.path.join(
                db_dir, f"Pydra_HR_{db_stem}_T{int(self.return_period)}.bnd"
            )
            _write_bnd(results, output_path, self.return_period)
            log.info("Written 1 row → %s", output_path)

        if make_plot:
            if not plot_output:
                base = output_path or f"Pydra_HR_{db_stem}_T{int(self.return_period)}_r{raai}"
                stem    = os.path.splitext(os.path.basename(base))[0]
                out_dir = os.path.dirname(base) or "."
                plot_output = os.path.join(out_dir, f"{stem}.pdf")
            _make_form_plot(results, beta, u_stat_max_eff, self.return_period, plot_output)

        return results
