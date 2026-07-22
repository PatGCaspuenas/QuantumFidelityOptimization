#!/usr/bin/env python3
"""
Generates figure_fcl_amp_n100_gpzoom.pdf

3x2 figure for N=100 BO traces (scale=0.5):
  Row 1 — 1D GP posterior slices along fsb, amp at iterations 10, 30, 50.
  Row 2 — 2D scatter of acquisition samples (x_acq) across all seeds and iters.
  Row 3 — 2D scatter of recommended best point (x_rec) across all seeds and iters.

Features:
  - Columns share x-axes (fcl, fsb, amp).
  - Y-axes for rows 2 and 3 cycle through the combinations to show all 2D pairings.
  - True Q_det landscape is overlaid on the 2D plots using black contour lines with labels.
  - Transparent scatter points reveal raw distributions.
  - Fuzzy GP variance shown for ALL iterations.
  - Custom ticks and dotted gridlines applied.
"""

import csv
import glob
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm as sp_norm
import seaborn as sns
from pathlib import Path

os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/Library/TeX/texbin:" + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# APS formatting 
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        9,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "lines.linewidth":  1.5,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
})
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}\usepackage{xcolor}",
})

# Extract 3 distinct colors from the magma palette for the GP iterations
magma_cols = sns.color_palette("magma", n_colors=6).as_hex()
COLORS_GP = [magma_cols[1], magma_cols[3], magma_cols[4]]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT / "figures"

GP_STEM         = "scale05_selectedN_gp_slices_iters10_30_50"
GP_SLICES_FILE  = DATA_DIR / "gp_slices" / f"{GP_STEM}.csv"
TRUE_SCORE_FILE = DATA_DIR / "score_cache" / "scale05_true_2ms_slices_freqspan10_grid201.csv"

def trace_dir_path(n_label):
    return DATA_DIR / f"traces_freqspan10_bound050_N{n_label}_nostop100_stream_100seeds"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_FOCUS       = os.environ.get("PLOT_N_FOCUS", "100")
ITER_GPS      = [10, 30, 50]

FREQ_SPAN_KHZ  = 10.0   
AMP_SPAN_RATIO = 0.2    
BOUND_SCALE    = 0.5

# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------
AXES_1D  = ["fcl", "fsb", "amp"]

# Matches X-axes of Row 1 while cycling the remaining variable on Y
# Col 0: X=fcl (w_cl), Y=fsb (\delta)
# Col 1: X=fsb (\delta), Y=amp (\Omega)
# Col 2: X=amp (\Omega), Y=fcl (w_cl)
PAIRS_2D = [("fcl", "fsb"), ("fsb", "amp"), ("fcl", "amp")]

AXIS_LABELS = {
    "fcl": r"$\Delta \omega_\mathrm{cl}\ (2\pi \cdot \mathrm{kHz})$",
    "fsb": r"$\Delta \delta\ (2\pi \cdot \mathrm{kHz})$",
    "amp": r"$\Omega / \Omega^*$",
}
AXIS_LIMS = {
    "fcl": (-5.0, 5.0),
    "fsb": (-5.0, 5.0),
    "amp": (1.0 - BOUND_SCALE * AMP_SPAN_RATIO, 1.0 + BOUND_SCALE * AMP_SPAN_RATIO),
}
# Updated exactly to the requested tick marks
AXIS_TICKS = {
    "fcl": [-5.0, -2.5, 0.0, 2.5, 5.0],
    "fsb": [-5.0, -2.5, 0.0, 2.5, 5.0],
    "amp": [0.90, 0.95, 1.0, 1.05, 1.10],
}
_U_COL = {"fcl": "u1", "fsb": "u2", "amp": "u3"}

def u_to_phys(axis, u):
    if axis == "amp":
        return 1.0 + AMP_SPAN_RATIO * float(u)
    return FREQ_SPAN_KHZ * float(u)

def u_col(axis, data):
    return data[_U_COL[axis]]

TRUE_COLOR = "black"

# Set True to colour scatter points by their observed Q value (magma: dark=0 → yellow=1).
# When False the original flat colour is used.
USE_SCORE_COLOR = False

HEATMAP_MAP = {
    ("fcl", "fsb"): ("sideband_fcl",   "fcl_2pi_khz",      "sideband_2pi_khz"),
    ("fcl", "amp"): ("rabi_fcl",       "fcl_2pi_khz",      "rabi_ratio"),
    ("fsb", "amp"): ("rabi_sideband",  "sideband_2pi_khz", "rabi_ratio"),
}
TRUE_CONTOUR_LEVELS = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

PANEL_LABELS = [[r"\textbf{a)}", r"\textbf{b)}"],
                [r"\textbf{c)}", r"\textbf{d)}"],
                [r"\textbf{e)}", r"\textbf{f)}"],]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_csv_file(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def load_true_score():
    rows = read_csv_file(TRUE_SCORE_FILE)
    out = {}
    for ax in AXES_1D:
        rs = sorted(
            [r for r in rows if r["axis"] == ax],
            key=lambda r: float(r["x_physical"]),
        )
        if ax == "amp":
            out[ax] = (np.array([1.0 + 0.2 * float(r["x_u"]) for r in rs]),
                       np.array([float(r["score"]) for r in rs]))
        else:
            out[ax] = (np.array([float(r["x_physical"]) for r in rs]),
                       np.array([float(r["score"]) for r in rs]))
    return out

def load_gp_slices():
    """Average the N_SEEDS_FOR_AVERAGE seeds' GP curves at each grid point,
    combining within-seed GP uncertainty and between-seed spread:
        sigma_total^2 = mean(sigma_i^2) + var(mu_i)
    """
    rows = read_csv_file(GP_SLICES_FILE)
    out = {ax: {} for ax in AXES_1D}
    for ax in AXES_1D:
        for it in ITER_GPS:
            rs = [r for r in rows
                  if r["N"] == N_FOCUS and r["axis"] == ax and int(r["iter"]) == it]
            if not rs:
                continue
            grouped = {}
            for r in rs:
                grouped.setdefault(float(r["x_u"]), []).append(r)
            xs = sorted(grouped)
            mus, sigmas = [], []
            for x_u in xs:
                mu_vals = np.array([float(r["mu"]) for r in grouped[x_u]])
                sig_vals = np.array([float(r["sigma"]) for r in grouped[x_u]])
                mus.append(mu_vals.mean())
                sigmas.append(np.sqrt(max((sig_vals ** 2).mean() + mu_vals.var(), 0.0)))
            if ax == "amp":
                x_phys = np.array([1.0 + 0.2 * x_u for x_u in xs])
            else:
                phys_by_u = {float(r["x_u"]): float(r["x_physical"]) for r in rs}
                x_phys = np.array([phys_by_u[x_u] for x_u in xs])
            out[ax][it] = (x_phys, np.array(mus), np.array(sigmas))
    return out

def load_contour_2d(ax_x, ax_y):
    """Return (x_phys, y_phys, score) for the 2D contour landscape, from the
    varms_2 heatmap CSVs. The heatmap's own scan already spans ±10 kHz
    (fcl/sideband) and [0.8, 1.2] (rabi_ratio = A/A_opt directly, no rescale
    needed) -- comfortably wider than the scale=0.5 window (±5 kHz, [0.9,1.1])
    these panels display, so no rescaling is needed either: just convert with
    the same factor plot_contour.py uses and let the existing axis limits
    (AXIS_LIMS, applied via set_xlim/set_ylim) crop the view to scale=0.5.
    """
    FREQ_SCALE = 5.0   # matches plot_contour.py: stored units are 2pi kHz
    for (kx, ky), (suffix, col_x, col_y) in HEATMAP_MAP.items():
        if {kx, ky} == {ax_x, ax_y}:
            path = DATA_DIR / f"varms_2_heatmap_{suffix}.csv"
            rows = [r for r in read_csv_file(path) if r.get("scan_kind", "heatmap") == "heatmap"]
            cx = col_x if ax_x == kx else col_y
            cy = col_y if ax_y == ky else col_x
            scale_x = FREQ_SCALE if cx in ("fcl_2pi_khz", "sideband_2pi_khz") else 1.0
            scale_y = FREQ_SCALE if cy in ("fcl_2pi_khz", "sideband_2pi_khz") else 1.0
            x = np.array([float(r[cx]) * scale_x for r in rows])
            y = np.array([float(r[cy]) * scale_y for r in rows])
            z = np.array([float(r["score"]) for r in rows])
            return x, y, z
    raise KeyError(f"No mapping for {ax_x}, {ax_y}")

def load_all_traces(n_label):
    d = trace_dir_path(n_label)
    paths = sorted(glob.glob(str(d / "trace_seed*_ucb.csv")))
    if not paths:
        raise FileNotFoundError(f"No traces in {d}")
    au1, au2, au3, ay = [], [], [], []
    ru1, ru2, ru3, ry = [], [], [], []
    for path in paths:
        for row in read_csv_file(path):
            au1.append(float(row["x_acq_u1"]))
            au2.append(float(row["x_acq_u2"]))
            au3.append(float(row["x_acq_u3"]))
            ru1.append(float(row["x_rec_u1"]))
            ru2.append(float(row["x_rec_u2"]))
            ru3.append(float(row["x_rec_u3"]))
            ay.append(float(row["y_acq"]))
            ry.append(float(row["q_det_rec"]))
    acq = {"u1": np.array(au1), "u2": np.array(au2), "u3": np.array(au3),
           "score": np.array(ay)}
    rec = {"u1": np.array(ru1), "u2": np.array(ru2), "u3": np.array(ru3),
           "score": np.array(ry)}
    return acq, rec

# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _tick_setup(ax, ax_x, ax_y=None, hide_x=False):
    ax.set_xticks(AXIS_TICKS[ax_x])
    if ax_y:
        ax.set_yticks(AXIS_TICKS[ax_y])
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if hide_x:
        ax.tick_params(labelbottom=False)

def _fuzzy_gradient_fill(ax, u, mu, sig, color, zorder=2):
    """50-layer fuzzy gradient fill identical to the reference script."""
    n_layers = 25 # Reduced from 50 to optimize rendering since it draws 3 times
    quantiles = np.linspace(0.05, 0.45, n_layers)
    base_alpha = 0.4 / n_layers
    for q in quantiles:
        z = sp_norm.ppf(1.0 - q)
        lower = np.clip(mu - z * sig, 0, 1)
        upper = np.clip(mu + z * sig, 0, 1)
        ax.fill_between(u, lower, upper, color=color, alpha=base_alpha,
                        linewidth=0, edgecolor="none", zorder=zorder)

def draw_1d_slice(ax, axis, true_data, gp_data, show_ylabel):
    # Add Gridlines
    ax.grid(True, which="major", color="#cccccc", linestyle=":", linewidth=0.8, zorder=0)

    x_tr, y_tr = true_data[axis]
    ax.plot(x_tr, y_tr, color=TRUE_COLOR, lw=1.5, ls="--", zorder=4)

    gp = gp_data.get(axis) if gp_data else None
    if gp is not None:
        for i, it in enumerate(ITER_GPS):
            if it in gp:
                x_gp, mu, sig = gp[it]
                ax.plot(x_gp, np.clip(mu, 0, 1), color=COLORS_GP[i], lw=1.5, zorder=3)
                # Apply fuzzy fill to all requested iterations
                _fuzzy_gradient_fill(ax, x_gp, mu, sig, COLORS_GP[i], zorder=2)

    ax.set_xlim(*AXIS_LIMS[axis])
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0])

    if show_ylabel:
        ax.set_ylabel(r"$Q$")
    else:
        ax.tick_params(labelleft=False)
        
    _tick_setup(ax, axis, hide_x=True)
    
    if axis in ["fcl", "fsb"]:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

def draw_2d_scatter(ax, data, ax_x, ax_y, col_idx, is_bottom_row, scatter_color, hide_ylabels=False, hide_yticklabels=False, contour_levels=None):
    # Add Gridlines
    ax.grid(True, which="major", color="#cccccc", linestyle=":", linewidth=0.8, zorder=0)

    xp = np.array([u_to_phys(ax_x, v) for v in u_col(ax_x, data)])
    yp = np.array([u_to_phys(ax_y, v) for v in u_col(ax_y, data)])
    xl, yl = AXIS_LIMS[ax_x], AXIS_LIMS[ax_y]

    if USE_SCORE_COLOR:
        scores = data.get("score", np.full(len(xp), np.nan))
        valid  = ~np.isnan(scores)
        if valid.any():
            ax.scatter(xp[valid], yp[valid], s=2.5,
                       c=scores[valid], cmap="magma", vmin=0.0, vmax=1.0,
                       alpha=0.25, edgecolors="none", zorder=1)
        if (~valid).any():
            ax.scatter(xp[~valid], yp[~valid], s=2.5,
                       color=scatter_color, alpha=0.12, edgecolors="none", zorder=1)
    else:
        ax.scatter(xp, yp, s=2.5, color=scatter_color, alpha=0.12, edgecolors="none", zorder=1)

    # Overlaid True score contour lines
    if contour_levels is None:
        contour_levels = TRUE_CONTOUR_LEVELS
    try:
        x_bg, y_bg, z_bg = load_contour_2d(ax_x, ax_y)
        cs = ax.tricontour(x_bg, y_bg, z_bg, levels=contour_levels,
                           colors="#555555", linewidths=0.45, linestyles="-", alpha=0.85, zorder=3)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    except (FileNotFoundError, KeyError) as e:
        print(f"  Heatmap contours unavailable for ({ax_x},{ax_y}): {e}", file=sys.stderr)

    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    
    if is_bottom_row:
        ax.set_xlabel(AXIS_LABELS[ax_x])
    else:
        ax.set_xlabel("")
        
    if hide_ylabels:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(AXIS_LABELS[ax_y])
        if hide_yticklabels:
            ax.tick_params(labelleft=False)
    hide_x = not is_bottom_row
    _tick_setup(ax, ax_x, ax_y, hide_x=hide_x)
    
    if ax_x in ["fcl", "fsb"]:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    if ax_y in ["fcl", "fsb"]:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

def add_panel_label(ax, label, x=-0.10):
    ax.text(x, 1.0, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold",
            clip_on=False, zorder=20)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_figure():
    true_data = load_true_score()
    gp_data   = load_gp_slices()
    acq_data, rec_data = load_all_traces(N_FOCUS)

    # 3×2 layout: row 1 shows fsb and amp slices; rows 2&3 show (fsb,fcl) and (amp,fcl)
    # so that δ and Ω land on the x-axis (transposed from original cols 0 & 2).
    ROW1_AXES   = ["fsb", "amp"]
    ROW23_PAIRS = [("fsb", "fcl"), ("amp", "fcl")]

    # Nested GridSpec: row 1 separated from the 2×2 block; rows 2&3 share the
    # same column width as row 1 (wspace=0.42 in both) and are flush vertically.
    fig_w = 3.375
    fig = plt.figure(figsize=(fig_w, 4.75))
    gs_outer = mgridspec.GridSpec(
        2, 1, figure=fig,
        left=0.17, right=0.88, top=0.87, bottom=0.08,
        height_ratios=[0.60, 1.7], hspace=0.12,
    )
    gs_r1  = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0],
                                               wspace=0.42)
    gs_r23 = mgridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[1],
                                               hspace=0.12, wspace=0.42)
    axes = np.array([
        [fig.add_subplot(gs_r1[0, 0]),   fig.add_subplot(gs_r1[0, 1])],
        [fig.add_subplot(gs_r23[0, 0]),  fig.add_subplot(gs_r23[0, 1])],
        [fig.add_subplot(gs_r23[1, 0]),  fig.add_subplot(gs_r23[1, 1])],
    ])

    # Row 1: 1D GP slices (fsb and amp only; rug marks omitted)
    for col, axis in enumerate(ROW1_AXES):
        ax = axes[0, col]
        draw_1d_slice(ax, axis, true_data, gp_data, show_ylabel=(col == 0))
        add_panel_label(ax, PANEL_LABELS[0][col], x=(-0.30 if col == 0 else -0.10))

    # Rows 2 & 3: 2D scatter with box (square) aspect ratio
    _levels_col1 = [l for l in TRUE_CONTOUR_LEVELS if l != 0.9]

    for col, (ax_x, ax_y) in enumerate(ROW23_PAIRS):
        hide_ylabels = (col != 0)
        levels = _levels_col1 if col == 1 else None

        # Row 2: acquisition samples
        ax = axes[1, col]
        draw_2d_scatter(ax, acq_data, ax_x, ax_y, col, is_bottom_row=False,
                        scatter_color="#005AB5", hide_ylabels=hide_ylabels,
                        contour_levels=levels)
        add_panel_label(ax, PANEL_LABELS[1][col], x=(-0.30 if col == 0 else -0.10))
        if col == 1:
            ax.text(1.08, 0.5, r"samples ($\bm{x}_{n}$)", transform=ax.transAxes,
                    rotation=-90, va="center", ha="left", fontsize=10, fontweight="bold")

        # Row 3: recommended best point
        ax = axes[2, col]
        draw_2d_scatter(ax, rec_data, ax_x, ax_y, col, is_bottom_row=True,
                        scatter_color="#E05C5C", hide_ylabels=hide_ylabels,
                        contour_levels=levels)
        add_panel_label(ax, PANEL_LABELS[2][col], x=(-0.30 if col == 0 else -0.10))
        if col == 1:
            ax.text(1.08, 0.5, r"estimated max ($\bm{x}_n^*$)", transform=ax.transAxes,
                    rotation=-90, va="center", ha="left", fontsize=10, fontweight="bold")

    # Two-row legend (ncol=3).  Matplotlib fills column-major (top-to-bottom per
    # column), so to display row-major we interleave: col0=[it0,Qdet], col1=[it1,mu],
    # col2=[it2,sigma] → handles order: it0, Qdet, it1, mu, it2, sigma.
    leg_handles = [
        Line2D([0],[0], color=COLORS_GP[0], lw=1.5, label=rf"${ITER_GPS[0]}$"),
        Line2D([0],[0], color="gray", lw=1.5, ls="--", label=r"$Q_\mathrm{ideal}$"),
        Line2D([0],[0], color=COLORS_GP[1], lw=1.5, label=rf"${ITER_GPS[1]}$"),
        Line2D([0],[0], color="gray", lw=1.5,          label=r"$\hat{\mu}$"),
        Line2D([0],[0], color=COLORS_GP[2], lw=1.5, label=rf"${ITER_GPS[2]}$"),
        Patch(facecolor="gray", alpha=0.35, edgecolor="none",
              label=r"$\hat{\sigma}$"),
    ]
    leg = fig.legend(handles=leg_handles, loc="upper center", bbox_to_anchor=(0.55, 0.99),
                     ncol=3, frameon=True, edgecolor="black", fancybox=False,
                     fontsize=9, handlelength=1.8, handletextpad=0.4, columnspacing=0.9)
    fig.canvas.draw()
    leg_bb      = leg.get_window_extent()
    fig_w_px    = fig.get_figwidth()  * fig.dpi
    fig_h_px    = fig.get_figheight() * fig.dpi
    row1_y_frac = (leg_bb.y1 - leg_bb.height * 0.25) / fig_h_px
    left_x_frac = (leg_bb.x0 - 6) / fig_w_px
    fig.text(left_x_frac, row1_y_frac, r"$n$",
             fontsize=9, ha="right", va="center")

    return fig

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out = FIGURE_DIR / f"figure_fcl_amp_n{N_FOCUS}_gpzoom.pdf"
    fig.savefig(out)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()