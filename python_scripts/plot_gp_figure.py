#!/usr/bin/env python3
"""
Generates figures/paper/figure_fcl_amp_n100_gpzoom.pdf

3×3 figure for N=100 BO traces (scale=0.5, full_l1):
  Row 1 — 1D GP posterior slices along fcl, fsb, amp for multiple iterations (10, 50, 100).
  Row 2 — 2D scatter of acquisition samples (x_acq) across all seeds and iters.
  Row 3 — 2D scatter of recommended best point (x_rec) across all seeds and iters.

Features:
  - Columns share x-axes (fcl, fsb, amp).
  - Y-axes for rows 2 and 3 cycle through the combinations to show all 2D pairings.
  - True Q_det landscape is overlaid on the 2D plots using black contour lines with labels.
  - Transparent scatter points reveal raw distributions.h
  - Training points near the slice are shown as a red square rug at the bottom.
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
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
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
    "backend": "pdf",   
})

# Extract 3 distinct colors from the magma palette for the GP iterations
magma_cols = sns.color_palette("magma", n_colors=6).as_hex()
COLORS_GP = [magma_cols[1], magma_cols[3], magma_cols[4]]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR  = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT

GP_STEM         = "scale05_selectedN_full_l1_avg5_gp_slices_iters10_30_50_100"
GP_SLICES_FILE  = DATA_DIR / "gp_slices" / f"{GP_STEM}.csv"
TRAINING_FILE   = DATA_DIR / "gp_slices" / f"{GP_STEM}_training_points.csv"
TRUE_SCORE_FILE = DATA_DIR / "score_cache" / "scale05_true_2ms_full_l1_slices_freqspan10_grid201.csv"

GRID_N = int(os.environ.get("GRID3D_N", "21"))
GRID3D_FILE = DATA_DIR / "score_cache" / f"scale05_true_2ms_full_l1_3d_bound050_grid{GRID_N}.csv"

def trace_dir_path(n_label):
    return DATA_DIR / f"traces_freqspan10_bound050_full_l1_N{n_label}_nostop100_stream_100seeds"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_FOCUS       = os.environ.get("PLOT_N_FOCUS", "100")
ITER_GPS      = [10, 30, 50]
SELECTED_RANK = 1        

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
    "fcl": r"$\Delta \omega_\mathrm{cl}\ (\mathrm{kHz})$",
    "fsb": r"$\Delta \delta\ (\mathrm{kHz})$",
    "amp": r"$\Omega / \Omega_\mathrm{opt}$",
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

PANEL_LABELS = [[r"\textbf{a)}", r"\textbf{b)}", r"\textbf{c)}"],
                [r"\textbf{d)}", r"\textbf{e)}", r"\textbf{f)}"],
                [r"\textbf{g)}", r"\textbf{h)}", r"\textbf{i)}"]]

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
            [r for r in rows if r["axis"] == ax and r.get("score_mode", "full_l1") == "full_l1"],
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
    rows = read_csv_file(GP_SLICES_FILE)
    out = {ax: {} for ax in AXES_1D}
    for ax in AXES_1D:
        for it in ITER_GPS:
            rs = sorted(
                [r for r in rows
                 if r["N"] == N_FOCUS
                 and r["axis"] == ax
                 and int(r["selected_rank"]) == SELECTED_RANK
                 and int(r["iter"]) == it],
                key=lambda r: float(r["x_physical"]),
            )
            if rs:
                if ax == "amp":
                    x_phys = np.array([1.0 + 0.2 * float(r["x_u"]) for r in rs])
                else:
                    x_phys = np.array([float(r["x_physical"]) for r in rs])
                    
                out[ax][it] = (x_phys,
                               np.array([float(r["mu"]) for r in rs]),
                               np.array([float(r["sigma"]) for r in rs]))
    return out

def load_training_points():
    rows = read_csv_file(TRAINING_FILE)
    rs = [r for r in rows
          if r["N"] == N_FOCUS
          and int(r["selected_rank"]) == SELECTED_RANK
          and int(r["iter"]) == ITER_GPS[-1]]
    if not rs:
        return None
    return {
        "u1": np.array([float(r["x_u1"]) for r in rs]),
        "u2": np.array([float(r["x_u2"]) for r in rs]),
        "u3": np.array([float(r["x_u3"]) for r in rs]),
        "fcl": np.array([u_to_phys("fcl", r["x_u1"]) for r in rs]),
        "fsb": np.array([u_to_phys("fsb", r["x_u2"]) for r in rs]),
        "amp": np.array([u_to_phys("amp", r["x_u3"]) for r in rs]),
        "y":   np.array([float(r["y"])                for r in rs]),
    }

def get_near_train_pts(train_pts, ax_key, tol=0.1):
    """Return (coords, original_indices) for training points near the 1D slice."""
    u1, u2, u3 = train_pts["u1"], train_pts["u2"], train_pts["u3"]
    if ax_key == "fcl":
        mask = np.sqrt(u2**2 + u3**2) < tol
    elif ax_key == "fsb":
        mask = np.sqrt(u1**2 + u3**2) < tol
    elif ax_key == "amp":
        mask = np.sqrt(u1**2 + u2**2) < tol
    else:
        return np.array([]), np.array([], dtype=int)
    return train_pts[ax_key][mask], np.where(mask)[0]

_3D_U_COL   = {"fcl": "u1",         "fsb": "u2",         "amp": "u3"}
_3D_PHY_COL = {"fcl": "x_fcl_khz",  "fsb": "x_fsb_khz",  "amp": "x_amp_ratio"}

def load_contour_2d(ax_x, ax_y):
    """Return (x_phys, y_phys, score) for the 2D contour landscape.

    Primary: central slice through the 3D score grid — physically correct at the
    actual scale=0.5 coordinates (±5 kHz, [0.9, 1.1]).
    Fallback: varms_2 heatmap CSVs rescaled from their scale=1 scan ranges.
    """
    if GRID3D_FILE.exists():
        return _load_3d_slice(ax_x, ax_y)
    return _load_heatmap_scaled(ax_x, ax_y)

def _load_3d_slice(ax_x, ax_y):
    """Central (u=0) slice of the 3D score grid. No coordinate transformation needed."""
    all_axes = {"fcl", "fsb", "amp"}
    fixed_axis = (all_axes - {ax_x, ax_y}).pop()
    fixed_u_col = _3D_U_COL[fixed_axis]

    rows = read_csv_file(GRID3D_FILE)
    u_vals = sorted({float(r[fixed_u_col]) for r in rows})
    u_center = min(u_vals, key=abs)            # grid value closest to 0
    tol = (u_vals[1] - u_vals[0]) * 0.5 if len(u_vals) > 1 else 1e-9
    slice_rows = [r for r in rows if abs(float(r[fixed_u_col]) - u_center) < tol]

    x = np.array([float(r[_3D_PHY_COL[ax_x]]) for r in slice_rows])
    y = np.array([float(r[_3D_PHY_COL[ax_y]]) for r in slice_rows])
    # amp column stores ΔA/A_opt; display axis uses A/A_opt = 1 + ΔA/A_opt
    if ax_x == "amp":
        x = 1.0 + x
    if ax_y == "amp":
        y = 1.0 + y
    z = np.array([float(r["score"]) for r in slice_rows])
    return x, y, z

# ---------------------------------------------------------------------------
# Fallback: varms_2 heatmap CSVs (scale=1 generation, rescaled to scale=0.5)
# ---------------------------------------------------------------------------
# Heatmap scan ranges at generation time (BOUND_SCALE=1.0):
#   fcl/sideband: ±2 kHz  →  rescale to ±(FREQ_SPAN_KHZ × BOUND_SCALE) = ±5 kHz
#   rabi_ratio:   [0.8, 1.2]  →  rescale to [1-AMP_SPAN×BS, 1+AMP_SPAN×BS] = [0.9, 1.1]
_CONTOUR_FREQ_RANGE = 2.0   # half-span of the original scan in kHz
_FREQ_COORD_SCALE   = (FREQ_SPAN_KHZ * BOUND_SCALE) / _CONTOUR_FREQ_RANGE   # = 2.5
_AMP_COORD_SCALE    = BOUND_SCALE                                             # = 0.5

def _heatmap_coord_scale(col_name, val):
    if col_name in ("fcl_2pi_khz", "sideband_2pi_khz"):
        return val * _FREQ_COORD_SCALE
    if col_name == "rabi_ratio":
        return 1.0 + (val - 1.0) * _AMP_COORD_SCALE
    return val

def _load_heatmap_scaled(ax_x, ax_y):
    for (kx, ky), (suffix, col_x, col_y) in HEATMAP_MAP.items():
        if {kx, ky} == {ax_x, ax_y}:
            path = DATA_DIR / f"varms_2_heatmap_{suffix}.csv"
            rows = [r for r in read_csv_file(path) if r.get("scan_kind", "heatmap") == "heatmap"]
            cx = col_x if ax_x == kx else col_y
            cy = col_y if ax_y == ky else col_x
            x = np.array([_heatmap_coord_scale(cx, float(r[cx])) for r in rows])
            y = np.array([_heatmap_coord_scale(cy, float(r[cy])) for r in rows])
            z = np.array([float(r["score"]) for r in rows])
            return x, y, z
    raise KeyError(f"No mapping for {ax_x}, {ax_y}")

def _row_score(row, candidates):
    """Return the first parseable score value from a list of candidate column names."""
    for col in candidates:
        v = row.get(col)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return np.nan

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
            ay.append(_row_score(row, ["y_acq", "score_noisy", "y", "score"]))
            ry.append(_row_score(row, ["y_rec", "score_rec", "score_best", "score_det", "score"]))
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

def draw_1d_slice(ax, axis, true_data, gp_data, train_pts, show_ylabel):
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

    if train_pts is not None:
        near_x, near_idx = get_near_train_pts(train_pts, axis, tol=0.1)
        if len(near_x) > 0:
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            # Split chronologically: first 10 (LHS init), next 20, last 20
            # and colour each batch by a distinct magma shade
            _cmap = mpl.colormaps.get_cmap("magma")
            batches = [(0, 10, _cmap(0.20)),
                       (10, 30, _cmap(0.55)),
                       (30, len(train_pts["u1"]), _cmap(0.68))]
            for b_start, b_end, b_color in batches:
                sel = (near_idx >= b_start) & (near_idx < b_end)
                if sel.any():
                    ax.plot(near_x[sel], np.full(sel.sum(), 0.03),
                            marker='|', markersize=7, markeredgewidth=1.2,
                            color=b_color, alpha=0.85, linestyle='none',
                            zorder=10, transform=trans, clip_on=False)

    ax.set_xlim(*AXIS_LIMS[axis])
    ax.set_ylim(0., 1.0)
    
    if show_ylabel:
        ax.set_ylabel(r"$Q$")
    else:
        ax.tick_params(labelleft=False)
        
    _tick_setup(ax, axis, hide_x=True)
    
    if axis in ["fcl", "fsb"]:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

def draw_2d_scatter(ax, data, ax_x, ax_y, col_idx, is_bottom_row, scatter_color, hide_ylabels=False, hide_yticklabels=False):
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
    try:
        x_bg, y_bg, z_bg = load_contour_2d(ax_x, ax_y)
        cs = ax.tricontour(x_bg, y_bg, z_bg, levels=TRUE_CONTOUR_LEVELS,
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

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    true_data = load_true_score()
    gp_data   = load_gp_slices()
    train_pts = load_training_points()
    acq_data, rec_data = load_all_traces(N_FOCUS)

    # Tight vertical layout because x-labels are shared. Generous wspace to fit all the Y-labels.
    fig, axes = plt.subplots(3, 3, figsize=(6.75, 5.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.08, hspace=0.10, wspace=0.30)

    # Row 1: 1D GP slices
    for col, axis in enumerate(AXES_1D):
        ax = axes[0, col]
        draw_1d_slice(ax, axis, true_data, gp_data, train_pts, show_ylabel=(col == 0))
        add_panel_label(ax, PANEL_LABELS[0][col], x=(-0.18 if col == 1 else -0.15))
    
    # Row 2: Acquisition scatter
    for col, (ax_x, ax_y) in enumerate(PAIRS_2D):
        ax = axes[1, col]
        draw_2d_scatter(ax, acq_data, ax_x, ax_y, col, is_bottom_row=False,
                        scatter_color="#005AB5", hide_yticklabels=(col == 2))
        add_panel_label(ax, PANEL_LABELS[1][col], x=(-0.18 if col == 1 else -0.15))
        if col == 2:
            ax.text(1.05, 0.5, r"samples ($\bm{x}_{n}$)", transform=ax.transAxes,
                    rotation=-90, va="center", ha="left", fontsize=10, fontweight="bold")

    # Row 3: Recommended-max scatter
    for col, (ax_x, ax_y) in enumerate(PAIRS_2D):
        ax = axes[2, col]
        draw_2d_scatter(ax, rec_data, ax_x, ax_y, col, is_bottom_row=True,
                        scatter_color="#E05C5C", hide_yticklabels=(col == 2))
        add_panel_label(ax, PANEL_LABELS[2][col], x=(-0.18 if col == 1 else -0.15))
        if col == 2:
            ax.text(1.05, 0.5, r"estimated max ($\bm{x}_n^*$)", transform=ax.transAxes,
                    rotation=-90, va="center", ha="left", fontsize=10, fontweight="bold")

    # Two-row legend (ncol=3).  Matplotlib fills column-major (top-to-bottom per
    # column), so to display row-major we interleave: col0=[it0,Qdet], col1=[it1,mu],
    # col2=[it2,sigma] → handles order: it0, Qdet, it1, mu, it2, sigma.
    leg_handles = [
        Line2D([0],[0], color=COLORS_GP[0], lw=1.5, label=rf"${ITER_GPS[0]}$"),
        Line2D([0],[0], color="gray", lw=1.5, ls="--", label=r"$Q_\mathrm{det}$"),
        Line2D([0],[0], color=COLORS_GP[1], lw=1.5, label=rf"${ITER_GPS[1]}$"),
        Line2D([0],[0], color="gray", lw=1.5,          label=r"$\hat{\mu}$"),
        Line2D([0],[0], color=COLORS_GP[2], lw=1.5, label=rf"${ITER_GPS[2]}$"),
        Patch(facecolor="gray", alpha=0.35, edgecolor="none",
              label=r"$\hat{\sigma}$"),
    ]
    leg = fig.legend(handles=leg_handles, loc="upper center", bbox_to_anchor=(0.54, 0.99),
                     ncol=3, frameon=True, edgecolor="black", fancybox=False,
                     fontsize=9, handlelength=1.8, handletextpad=0.4, columnspacing=0.9)
    # Place "n" outside the legend box to the left, vertically centred with row 1
    fig.canvas.draw()
    leg_bb      = leg.get_window_extent()
    fig_w_px    = fig.get_figwidth()  * fig.dpi
    fig_h_px    = fig.get_figheight() * fig.dpi
    row1_y_frac = (leg_bb.y1 - leg_bb.height * 0.25) / fig_h_px
    left_x_frac = (leg_bb.x0 - 6) / fig_w_px
    fig.text(left_x_frac, row1_y_frac, r"$n$",
             fontsize=9, ha="right", va="center")

    out = FIGURE_DIR / f"figure_fcl_amp_n{N_FOCUS}_gpzoom.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()