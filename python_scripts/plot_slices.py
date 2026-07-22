#!/usr/bin/env python3
"""
plot_slices.py — 1-D slice comparison: Q_det vs GP surrogate.

Layout: 1 row × 3 cols (one panel per axis: fcl / fsb / amp)
Each panel overlays GP predictions for all N_shots values (using magma palette).
GP uncertainty shown as a 50-layer fuzzy gradient fill.
Nearby training-point rug marks in red at the bottom of each panel.

For each N_shots the median-RMSE seed is used.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm as sp_norm
import seaborn as sns
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
_HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_HERE, "data")
FIGURE_DIR  = os.path.join(_HERE, "figures")
SLICES_CSV  = os.path.join(DATA_DIR, "slices_output.csv")
METRICS_CSV = os.path.join(DATA_DIR, "gp_fit_quality_3d_2ms.csv")
NEAR_CSV    = os.path.join(DATA_DIR, "train_near_output.csv")
OUT_FILE    = os.path.join(FIGURE_DIR, "slices_plot.pdf")

N_PTS_SHOW   = 50
N_SHOTS_SHOW = None
# ═══════════════════════════════════════════════════════════════════════════════

AXES_ORDER  = ["fcl", "fsb", "amp"]

# Using formatting identical to the contour plots
AXIS_XLABEL = {
    "fcl": r"$\Delta\omega_\mathrm{cl}\ (2 \pi \cdot \mathrm{kHz})$",
    "fsb": r"$\Delta\delta\ (2 \pi \cdot \mathrm{kHz})$",
    "amp": r"$\Omega / \Omega^*$",
}

# Input spans for the transformed variables
def axis_limits(axis):
    if axis in ("fcl", "fsb"):
        return (-5.0, 5.0)
    elif axis == "amp":
        return (0.9, 1.1)
    return None

def axis_ticks(axis):
    if axis in ("fcl", "fsb"):
        return [-5.0, 0.0, 5.0]
    elif axis == "amp":
        return [0.9, 1.0, 1.1]
    return None

def transform_u(u_vals, ax_key):
    """Transforms the underlying u bounds [-0.5, 0.5] to physical dimensions."""
    if ax_key in ("fcl", "fsb"):
        return 10.0 * u_vals
    elif ax_key == "amp":
        return 1.0 + 0.2 * u_vals
    return u_vals

C_DET = "0.20"
C_RUG = "#cc2222"

# Magma palette extracted identically to figure_ab_vertical.py
_MAGMA_COLORS = sns.color_palette("magma", n_colors=8).as_hex()
N_COLORS = {
    100:    _MAGMA_COLORS[6],
    1000:   _MAGMA_COLORS[5],
    10000:  _MAGMA_COLORS[4],
    100000: _MAGMA_COLORS[3],
    float('inf'): _MAGMA_COLORS[0],
}

def apply_style():
    # APS Plot Formatting
    mpl.rcParams.update({
        "figure.figsize": (6.75, 2.8),   # Double column width
        "font.family": "serif",          
        "mathtext.fontset": "cm",        
        "font.size": 10,                 
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,           
        "lines.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}\usepackage{xcolor}",
    })

def ns_label_no_N(ns):
    return r"$\infty$" if np.isinf(ns) else rf"$10^{{{int(np.log10(ns))}}}$"

def assign_colors(nshots_use):
    return {ns: N_COLORS.get(ns, "#000000") for ns in nshots_use}

def find_median_seeds(df_metrics, n_pts):
    sub = df_metrics[df_metrics["n_pts"] == n_pts]
    result = {}
    for ns, grp in sub.groupby("N_shots", sort=False):
        med = grp["rmse"].median()
        idx = (grp["rmse"] - med).abs().idxmin()
        result[ns] = (int(grp.loc[idx, "seed"]), float(grp.loc[idx, "rmse"]))
    return result

def _fuzzy_gradient_fill(ax, u, mu, sig, color, zorder=2):
    """Fuzzy 50-layer gradient fill replacing the solid blocks."""
    n_layers = 50
    quantiles = np.linspace(0.05, 0.45, n_layers)
    base_alpha = 0.5 / n_layers
    for q in quantiles:
        z = sp_norm.ppf(1.0 - q)
        lower = mu - z * sig
        upper = mu + z * sig
        ax.fill_between(u, lower, upper, color=color, alpha=base_alpha,
                        linewidth=0, edgecolor="none", zorder=zorder)

def draw_panel(ax, panel_df, near_df, ax_key, nshots_use, colors, first_col, panel_letter):
    # Q_det 
    ref = panel_df[panel_df["N_shots"] == nshots_use[0]].sort_values("u")
    if not ref.empty:
        u_trans = transform_u(ref["u"].values, ax_key)
        ax.plot(u_trans, ref["q_det"].values, color=C_DET, lw=1.2, ls="--", zorder=5)

    # GP surrogate
    for ns in nshots_use:
        d = panel_df[panel_df["N_shots"] == ns].sort_values("u")
        if d.empty:
            continue
        color = colors[ns]
        u_d_trans = transform_u(d["u"].values, ax_key)
        _fuzzy_gradient_fill(ax, u_d_trans, d["mu_gp"].values, d["sigma_gp"].values, color)
        ax.plot(u_d_trans, d["mu_gp"].values, color=color, lw=1.2, zorder=4)

    # Rug marks
    if near_df is not None and len(near_df) > 0:
        near_ax = near_df[near_df["axis"] == ax_key]
        for xu in np.unique(near_ax["u_proj"].values):
            xu_trans = transform_u(xu, ax_key)
            ax.axvline(xu_trans, ymin=0.0, ymax=0.04,
                       color=C_RUG, lw=0.9, alpha=0.85, zorder=7)

    ax.grid(True, which="major", color="#e0e0e0", linestyle=":", linewidth=0.6, zorder=0)
    ax.grid(True, which="minor", color="#eeeeee", linestyle=":", linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    
    # Restrict to scaled limits
    lims = axis_limits(ax_key)
    if lims:
        ax.set_xlim(*lims)
    
    ticks = axis_ticks(ax_key)
    if ticks:
        ax.set_xticks(ticks)
        
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    
    # Tick formatting strictly pointing inwards
    ax.tick_params(axis="y", which="both", direction="in", right=True)
    ax.tick_params(axis="x", which="both", direction="in", top=True)

    ax.set_xlabel(AXIS_XLABEL.get(ax_key, ax_key))

    if first_col:
        ax.set_ylabel(r"$Q$")
    else:
        ax.tick_params(labelleft=False)

    # Add panel letter (a, b, c) matching contour plots
    x_letter = -0.14 if first_col else -0.05
    ax.text(x_letter, 0.98, panel_letter, transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold")

def build_figure(n_pts=N_PTS_SHOW, nshots=N_SHOTS_SHOW, rug=True):
    df_slices  = pd.read_csv(SLICES_CSV)
    df_metrics = pd.read_csv(METRICS_CSV)

    median_seeds = find_median_seeds(df_metrics, n_pts)
    if not median_seeds:
        raise SystemExit(f"No metrics data for n_pts={n_pts}.")

    def _is_median(row):
        info = median_seeds.get(row["N_shots"])
        return info is not None and row["seed"] == info[0]

    sub = df_slices[df_slices["n_pts"] == n_pts].copy()
    sub = sub[sub.apply(_is_median, axis=1)]

    if sub.empty:
        raise SystemExit(f"No slice data for n_pts={n_pts}.")

    avail_nshots = sorted(sub["N_shots"].unique())
    nshots_use = [ns for ns in nshots if ns in avail_nshots] if nshots else avail_nshots
    colors = assign_colors(nshots_use)

    near_df = None
    if rug and os.path.isfile(NEAR_CSV):
        df_near = pd.read_csv(NEAR_CSV)
        near_df = df_near[(df_near["n_pts"] == n_pts) & df_near.apply(_is_median, axis=1)]

    # Figure Layout 
    fig, axes = plt.subplots(1, 3, constrained_layout=False, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.35, wspace=0.15)

    panel_letters = [r"\textbf{a)}", r"\textbf{b)}", r"\textbf{c)}"]

    for ci, ax_key in enumerate(AXES_ORDER):
        ax = axes[ci]
        panel = sub[sub["axis"] == ax_key]
        if panel.empty:
            ax.set_visible(False)
            continue
        draw_panel(ax, panel, near_df, ax_key, nshots_use, colors,
                   first_col=(ci == 0), panel_letter=panel_letters[ci])

    # -----------------------------------------------------------------------
    # LEGEND CONSTRUCTION (Properly Interleaved for Column-Major Filling)
    # -----------------------------------------------------------------------
    dummy_handle = Line2D([], [], color='none')
    
    h_n = [Line2D([0], [0], color=colors[ns], lw=1.5) for ns in nshots_use]
    l_n = [ns_label_no_N(ns) for ns in nshots_use]

    h_qdet = Line2D([0], [0], color=C_DET, lw=1.5, ls="--")
    l_qdet = r"$Q_\mathrm{ideal}$"

    h_mu = Line2D([0], [0], color="0.45", lw=1.5, ls="-")
    l_mu = r"$\hat{\mu}$"

    h_sig = Patch(facecolor="0.60", alpha=0.3, edgecolor="none")
    l_sig = r"$\hat{\sigma}$"

    if len(nshots_use) == 5:
        # Matplotlib populates legends COLUMN-BY-COLUMN. 
        # By alternating Top/Bottom components, we force 2 perfect rows.
        handles = [
            h_n[0], dummy_handle, # Col 1: 10^2 (top), blank (bottom)
            h_n[1], h_qdet,       # Col 2: 10^3 (top), Q_det (bottom)
            h_n[2], h_mu,         # Col 3: 10^4 (top), mu (bottom)
            h_n[3], h_sig,        # Col 4: 10^5 (top), sigma (bottom)
            h_n[4], dummy_handle  # Col 5: inf (top), blank (bottom)
        ]
        labels = [
            l_n[0], "", 
            l_n[1], l_qdet, 
            l_n[2], l_mu, 
            l_n[3], l_sig, 
            l_n[4], ""
        ]
        ncol_val = 5
    else:
        # Fallback 
        handles = h_n + [h_qdet, h_mu, h_sig]
        labels = l_n + [l_qdet, l_mu, l_sig]
        ncol_val = 4

    leg = fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=ncol_val,
        fontsize=9,
        handlelength=1.5, 
        handletextpad=0.5,
        columnspacing=1.8,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        fancybox=False
    )

    if len(nshots_use) == 5:
        # Hide the dummy texts completely to avoid bounding box artifacts
        texts = leg.get_texts()
        texts[1].set_alpha(0)  # Hides text under 10^2
        texts[9].set_alpha(0)  # Hides text under inf

    # Place the "N" label strictly outside the legend box aligned with the top row
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.transFigure.inverted())
    
    # In a 2-row layout, 0.75 targets the vertical center of the top row
    y_top_row = bbox.y0 + (bbox.y1 - bbox.y0) * 0.75
    fig.text(bbox.x0 - 0.01, y_top_row, r"$N$", ha="right", va="center", fontsize=10)

    return fig

def main():
    apply_style()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npts",   type=int,          default=N_PTS_SHOW)
    parser.add_argument("--nshots", type=float, nargs="+", default=N_SHOTS_SHOW)
    parser.add_argument("--no-rug", dest="rug", action="store_false", default=True)
    parser.add_argument("--out",    default=OUT_FILE)
    args = parser.parse_args()

    fig = build_figure(n_pts=args.npts, nshots=args.nshots, rug=args.rug)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()