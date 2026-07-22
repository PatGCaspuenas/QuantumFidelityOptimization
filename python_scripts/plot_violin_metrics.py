#!/usr/bin/env python3
"""
plot_violin_metrics.py — GP fit-quality split violin plots.

Each violin is split vertically:
  left  half  →  RMSE  (left y-axis, log scale)
  right half  →  MSSE  (right y-axis, linear scale;  ≈1 = well-calibrated)

X-axis : N groups, each with one split violin per m value.
"""

import argparse
import os

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from scipy.stats import gaussian_kde
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
_HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_HERE, "data")
FIGURE_DIR  = os.path.join(_HERE, "figures")
CSV_FILE    = os.path.join(DATA_DIR, "gp_fit_quality_3d_2ms.csv")
# ═══════════════════════════════════════════════════════════════════════════════

def _magma_palette(n):
    """n evenly-spaced colours from magma (avoiding the very dark/light extremes)."""
    import seaborn as sns
    return sns.color_palette("magma", n_colors=n+2).as_hex()[1:-1]

def apply_style():
    # ---------------------------------------------------------------------------
    # APS Plot Formatting (PRX Intelligence / PR Applied)
    # ---------------------------------------------------------------------------
    mpl.rcParams.update({
        "figure.figsize": (3.375, 3),  
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


def _kde_half(ax, values, x, side, half_width, color, alpha, log_scale, zorder):
    """Draw one half of a split violin on `ax`."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 4:
        med = np.median(vals) if len(vals) else np.nan
        if np.isfinite(med):
            ax.scatter([x], [med], color=color, s=20, zorder=zorder + 2)
        return

    if log_scale:
        v = np.log(vals[vals > 0])
        if len(v) < 4:
            return
        kde  = gaussian_kde(v, bw_method="silverman")
        lo, hi = v.min(), v.max()
        pad  = 0.08 * (hi - lo) if hi > lo else 0.1
        grid = np.linspace(lo - pad, hi + pad, 300)
        dens = kde(grid)
        ys   = np.exp(grid)
    else:
        kde  = gaussian_kde(vals, bw_method="silverman")
        lo, hi = vals.min(), vals.max()
        pad  = 0.05 * (hi - lo) if hi > lo else 1e-10
        grid = np.linspace(lo - pad, hi + pad, 300)
        dens = kde(grid)
        ys   = grid

    dens = dens / dens.max() * half_width

    if side == "left":
        ax.fill_betweenx(ys, x - dens, x,
                         color=color, alpha=alpha, linewidth=0, zorder=zorder)
        ax.plot(x - dens, ys, color=color, lw=0.6, alpha=0.9, zorder=zorder + 1)
    else:
        ax.fill_betweenx(ys, x, x + dens,
                         color=color, alpha=alpha, linewidth=0, zorder=zorder)
        ax.plot(x + dens, ys, color=color, lw=0.6, alpha=0.9, zorder=zorder + 1)

def _kde_full(ax, values, x, half_width, color, alpha, log_scale, zorder):
    """Full (both-halves) violin on a single axis."""
    _kde_half(ax, values, x, "left",  half_width, color, alpha, log_scale, zorder)
    _kde_half(ax, values, x, "right", half_width, color, alpha, log_scale, zorder)

def add_left_title(ax, leg, text):
    """Dynamically places a text title to the immediate left of the legend box."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bbox.x0 - 0.02, bbox.y0 + (bbox.height / 2), text, 
            transform=ax.transAxes, ha="right", va="center", fontsize=10)

def add_slave_legend(ax, leg_master, handle, label):
    """Places a slave legend vertically centered in the far-right side of the master legend box."""
    fig = ax.figure
    fig.canvas.draw()
    # Grab the exact physical boundaries of the drawn Master Legend box
    bbox = leg_master.get_window_extent().transformed(ax.transAxes.inverted())
    
    # Place the slave legend in the vertical center of those boundaries
    leg_slave = ax.legend(handle, label, loc="center right", 
                          bbox_to_anchor=(bbox.x1 + 0.01, (bbox.y0 + bbox.y1) / 2),
                          fontsize=9, handlelength=1.5, frameon=False)
    ax.add_artist(leg_slave)

def _ax_style(ax, group_xs):
    """Common axis chrome for a single violin subplot."""
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_linewidth(0.8)
    ax.tick_params(which="both", direction="in", top=True, right=False)
    ax.tick_params(axis="x", which="both", length=0)
    ax.yaxis.set_ticks_position("left")
    ax.grid(True, axis="y", which="major", color="#e0e0e0", linestyle=":", linewidth=0.7, zorder=0)
    ax.grid(True, axis="y", which="minor", color="#eeeeee", linestyle=":", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for g in group_xs[1:]:
        ax.axvline(g - 0.5, color="0.88", lw=0.8, zorder=0)


def _fmt_n(n):
    if np.isinf(n):
        return r"$\infty$"
    exp = int(np.round(np.log10(n)))
    return rf"$10^{{{exp}}}$"


def _set_xaxis(ax, group_xs, nshots_vals):
    ax.set_xticks(group_xs)
    ax.set_xticklabels([_fmt_n(n) for n in nshots_vals])
    ax.set_xlabel(r"$N$")
    ax.set_xlim(-0.5, len(nshots_vals) - 0.5)


def _place_legend(fig, handles, **kw_extra):
    """Draw fig.legend without a title, then place 'm' as text to its left."""
    kw = dict(ncol=3, fontsize=9,
              handlelength=1.5, handleheight=0.8,
              borderpad=0.5, labelspacing=0.2, columnspacing=0.6,
              frameon=True, edgecolor="black", facecolor="white", fancybox=False)
    kw.update(kw_extra)
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.58, 0.98),
                     bbox_transform=fig.transFigure, **kw)
    # Place "m" label to the left of the legend box in figure coordinates
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x0 - 0.02, (bb.y0 + bb.y1) / 2, r"$m$",
             ha="right", va="center", fontsize=10)
    return leg


def _legend_npts(fig, npts_color, npts_vals):
    h = [Patch(facecolor=npts_color[n], alpha=0.85, label=rf"${n}$") for n in npts_vals]
    _place_legend(fig, h)


def build_figures(msse_variant="msse_gp"):
    df = pd.read_csv(CSV_FILE, encoding="utf-8")

    msse_col = "msse_gp" if msse_variant == "msse_gp" else "msse"

    npts_vals   = sorted(df["n_pts"].unique())
    nshots_vals = sorted(df["N_shots"].unique())
    colors      = _magma_palette(len(npts_vals))
    npts_color  = {n: colors[i] for i, n in enumerate(npts_vals)}

    n_v      = len(npts_vals)
    v_gap    = 0.18
    half_vw  = 0.07
    offsets  = np.linspace(-(n_v - 1) / 2, (n_v - 1) / 2, n_v) * v_gap
    group_xs = list(range(len(nshots_vals)))
    fig_w    = 3.375

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 1 — RMSE  (with σ_ε dashed lines + sigma legend entry)
    # ═══════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(fig_w, 3))
    fig1.tight_layout(rect=[0.10, 0.06, 1.0, 0.87])

    ax1.set_yscale("log");  ax1.set_ylim(1e-3, 1e0)
    _ax_style(ax1, group_xs)

    for g_idx, nshots in enumerate(nshots_vals):
        gc = group_xs[g_idx]
        # σ_ε dashed line per N_shots group
        s_vals = df.loc[df["N_shots"] == nshots, "mean_sigma_noise"].dropna().values
        if len(s_vals):
            xmin = gc + offsets[0]  - half_vw * 1.5
            xmax = gc + offsets[-1] + half_vw * 1.5
            ax1.hlines(np.median(s_vals), xmin, xmax,
                       color="0.3", linestyle="--", linewidth=1.2, zorder=3)
        for v_idx, npts in enumerate(npts_vals):
            xpos = gc + offsets[v_idx]
            mask = (df["N_shots"] == nshots) & (df["n_pts"] == npts)
            rmse_vals = df.loc[mask, "rmse"].dropna().values
            _kde_full(ax1, rmse_vals, xpos, half_vw, npts_color[npts],
                      alpha=0.75, log_scale=True, zorder=2)

    _set_xaxis(ax1, group_xs, nshots_vals)
    ax1.set_ylabel("RMSE")

    # Legend: n_pts patches + σ_ε dashed line
    from matplotlib.lines import Line2D
    h_npts = [Patch(facecolor=npts_color[n], alpha=0.85, label=rf"${n}$") for n in npts_vals]
    h_sigma = [Line2D([0], [0], color="0.3", ls="--", lw=1.2, label=r"$\bar{\sigma}_\varepsilon$")]
    _place_legend(fig1, h_npts + h_sigma,
                  ncol=min(4, len(h_npts) + len(h_sigma)))

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 2 — MSSE  (no sigma lines, current legend)
    # ═══════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(fig_w, 3))
    fig2.tight_layout(rect=[0.10, 0.06, 1.0, 0.87])

    ax2.set_yscale("log");  ax2.set_ylim(1e-1, 1e1)
    _ax_style(ax2, group_xs)
    ax2.axhline(1.0, color="0.45", lw=0.9, ls="--", alpha=0.6, zorder=1)

    for g_idx, nshots in enumerate(nshots_vals):
        gc = group_xs[g_idx]
        for v_idx, npts in enumerate(npts_vals):
            xpos = gc + offsets[v_idx]
            mask = (df["N_shots"] == nshots) & (df["n_pts"] == npts)
            msse_vals = df.loc[mask, msse_col].dropna().values
            _kde_full(ax2, msse_vals, xpos, half_vw, npts_color[npts],
                      alpha=0.75, log_scale=True, zorder=2)

    _set_xaxis(ax2, group_xs, nshots_vals)
    ax2.set_ylabel(r"$\mathrm{MSSE}$")

    _legend_npts(fig2, npts_color, npts_vals)

    return fig1, fig2

def main():
    apply_style()

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--msse", default="msse_gp",
                        choices=["msse_gp", "msse_adj"],
                        help="MSSE variant")
    parser.add_argument("--out-rmse", default=os.path.join(FIGURE_DIR, "violin_rmse.pdf"))
    parser.add_argument("--out-msse", default=os.path.join(FIGURE_DIR, "violin_msse.pdf"))
    args = parser.parse_args()

    fig1, fig2 = build_figures(msse_variant=args.msse)

    Path(args.out_rmse).parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(args.out_rmse)
    print(f"Saved → {args.out_rmse}")
    plt.close(fig1)

    Path(args.out_msse).parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.out_msse)
    print(f"Saved → {args.out_msse}")
    plt.close(fig2)

if __name__ == "__main__":
    main()