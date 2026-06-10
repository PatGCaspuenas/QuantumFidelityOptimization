"""
Generates figures/paper/contour_2ms.pdf

2D parameter landscape heatmaps for the two-MS_(pi/2) calibration sequence.
Reads: data/varms_2_heatmap_{axis_x}_{axis_y}.csv
Score: full_l1 = P_ee (probability of |11> state, deterministic/no shot noise)

Panels (matching Fig. ref{fig:contour}):
  (a) amplitude vs sideband detuning
  (b) amplitude vs center-line detuning
  (c) sideband vs center-line detuning
"""

import csv
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/Library/TeX/texbin:" + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# APS formatting (matching plot_figure_ab_vertical.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.figsize": (6.75, 2.),
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
})
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
})
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MS = int(os.environ.get("PLOT_NUM_MS", "2"))
N_LEVELS = 20
CMAP = "magma"

# Three axis pairs shown in the paper figure
AXIS_PAIRS = [
    ("rabi", "sideband"),
    ("rabi", "fcl"),
    ("sideband", "fcl"),
]

PANEL_LABELS = ["(a)", "(b)", "(c)"]

# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def axis_column(axis):
    return {
        "rabi":     "rabi_ratio",
        "sideband": "sideband_2pi_khz",
        "fcl":      "fcl_2pi_khz",
        "phase":    "phase_pi",
    }[axis]


def axis_label(axis):
    return {
        "rabi":     r"$\Omega / \Omega_\mathrm{opt}$",
        "sideband": r"$\Delta f_\mathrm{sb}\ (\mathrm{kHz})$",
        "fcl":      r"$\Delta f_\mathrm{cl}\ (\mathrm{kHz})$",
        "phase":    r"$\Delta\phi / \pi$",
    }[axis]


FREQ_SCALE = 5.0   # stored units are 2pi kHz; multiply by 5 to get kHz


def axis_limits(axis):
    if axis in ("sideband", "fcl"):
        return (-10.0, 10.0)
    elif axis == "rabi":
        return (0.8, 1.2)
    elif axis == "phase":
        return (-0.5, 0.5)
    return None


def axis_ticks(axis):
    if axis == "rabi":
        return [0.8, 0.9, 1.0, 1.1, 1.2]
    elif axis in ("sideband", "fcl"):
        return [-10.0, -5.0, 0.0, 5.0, 10.0]
    elif axis == "phase":
        return [-0.5, -0.25, 0.0, 0.25, 0.5]
    return None

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_heatmap(num_ms, axis_x, axis_y):
    path = DATA_DIR / f"varms_{num_ms}_heatmap_{axis_x}_{axis_y}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing heatmap: {path}")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scan_kind", "heatmap") != "heatmap":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No heatmap rows found in {path}")
    return rows


def make_grid(rows, axis_x, axis_y):
    """Reconstruct the regular 2D grid from the flat CSV row list."""
    col_x = axis_column(axis_x)
    col_y = axis_column(axis_y)

    scale_x = FREQ_SCALE if axis_x in ("sideband", "fcl") else 1.0
    scale_y = FREQ_SCALE if axis_y in ("sideband", "fcl") else 1.0

    xs     = np.array([float(r[col_x]) * scale_x for r in rows])
    ys     = np.array([float(r[col_y]) * scale_y for r in rows])
    scores = np.array([float(r["score"])    for r in rows])
    xi     = np.array([int(r["x_index"])    for r in rows])
    yi     = np.array([int(r["y_index"])    for r in rows])

    n_x = xi.max()
    n_y = yi.max()

    X = np.empty((n_y, n_x))
    Y = np.empty((n_y, n_x))
    Z = np.empty((n_y, n_x))

    for x, y, s, ix, iy in zip(xs, ys, scores, xi, yi):
        X[iy - 1, ix - 1] = x
        Y[iy - 1, ix - 1] = y
        Z[iy - 1, ix - 1] = s

    return X, Y, Z

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_panel(ax, axis_x, axis_y, num_ms, panel_label):
    rows = read_heatmap(num_ms, axis_x, axis_y)
    X, Y, Z = make_grid(rows, axis_x, axis_y)

    levels = np.linspace(0.0, 1.0, N_LEVELS + 1)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=CMAP)

    xlim = axis_limits(axis_x)
    ylim = axis_limits(axis_y)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    xticks = axis_ticks(axis_x)
    yticks = axis_ticks(axis_y)
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)

    ax.set_xlabel(axis_label(axis_x))
    ax.set_ylabel(axis_label(axis_y))
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.text(0.04, 0.95, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, fontweight="bold", color="white")

    return cf

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    n_panels = len(AXIS_PAIRS)
    
    # 1. Set exact APS width (6.75) and a balanced height (2.5)
    # 2. Reduced wspace to 0.25 since the y-tick numbers are gone, keeping it tight
    fig, all_axes = plt.subplots(
        1, n_panels + 1, 
        figsize=(6.75, 2.4), 
        constrained_layout=False,
        gridspec_kw={'width_ratios': [1, 1, 1, 0.08], 'wspace': 0.25}
    )
    
    # Adjust margins 
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.22)

    axes = all_axes[:3]
    cbar_ax = all_axes[3]

    cf_last = None
    for i, (ax, (axis_x, axis_y), label) in enumerate(zip(axes, AXIS_PAIRS, PANEL_LABELS)):
        try:
            cf_last = draw_panel(ax, axis_x, axis_y, NUM_MS, label)
            
            # Hide y-tick numbers for panels (b) and (c), but KEEP the ylabel
            if i in [1, 2]:
                ax.tick_params(labelleft=False)
                
            # ELEGANT FIX: Shift the extreme labels to prevent corner collisions
            fig.canvas.draw_idle() 
            
            # 1. Fix horizontal overlaps on the x-axis
            x_labels = ax.get_xticklabels()

            # 2. Fix the vertical overlap in the bottom-left corner of panel (a)
            if i == 0:
                y_labels = ax.get_yticklabels()
                if len(y_labels) > 1:
                    y_labels[0].set_verticalalignment('bottom') # Shifts the '-2' UP slightly

            print(f"  Panel {label}: varms_{NUM_MS} {axis_x}/{axis_y} — OK")
        except FileNotFoundError as e:
            print(f"  Skipping {label}: {e}", file=sys.stderr)
            ax.set_visible(False)

    if cf_last is not None:
        cbar = fig.colorbar(cf_last, cax=cbar_ax)
        cbar.set_label(r"$Q$", fontsize=9)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=9)

    out = FIGURE_DIR / f"contour_{NUM_MS}ms.pdf"
    plt.savefig(out)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
