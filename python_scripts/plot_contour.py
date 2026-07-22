"""
Generates figures/paper/contour_2ms.pdf

2D parameter landscape heatmaps for the two-MS_(pi/2) calibration sequence.
Reads: data/varms_2_heatmap_{axis_x}_{axis_y}.csv
Score: Q_infinity = P_ee (probability of |11> state, deterministic/no shot noise)

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
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT / "figures"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MS = int(os.environ.get("PLOT_NUM_MS", "2"))
N_LEVELS = 20
CMAP = "magma"

# Three axis pairs: (x_axis, y_axis) — left to right
# (a) x=Δω_cl, y=Δδ   (b) x=Δδ, y=Ω   (c) x=Δω_cl, y=Ω
AXIS_PAIRS = [
    ("fcl",      "sideband"),   # (a)
    ("sideband", "rabi"),       # (b)
    ("fcl",      "rabi"),       # (c)
]

PANEL_LABELS = [r"\textbf{a)}", r"\textbf{b)}", r"\textbf{c)}"]

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
        "rabi":     r"$\Omega / \Omega^*$",
        "sideband": r"$\Delta \delta\ (2\pi \cdot \mathrm{kHz})$",
        "fcl":      r"$\Delta \omega_\mathrm{cl}\ (2\pi \cdot \mathrm{kHz})$",
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
    """Load heatmap rows; tries both (axis_x, axis_y) and reversed file names."""
    for a, b in [(axis_x, axis_y), (axis_y, axis_x)]:
        path = DATA_DIR / f"varms_{num_ms}_heatmap_{a}_{b}.csv"
        if path.exists():
            rows = []
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("scan_kind", "heatmap") == "heatmap":
                        rows.append(row)
            if rows:
                return rows
    raise FileNotFoundError(
        f"Missing heatmap for {axis_x}/{axis_y} (tried both orderings)")


def make_grid(rows, axis_x, axis_y):
    """Build a regular 2D grid with axis_x on the x-axis and axis_y on the y-axis.
    Works regardless of which file ordering was found."""
    col_x  = axis_column(axis_x)
    col_y  = axis_column(axis_y)
    scale_x = FREQ_SCALE if axis_x in ("sideband", "fcl") else 1.0
    scale_y = FREQ_SCALE if axis_y in ("sideband", "fcl") else 1.0

    xs     = np.array([float(r[col_x]) * scale_x for r in rows])
    ys     = np.array([float(r[col_y]) * scale_y for r in rows])
    scores = np.array([float(r["score"])          for r in rows])

    x_uniq = np.sort(np.unique(xs))
    y_uniq = np.sort(np.unique(ys))

    Z = np.full((len(y_uniq), len(x_uniq)), np.nan)
    for x, y, s in zip(xs, ys, scores):
        ix = np.searchsorted(x_uniq, x)
        iy = np.searchsorted(y_uniq, y)
        Z[iy, ix] = s

    X, Y = np.meshgrid(x_uniq, y_uniq)
    return X, Y, Z

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_panel(ax, axis_x, axis_y, num_ms):
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

    return cf

def build_figure():
    fig = plt.figure(figsize=(7.4, 2.4))

    # Manual axes positions [x0, y0, width, height] in figure fraction.
    # All three panels share identical width W so the layout is symmetric.
    y0, h = 0.22, 0.70   # bottom / height
    W     = 0.21          # panel width (figure fraction)
    # x0 positions:  (a) at 0.11 | gap 0.08 | (b) at 0.40 | gap 0.05 | (c) at 0.66
    # colorbar tight to (c): gap 0.01, width W*0.08
    ax_a    = fig.add_axes([0.11, y0, W,        h])
    ax_b    = fig.add_axes([0.40, y0, W,        h])
    ax_c    = fig.add_axes([0.66, y0, W,        h])
    cbar_ax = fig.add_axes([0.88, y0, W * 0.08, h])
    axes    = [ax_a, ax_b, ax_c]

    cf_last = None
    for i, (ax, (axis_x, axis_y), label) in enumerate(zip(axes, AXIS_PAIRS, PANEL_LABELS)):
        try:
            cf_last = draw_panel(ax, axis_x, axis_y, NUM_MS)

            # Hide y-tick labels only for panel (c); show for (a) and (b)
            if i == 2:
                ax.tick_params(labelleft=False)

            # Panel letter outside-left; panels with ytick labels need more offset
            x_off = -0.22 if i < 2 else -0.08
            ax.text(x_off, 1.0, label,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, fontweight="bold", color="black", clip_on=False)

            print(f"  Panel {label}: varms_{NUM_MS} {axis_x}/{axis_y} — OK")
        except FileNotFoundError as e:
            print(f"  Skipping {label}: {e}", file=sys.stderr)
            ax.set_visible(False)

    if cf_last is not None:
        cbar = fig.colorbar(cf_last, cax=cbar_ax)
        cbar.set_label(r"$Q$", fontsize=9)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=9)

    return fig

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out = FIGURE_DIR / f"contour_{NUM_MS}ms.pdf"
    fig.savefig(out)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()

