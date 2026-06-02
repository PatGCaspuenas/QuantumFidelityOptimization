"""
Generates figures/paper/figure_ab_vertical.png with two panels:
  Panel A: infidelity convergence curves for three search-box scales (N=Inf).
  Panel B: infidelity convergence curves for five shot counts (full_l1, scale=0.5).
"""

import glob
import csv
import os
import sys

# Ensure Homebrew binaries (dvipng, etc.) are visible to matplotlib
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/Library/TeX/texbin:" + os.environ.get("PATH", "")

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# APS Plot Formatting (PRX Intelligence / PR Applied)
# ---------------------------------------------------------------------------
# Double column width is typically ~6.75 inches.
mpl.rcParams.update({
    "figure.figsize": (6.75, 8.0),   # Double-column width, proportional height
    "font.family": "serif",          # Fallback to serif
    "mathtext.fontset": "cm",        # Computer Modern for math
    "font.size": 12,                 # Base size 12
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,           # Legend size 10
    "lines.linewidth": 1.5,          # Line width 1.5
})

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "backend": "pdf",   # use pdf backend — avoids dvipng, uses pdflatex directly
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
print(f"Repo root: {REPO_ROOT}")
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT 

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ITER = 100
TRACE_INFLOOR = 1e-10
SCALE_INFLOOR = 1e-5
SELECTED_Y_FLOOR = 1e-7
SCALE_DROP_FAILURES = True
SCALE_FAILURE_Q_THRESHOLD = 0.99

MODEL_DASH_EPS_INF = 4.6404956e-7
MODEL_DASH_A = 0.00891668
MODEL_DASH_ALPHA = 0.4947

# Palettes
flare_colors = sns.color_palette("flare", n_colors=3).as_hex()
husl_colors = sns.color_palette("husl", n_colors=4).as_hex()

SCALE_GROUPS = [
    {
        "label": "scale = 0.1",
        "color": flare_colors[0],
        "dir": DATA_DIR / "traces_freqspan10_bound010_NInf_lhs12_restart_fullbudget_40seeds",
    },
    {
        "label": "scale = 0.5",
        "color": flare_colors[1],
        "dir": DATA_DIR / "traces_freqspan10_bound050_NInf_lhs12_restart_fullbudget100_40seeds",
    },
    {
        "label": "scale = 1.0",
        "color": flare_colors[2],
        "dir": DATA_DIR / "traces_freqspan10_bound100_NInf_lhs12_restart_fullbudget100_40seeds",
    },
]

N_LABELS = ["100", "1000", "10000", "100000", "Inf"]
N_COLORS = {
    "100":    husl_colors[0],
    "1000":   husl_colors[1],
    "10000":  husl_colors[2],
    "100000": husl_colors[3],
    "Inf":    "#000000",  # Black for deterministic
}
N_LEGENDS = {
    "100":    r"$N = 10^2$",
    "1000":   r"$N = 10^3$",
    "10000":  r"$N = 10^4$",
    "100000": r"$N = 10^5$",
    "Inf":    r"$N = \infty$",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def trace_paths(directory: Path):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing trace directory: {directory}")
    paths = sorted(glob.glob(str(directory / "trace_seed*_ucb.csv")))
    if not paths:
        raise FileNotFoundError(f"No trace CSVs found in {directory}")
    return paths

def read_trace(path: str):
    iters, qdets = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(float(row["iter"])))
            qdets.append(float(np.clip(float(row["q_det_rec"]), 0.0, 1.0)))
    return {"iter": iters, "qdet": qdets}

def read_trace_group(directory: Path, drop_failures: bool = False):
    traces = [read_trace(p) for p in trace_paths(directory)]
    if drop_failures:
        traces = [t for t in traces if t["qdet"][-1] >= SCALE_FAILURE_Q_THRESHOLD]
        if not traces:
            raise ValueError(f"No traces left after failure filtering in {directory}")
    return traces

def forward_fill_infid(trace, max_iter: int = MAX_ITER, floor: float = TRACE_INFLOOR):
    values = np.empty(max_iter)
    cursor = 0
    last_q = trace["qdet"][0]
    for i, it in enumerate(range(1, max_iter + 1)):
        while cursor < len(trace["iter"]) and trace["iter"][cursor] <= it:
            last_q = trace["qdet"][cursor]
            cursor += 1
        values[i] = max(1.0 - last_q, floor)
    return values

def trace_matrix(traces, floor: float = TRACE_INFLOOR) -> np.ndarray:
    return np.vstack([forward_fill_infid(t, floor=floor) for t in traces])

def full_l1_trace_dir(n_label: str) -> Path:
    return DATA_DIR / f"traces_freqspan10_bound050_full_l1_N{n_label}_nostop100_stream_40seeds"

def model_dash_infid(n_label: str) -> float:
    return MODEL_DASH_EPS_INF + MODEL_DASH_A * float(n_label) ** (-MODEL_DASH_ALPHA)

def quantile_rows(matrix: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(matrix, q, axis=0)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fuzzy_gradient_fill(ax, xs, mat, color, zorder=2, smooth_sigma=None):
    """Layered quantile shading.
    smooth_sigma=None  → raw empirical quantiles (no smoothing)
    smooth_sigma=N     → Gaussian-smoothed bands (sigma=N iterations)
    """
    n_layers   = 50
    quantiles  = np.linspace(0.05, 0.45, n_layers)
    base_alpha = 0.5 / n_layers
    for q in quantiles:
        lower = np.nanquantile(mat, q,       axis=0)
        upper = np.nanquantile(mat, 1.0 - q, axis=0)
        if smooth_sigma is not None:
            lower = gaussian_filter1d(lower, sigma=smooth_sigma)
            upper = gaussian_filter1d(upper, sigma=smooth_sigma)
        ax.fill_between(xs, lower, upper, color=color, alpha=base_alpha,
                        linewidth=0, edgecolor="none", zorder=zorder)

def style_log_axis(ax, y_floor: float):
    min_exp = int(np.floor(np.log10(y_floor)))
    exponents = list(range(0, min_exp - 1, -1))
    yticks = [10.0 ** e for e in exponents]
    ylabels = [f"$10^{{{e}}}$" for e in exponents]
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    
    # Inward ticks for both major and minor
    ax.tick_params(which="both", direction="in", top=True, right=True)
    
    ax.set_ylim(y_floor, 1.0)
    ax.set_xlim(0, MAX_ITER)
    ax.set_xlabel(r"$n$")
    
    # Updated clearer label
    ax.set_ylabel(r"$1 - Q(\bm{x}_n^*, N=\infty)$")
    
    # Light gray dotted gridlines
    ax.grid(True, which="both", color="lightgray", linestyle=":", linewidth=0.5, alpha=0.7)


def draw_scale_trend(ax):
    xs = np.arange(1, MAX_ITER + 1)
    for group in SCALE_GROUPS:
        try:
            traces = read_trace_group(group["dir"], drop_failures=SCALE_DROP_FAILURES)
        except FileNotFoundError as e:
            print(f"  Skipping scale group '{group['label']}': {e}", file=sys.stderr)
            continue

        mat = trace_matrix(traces, floor=SCALE_INFLOOR)
        q05 = quantile_rows(mat, 0.05)
        q50 = quantile_rows(mat, 0.50)
        q95 = quantile_rows(mat, 0.95)
        c = group["color"]

        _fuzzy_gradient_fill(ax, xs, mat, color=c)
        #ax.plot(xs, q05, color=c, lw=0.8, ls="--", alpha=0.5, zorder=3)
        #ax.plot(xs, q95, color=c, lw=0.8, ls="--", alpha=0.5, zorder=3)
        ax.plot(xs, q50, color=c, label=group["label"], zorder=4)

        print(f"  {group['label']}: {mat.shape[0]} traces, final median {q50[-1]:.6g}")

    style_log_axis(ax, SCALE_INFLOOR)


def draw_selected_n_trend(ax):
    xs = np.arange(1, MAX_ITER + 1)
    dash_refs = []

    for n_label in N_LABELS:
        try:
            traces = read_trace_group(full_l1_trace_dir(n_label))
        except FileNotFoundError as e:
            print(f"  Skipping N={n_label}: {e}", file=sys.stderr)
            continue

        mat = trace_matrix(traces, floor=TRACE_INFLOOR)
        q05 = quantile_rows(mat, 0.05)
        q50 = quantile_rows(mat, 0.50)
        q95 = quantile_rows(mat, 0.95)
        c = N_COLORS[n_label]

        _fuzzy_gradient_fill(ax, xs, mat, color=c)
        #ax.plot(xs, q05, color=c, lw=0.8, ls="--", alpha=0.5, zorder=3)
        #ax.plot(xs, q95, color=c, lw=0.8, ls="--", alpha=0.5, zorder=3)
        ax.plot(xs, q50, color=c, label=N_LEGENDS[n_label], zorder=4)

        if n_label != "Inf":
            ref = max(model_dash_infid(n_label), TRACE_INFLOOR)
            dash_refs.append((ref, c))

        print(f"  N={N_LEGENDS[n_label]}: {mat.shape[0]} traces, final median {q50[-1]:.6g}")

    for ref, c in dash_refs:
        ax.axhline(ref, color=c, alpha=0.98, linestyle="--", linewidth=1.5, zorder=1)

    style_log_axis(ax, SELECTED_Y_FLOOR)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_scale, ax_selected) = plt.subplots(
        2, 1,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.15, right=0.95, top=0.97, bottom=0.08, hspace=0.15)

    print("Panel A — scale trend:")
    draw_scale_trend(ax_scale)
    ax_scale.set_xlabel("")
    ax_scale.set_xticklabels([])
    ax_scale.set_yscale("log")
    ax_scale.legend(loc="upper right", frameon=False, handlelength=2.0, labelspacing=0.3)

    print("Panel B — selected-N trend:")
    draw_selected_n_trend(ax_selected)
    ax_selected.set_yscale("log")
    ax_selected.legend(loc="upper right", frameon=False, handlelength=2.0, labelspacing=0.3)
    
    out = FIGURE_DIR / "figure_ab_vertical.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()