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
# Single column width: 3.375 inches.
mpl.rcParams.update({
    "figure.figsize": (3.375, 6.0),  
    "font.family": "serif",          
    "mathtext.fontset": "cm",        
    "font.size": 12,                 
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,           
    "lines.linewidth": 1.5,          
})

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "backend": "pdf",   
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
SCALE_INFLOOR = 1e-10  
SELECTED_Y_FLOOR = 1e-7
SCALE_DROP_FAILURES = True
SCALE_FAILURE_Q_THRESHOLD = 0.99

MODEL_DASH_EPS_INF = 4.6404956e-7
MODEL_DASH_A = 0.00891668
MODEL_DASH_ALPHA = 0.4947

# Palettes
# Generate 8 colors from magma to access distinct adjacent dark shades
magma_colors = sns.color_palette("magma", n_colors=8).as_hex()

SCALE_GROUPS = [
    {
        "label": r"$r$ = 0.1",
        "color": "#010152",  # Very dark purple (similar to 0, but distinct)
        "ls": ":",                 
        "dir": DATA_DIR / "traces_freqspan10_bound010_NInf_lhs12_restart_fullbudget100_100seeds",
    },
    {
        "label": r"$r$ = 0.5",
        "color": magma_colors[0],  # Absolute darkest (Matches N=Inf exactly)
        "ls": "-",                
        "dir": DATA_DIR / "traces_freqspan10_bound050_NInf_lhs12_restart_fullbudget100_100seeds",
    },
    {
        "label": r"$r$ = 1.0",
        "color": "#000000",  # Dark purple (similar to 0, but distinct)
        "ls": "--",                 
        "dir": DATA_DIR / "traces_freqspan10_bound100_NInf_lhs12_restart_fullbudget100_100seeds",
    },
]

N_LABELS = ["100", "1000", "10000", "100000", "Inf"]
N_COLORS = {
    "100":    magma_colors[6], # Orange
    "1000":   magma_colors[5], # Pink/Orange
    "10000":  magma_colors[4], # Magenta/Purple
    "100000": magma_colors[3], # Medium-dark Purple
    "Inf":    magma_colors[0], # Absolute darkest
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
    return DATA_DIR / f"traces_freqspan10_bound050_full_l1_N{n_label}_nostop100_stream_100seeds"

def model_dash_infid(n_label: str) -> float:
    return MODEL_DASH_EPS_INF + MODEL_DASH_A * float(n_label) ** (-MODEL_DASH_ALPHA)

def quantile_rows(matrix: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(matrix, q, axis=0)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fuzzy_gradient_fill(ax, xs, mat, color, zorder=2, smooth_sigma=None):
    """Layered quantile shading."""
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

def style_log_axis(ax):
    ax.set_xlim(0, MAX_ITER)
    ax.set_ylim(1e-7, 1.0)
    ax.set_yscale("log")
    
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    
    # Setup Locators for major and minor ticks to ensure all exist
    locmaj = ticker.LogLocator(base=10.0, numticks=15)
    ax.yaxis.set_major_locator(locmaj)
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)
    ax.yaxis.set_minor_locator(locmin)
    
    # Custom formatter to only print labels for desired specific exponents
    def custom_fmt(x, pos):
        if x > 0:
            exp = int(np.round(np.log10(x)))
            if exp in [0, -2, -4, -6]:
                return f"$10^{{{exp}}}$"
        return ""
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_fmt))
    
    # Inward ticks for both major and minor
    ax.tick_params(which="both", direction="in", top=True, right=True)
    
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
        q50 = quantile_rows(mat, 0.50)
        c = group["color"]
        ls = group["ls"]

        _fuzzy_gradient_fill(ax, xs, mat, color=c)
        ax.plot(xs, q50, color=c, linestyle=ls, label=group["label"], zorder=4)

        print(f"  {group['label']}: {mat.shape[0]} traces, final median {q50[-1]:.6g}")

    style_log_axis(ax)


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
        q50 = quantile_rows(mat, 0.50)
        c = N_COLORS[n_label]

        _fuzzy_gradient_fill(ax, xs, mat, color=c)
        ax.plot(xs, q50, color=c, label=N_LEGENDS[n_label], zorder=4)

        if n_label != "Inf":
            ref = max(model_dash_infid(n_label), TRACE_INFLOOR)
            dash_refs.append((ref, c))

        print(f"  N={N_LEGENDS[n_label]}: {mat.shape[0]} traces, final median {q50[-1]:.6g}")

    for ref, c in dash_refs:
        ax.axhline(ref, color=c, alpha=0.98, linestyle="--", linewidth=1.5, zorder=1)

    style_log_axis(ax)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Use gridspec_kw to make ax_selected (bottom) taller than ax_scale (top)
    fig, (ax_scale, ax_selected) = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, 1]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.15, right=0.95, top=0.97, bottom=0.08, hspace=0.15)

    print("Panel A — scale trend:")
    draw_scale_trend(ax_scale)
    ax_scale.set_ylabel(r"$1 - Q(\bm{x}_n^*, N=\infty)$")
    ax_scale.set_xlabel("")
    ax_scale.set_xticklabels([])
    ax_scale.legend(loc="upper right", frameon=False, handlelength=1.0, labelspacing=0.3)
    ax_scale.text(0.03, 0.05, r"\textbf{a)}", transform=ax_scale.transAxes, ha="left", va="bottom", fontsize=12)
    ax_scale.set_ylim([1e-7, 1.0])
    ax_scale.set_yticks([1e-6, 1e-4, 1e-2, 1.0])

    print("Panel B — selected-N trend:")
    draw_selected_n_trend(ax_selected)
    ax_selected.set_ylabel(r"$1 - Q(\bm{x}_n^*, N)$")
    ax_selected.set_xlabel(r"$n$")
    ax_selected.legend(loc="upper right", frameon=False, handlelength=1.0, labelspacing=0.3)
    ax_selected.text(0.03, 0.05, r"\textbf{b)}", transform=ax_selected.transAxes, ha="left", va="bottom", fontsize=12)
    ax_selected.set_ylim([1e-7, 1.0])
    ax_selected.set_yticks([1e-6, 1e-4, 1e-2, 1.0])
    
    out = FIGURE_DIR / "figure_ab_vertical.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()