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
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# APS Plot Formatting (PRX Intelligence / PR Applied)
# ---------------------------------------------------------------------------
# Single column width: 3.375 inches.
mpl.rcParams.update({
    "figure.figsize": (3.375, 4.5),  
    "font.family": "serif",          
    "mathtext.fontset": "cm",        
    "font.size": 10,                 
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,           
    "lines.linewidth": 1.5,          
})

# IMPORTANT: Added \usepackage{xcolor} so we can use \textcolor{white}
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}\usepackage{xcolor}",
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
        "label": "0.1",
        "color": "#648FFF",        # Aesthetic vibrant light blue
        "ls": ":",                 
        "dir": DATA_DIR / "traces_freqspan10_bound010_NInf_lhs12_restart_fullbudget100_100seeds",
    },
    {
        "label": "0.5",
        "color": magma_colors[0],  # Absolute darkest (Matches N=Inf exactly)
        "ls": "-",                
        "dir": DATA_DIR / "traces_freqspan10_bound050_NInf_lhs12_restart_fullbudget100_100seeds",
    },
    {
        "label": "1.0",
        "color": "#005AB5",        # Aesthetic rich deep blue
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
    "100":    r"$10^2$",
    "1000":   r"$10^3$",
    "10000":  r"$10^4$",
    "100000": r"$10^5$",
    "Inf":    r"$\infty$",
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
    
    locmaj = ticker.LogLocator(base=10.0, numticks=15)
    ax.yaxis.set_major_locator(locmaj)
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)
    ax.yaxis.set_minor_locator(locmin)
    
    def custom_fmt(x, pos):
        if x > 0:
            exp = int(np.round(np.log10(x)))
            if exp in [0, -2, -4, -6]:
                return f"$10^{{{exp}}}$"
        return ""
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_fmt))
    
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="both", color="lightgray", linestyle=":", linewidth=0.5, alpha=0.7)


def draw_scale_trend(ax):
    xs = np.arange(1, MAX_ITER + 1)
    handles = {}
    
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
        line, = ax.plot(xs, q50, color=c, linestyle=ls, zorder=4)
        handles[group["label"]] = line

        print(f"  {group['label']}: {mat.shape[0]} traces, final median {q50[-1]:.6g}")

    style_log_axis(ax)
    return handles


def draw_selected_n_trend(ax):
    xs = np.arange(1, MAX_ITER + 1)
    dash_refs = []
    handles = {}

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
        line, = ax.plot(xs, q50, color=c, zorder=4)
        handles[n_label] = line

        if n_label != "Inf":
            ref = max(model_dash_infid(n_label), TRACE_INFLOOR)
            dash_refs.append((ref, c))

        print(f"  N={N_LEGENDS[n_label]}: {mat.shape[0]} traces, final median {q50[-1]:.6g}")

    for ref, c in dash_refs:
        ax.axhline(ref, color=c, alpha=0.98, linestyle="--", linewidth=1.5, zorder=1)

    style_log_axis(ax)
    return handles

def add_side_title(ax, leg, text):
    """Dynamically places a text title to the immediate left of the legend box."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bbox.x0 - 0.03, bbox.y0 + (bbox.height / 2), text, 
            transform=ax.transAxes, ha="right", va="center", fontsize=10)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_scale, ax_selected) = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, 1]},
        constrained_layout=False,
    )
    
    fig.subplots_adjust(left=0.25, right=0.85, top=0.95, bottom=0.1, hspace=0.1)

    # -----------------------------------------------------------------------
    # PANEL A: SCALE TREND
    # -----------------------------------------------------------------------
    print("Panel A — scale trend:")
    h_a_dict = draw_scale_trend(ax_scale)
    
    handles_a = []
    labels_a = []
    for g in SCALE_GROUPS:
        if g["label"] in h_a_dict:
            handles_a.append(h_a_dict[g["label"]])
            labels_a.append(g["label"])

    ax_scale.set_ylabel(r"$1 - Q_{det}(\bm{x}_n^*)$")
    ax_scale.set_xlabel("")
    ax_scale.set_xticklabels([])
    ax_scale.text(0.03, 0.05, r"\textbf{a)}", transform=ax_scale.transAxes, ha="left", va="bottom", fontsize=9)
    ax_scale.set_ylim([1e-7, 1.0])
    ax_scale.set_yticks([1e-6, 1e-4, 1e-2, 1.0])

    # Legend A 
    leg_a = ax_scale.legend(handles_a, labels_a, loc="upper right", bbox_to_anchor=(0.99, 0.99), 
                            frameon=True, edgecolor="black", facecolor="white", 
                            fancybox=False, handlelength=1.5, labelspacing=0.2)
    add_side_title(ax_scale, leg_a, r"$r$")


    # -----------------------------------------------------------------------
    # PANEL B: SELECTED N TREND
    # -----------------------------------------------------------------------
    print("Panel B — selected-N trend:")
    h_b_dict = draw_selected_n_trend(ax_selected)
    
    # White line dummy handle to match the spacing flawlessly
    dummy_handle = Line2D([], [], color='white', linestyle='-', linewidth=1.5)
    
    # 1. Master Legend
    handles_b1 = [
        h_b_dict["100"], h_b_dict["1000"], h_b_dict["10000"],
        dummy_handle, dummy_handle, dummy_handle,
    ]
    
    # Put an actual string in the 4th slot so Matplotlib perfectly calculates the width
    labels_b1 = [
        r"$10^2$",
        r"$10^3$",
        r"$10^4$",
        r"$10^4$",   # <-- We will make this invisible below
        "",
        "",
    ]

    ax_selected.set_ylabel(r"$1 - Q_{det}(\bm{x}_n^*)$")
    ax_selected.set_xlabel(r"$n$")
    ax_selected.text(0.03, 0.05, r"\textbf{b)}", transform=ax_selected.transAxes, ha="left", va="bottom", fontsize=9)
    ax_selected.set_ylim([1e-7, 1.0])
    ax_selected.set_yticks([1e-6, 1e-4, 1e-2, 1.0])
    
    # Draw the box and the 3 items on the left
    leg_b1 = ax_selected.legend(handles_b1, labels_b1, loc="center right", bbox_to_anchor=(0.99, 0.82),
                                ncol=2, frameon=True, edgecolor="black", facecolor="white",
                                fancybox=False, handlelength=1.5, labelspacing=0.2, columnspacing=0.5)
    ax_selected.add_artist(leg_b1)
    
    # --- THE MAGIC BULLET ---
    # Grab the text elements in the legend and make the 4th one (index 3) completely transparent
    leg_b1.get_texts()[3].set_alpha(0)
    
    # 2. Slave Legend (Vertically centered in the dummy space)
    handles_b2 = [h_b_dict["100000"], h_b_dict["Inf"]]
    labels_b2 = [r"$10^5$", r"$\infty$"]
    
    # Shifted slightly left so it perfectly overlays the transparent text
    leg_b2 = ax_selected.legend(handles_b2, labels_b2, loc="center right", bbox_to_anchor=(0.99, 0.82),
                                ncol=1, frameon=False, 
                                handlelength=1.5, labelspacing=0.4)
    
    # Add the N title outside the master box
    add_side_title(ax_selected, leg_b1, r"$N$")

    # Output
    out = FIGURE_DIR / "figure_ab_vertical.pdf"
    plt.savefig(out)
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()