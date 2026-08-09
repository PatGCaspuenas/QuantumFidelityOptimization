"""Panel-b style convergence figure for the hardware/emulator BO sweep.

Reads the pickled sweep DataFrame produced by UW_OutputCode_20260727.ipynb
(`save_results` -> results/bo_sweep_updated.pkl) and plots, for each shot
budget N, the median infidelity 1 - Q(x_n^*) of the recommended point versus
BO iteration, with layered quantile shading across the matched-seed trials.

Provenance of the plotted quantity
----------------------------------
`X_opt[:, n]` (src/bayes_hetero_opt.jl on branch SANDIA_v0, `HeteroBOResult`)
is the GP-*recommended* optimum at iteration n — the argmax of the posterior
mean from `recommend_mean`, not the UCB acquisition point (that one lives in
`X` / `i_acq`).  With `fidelity_threshold = nothing` the optimizer measures the
recommendation on hardware only once, after the loop (-> `y_last_opt`, the
`score` column), so the per-iteration hardware trace comes instead from the
notebook's `record_run`, which re-queries every `X_opt[:, n]` at N shots and
stores the result in `recommended_populations` (n_iter x 4).  That is what is
plotted here, mapped through the same `full_l1` score as Julia's
`score_from_probs`.

Because each point is an N-shot estimate, 1 - Q is quantized at 1/N and lands
exactly on zero for a small fraction of iterations (2.6% at N=50, 0.3% at
N=100 in the current sweep).  Those are measurement-limited, not true zeros,
and are pinned to the 1/(2N) sampling-resolution floor drawn as a dashed line.

Usage:
    python "python plots/plot_bo_sweep_N_traces.py" [path/to/bo_sweep_updated.pkl]
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import seaborn as sns
except ModuleNotFoundError:  # seaborn is optional — fall back to matplotlib
    sns = None

# ---------------------------------------------------------------------------
# APS Plot Formatting (PRX Intelligence / PR Applied)
# ---------------------------------------------------------------------------
# Single column width: 3.375 inches.
mpl.rcParams.update({
    "figure.figsize": (3.375, 2.4),
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
})

# Ensure Homebrew / MacTeX binaries (dvipng, latex) are visible to matplotlib.
for _bin in ("/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin"):
    if _bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _bin

# usetex needs a working latex + dvipng; fall back to mathtext when absent.
USE_TEX = all(shutil.which(tool) for tool in ("latex", "dvipng"))
if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}\usepackage{xcolor}",
    })
else:
    print("latex/dvipng not found — falling back to mathtext rendering.",
          file=sys.stderr)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return start

REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
FIGURE_DIR = REPO_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PICKLES = [
    REPO_ROOT / "results" / "bo_sweep_updated.pkl",
    REPO_ROOT / "results" / "bo_sweep_N_traces.pkl",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ITER = 75            # N_ITER in the sweep notebook
SCORE_MODE = "full_l1"   # matches SCORE_MODE in the notebook
TARGET_POPULATION_INDEX = 3

# Infidelity floor: a finite-shot estimate cannot resolve better than the
# sampling resolution 1/N, so exact 1 - Q = 0 is pinned at half a shot.
FLOOR_SHOTS_FRACTION = 0.5
ABS_INFLOOR = 1e-4

Y_LIMITS = (4e-3, 1.0)
Y_TICKS = [1e-2, 1e-1, 1.0]
Y_TICK_EXPONENTS = [0, -1, -2]

# Palette: same magma ramp as the trace figure (darker = more shots).
def _magma_hexes(n_colors: int = 8):
    if sns is not None:
        return sns.color_palette("magma", n_colors=n_colors).as_hex()
    # Same sampling seaborn uses for continuous cmaps: trim both extremes.
    cmap = mpl.colormaps["magma"]
    stops = np.linspace(0, 1, n_colors + 2)[1:-1]
    return [mpl.colors.to_hex(cmap(s)) for s in stops]

magma_colors = _magma_hexes(8)
# Light (few shots) -> dark (many shots); spread evenly over the ramp so that
# a two-N sweep still gets two clearly distinguishable colours.
MAGMA_RAMP_INDICES = (6, 5, 4, 3, 1, 0)


def n_colors(n_values):
    k = len(n_values)
    if k == 1:
        picks = [MAGMA_RAMP_INDICES[2]]
    else:
        span = np.linspace(0, len(MAGMA_RAMP_INDICES) - 1, k)
        picks = [MAGMA_RAMP_INDICES[int(round(s))] for s in span]
    return {n: magma_colors[p] for n, p in zip(n_values, picks)}

def n_legend(n: int) -> str:
    exponent = np.log10(n)
    if np.isclose(exponent, round(exponent)):
        return rf"$10^{{{int(round(exponent))}}}$"
    return rf"${n}$"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def resolve_pickle(argv) -> Path:
    if len(argv) > 1:
        path = Path(argv[1]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing sweep pickle: {path}")
        return path
    for candidate in DEFAULT_PICKLES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find bo_sweep_updated.pkl; pass its path as an argument."
    )


def score_from_populations(probs: np.ndarray) -> np.ndarray:
    """Vectorised copy of the notebook's score_from_populations.

    `probs` is (n_iter, 4) in population order p00, p01, p10, p11.
    """
    probs = np.asarray(probs, dtype=float)
    if SCORE_MODE == "p11":
        return np.clip(probs[:, TARGET_POPULATION_INDEX], 0.0, 1.0)
    l1 = (np.abs(probs[:, 0]) + np.abs(probs[:, 1])
          + np.abs(probs[:, 2]) + np.abs(probs[:, 3] - 1.0))
    return np.clip(1.0 - 0.5 * l1, 0.0, 1.0)


def infid_trace(probs: np.ndarray, floor: float, max_iter: int = MAX_ITER) -> np.ndarray:
    """Infidelity of the recommended point per iteration, padded/truncated."""
    values = np.maximum(1.0 - score_from_populations(probs), floor)
    if len(values) >= max_iter:
        return values[:max_iter]
    return np.concatenate([values, np.full(max_iter - len(values), values[-1])])


def infid_matrix(group: pd.DataFrame, n_shots: int) -> np.ndarray:
    floor = max(FLOOR_SHOTS_FRACTION / n_shots, ABS_INFLOOR)
    return np.vstack([
        infid_trace(row, floor) for row in group["recommended_populations"]
    ])


def quantile_rows(matrix: np.ndarray, q: float) -> np.ndarray:
    return np.nanquantile(matrix, q, axis=0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fuzzy_gradient_fill(ax, xs, mat, color, zorder=2):
    """Layered quantile shading."""
    n_layers = 50
    quantiles = np.linspace(0.05, 0.45, n_layers)
    base_alpha = 0.5 / n_layers
    for q in quantiles:
        lower = np.nanquantile(mat, q, axis=0)
        upper = np.nanquantile(mat, 1.0 - q, axis=0)
        ax.fill_between(xs, lower, upper, color=color, alpha=base_alpha,
                        linewidth=0, edgecolor="none", zorder=zorder)


def style_log_axis(ax):
    ax.set_xlim(0, MAX_ITER)
    ax.set_ylim(*Y_LIMITS)
    ax.set_yscale("log")

    xticks = [t for t in (0, 15, 30, 45, 60, 75) if t <= MAX_ITER]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)
    )

    def custom_fmt(x, pos):
        if x > 0:
            exp = int(np.round(np.log10(x)))
            if exp in Y_TICK_EXPONENTS:
                return f"$10^{{{exp}}}$"
        return ""

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_fmt))

    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="both", color="lightgray", linestyle=":",
            linewidth=0.5, alpha=0.7)


def draw_n_trend(ax, df: pd.DataFrame):
    xs = np.arange(1, MAX_ITER + 1)
    n_values = sorted(int(n) for n in df["N"].unique())
    colors = n_colors(n_values)

    handles, labels = [], []
    for n in n_values:
        group = df[df["N"] == n]
        mat = infid_matrix(group, n)
        q50 = quantile_rows(mat, 0.50)
        c = colors[n]

        _fuzzy_gradient_fill(ax, xs, mat, color=c)
        line, = ax.plot(xs, q50, color=c, zorder=4)
        handles.append(line)
        labels.append(n_legend(n))

        # Sampling-resolution floor for this shot budget.
        ax.axhline(max(FLOOR_SHOTS_FRACTION / n, ABS_INFLOOR), color=c, alpha=0.98,
                   linestyle="--", linewidth=1.5, zorder=1)

        print(f"  N={n}: {mat.shape[0]} trials, "
              f"final median {q50[-1]:.6g}, best {np.nanmin(mat[:, -1]):.6g}")

    style_log_axis(ax)
    return handles, labels


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

def build_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(constrained_layout=False)
    fig.subplots_adjust(left=0.25, right=0.85, top=0.95, bottom=0.19)

    print("Panel — shot-budget trend:")
    handles, labels = draw_n_trend(ax, df)

    ax.set_ylabel(r"$1 - Q(\bm{x}_n^*)$" if USE_TEX else r"$1 - Q(x_n^*)$")
    ax.set_xlabel(r"$n$")
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks(Y_TICKS)

    leg = ax.legend(handles, labels, loc="center right", bbox_to_anchor=(0.99, 0.82),
                    ncol=1, frameon=True, edgecolor="black", facecolor="white",
                    fancybox=False, handlelength=1.5, labelspacing=0.2)
    add_side_title(ax, leg, r"$N$")

    return fig


def show_and_save(fig, filename):
    out = FIGURE_DIR / filename
    fig.savefig(out)
    print(f"Saved -> {out}")


def main(argv=None):
    argv = sys.argv if argv is None else argv
    pickle_path = resolve_pickle(argv)
    print(f"Reading {pickle_path}")
    df = pd.read_pickle(pickle_path)

    metadata = df.attrs.get("metadata", {})
    if metadata:
        globals()["SCORE_MODE"] = metadata.get("score_mode", SCORE_MODE)
        globals()["MAX_ITER"] = int(metadata.get("n_iter", MAX_ITER))
        print(f"  score_mode={SCORE_MODE}, n_iter={MAX_ITER}, "
              f"N_LIST={metadata.get('N_LIST')}, trials={len(df)}")

    fig = build_figure(df)
    show_and_save(fig, "figure_bo_sweep_N.pdf")
    return fig


if __name__ == "__main__":
    main()
