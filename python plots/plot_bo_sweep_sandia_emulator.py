"""Panel-b style convergence figure for the SANDIA emulator BO sweep.

Counterpart of plot_bo_sweep_N_traces.py, for the pickle produced by
SANDIA_code1_modified.ipynb with `emulator = True`
(results/bo_sweep_sandia_emulator.pkl).

Provenance of the plotted quantity
----------------------------------
`X_opt[:, n]` (src/bayes_hetero_opt.jl, `ThresholdBOResult`) is the GP-recommended
optimum at iteration n — the argmax of the posterior mean from `recommend_mean`
(random scan + bounded L-BFGS refinement), not the UCB acquisition point.  The
notebook's `record_run` re-queries every `X_opt[:, n]`, so the trace below is the
optimum as evaluated by the backend at each iteration.

The score is P(|11>) — the same quantity `Q_varMS` optimises on the main branch.

Two traces are recorded and `--trace=` selects between them:

  exact   (default) `recommended_populations_exact` — the noiseless emulator
          evaluation at each recommended point, i.e. the emulator analogue of
          q_det_rec in the paper figure. Use this to see convergence.
  sampled `recommended_populations` — the N-shot measurement the optimizer
          actually saw. Being a binomial proportion it is quantised at 1/N, and
          since the emulator optimum sits at 1 - p11 ~ 5e-4 most measurements
          return exactly 1.0; those are pinned to the 1/(2N) sampling-resolution
          floor drawn as a dashed line. Informative about measurement limits,
          not about convergence.

Emulator caveat: SNLToy1 is driven by amp_ia and zeta only, so sideband_offset
and carrier_offset are exactly flat directions in emulator mode. The search is
effectively 2D here, and the traces are NOT directly comparable to a 4D hardware
run.

Usage:
    python "python plots/plot_bo_sweep_sandia_emulator.py" [results.pkl] [--trace=exact|sampled]
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

USE_TEX = all(shutil.which(tool) for tool in ("latex", "dvipng"))
if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}\usepackage{xcolor}",
    })
else:
    print("latex/dvipng not found — falling back to mathtext rendering.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    # Marker must exist on every branch. `scripts/` does not: it is absent on
    # SANDIA_v0, which would make this fall back to the script's own directory
    # and then look for results/ inside "python plots/".
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "Project.toml").is_file():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start

REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
FIGURE_DIR = REPO_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PICKLES = [
    REPO_ROOT / "results" / "bo_sweep_sandia_emulator.pkl",
    REPO_ROOT / "results" / "bo_sweep_sandia.pkl",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ITER = 75                    # overridden from the pickle's metadata
TARGET_POPULATION_INDEX = 3      # p11 in population order p00, p01, p10, p11

TRACE = "exact"                  # "exact" | "sampled"; see the module docstring

FLOOR_SHOTS_FRACTION = 0.5
ABS_INFLOOR = 1e-4
EXACT_INFLOOR = 1e-6

Y_LIMITS_EXACT = (1e-4, 1.0)
Y_TICKS_EXACT = [1e-4, 1e-2, 1.0]
Y_EXPONENTS_EXACT = [0, -2, -4]

Y_LIMITS_SAMPLED = (4e-3, 1.0)
Y_TICKS_SAMPLED = [1e-2, 1e-1, 1.0]
Y_EXPONENTS_SAMPLED = [0, -1, -2]

SHOW_MU_OPT = False              # set True to overlay 1 - mu_opt, the GP's own belief
                                 # at its recommendation (dotted, no shot noise)


def _y_style():
    return ((Y_LIMITS_EXACT, Y_TICKS_EXACT, Y_EXPONENTS_EXACT) if TRACE == "exact"
            else (Y_LIMITS_SAMPLED, Y_TICKS_SAMPLED, Y_EXPONENTS_SAMPLED))


def _magma_hexes(n_colors: int = 8):
    if sns is not None:
        return sns.color_palette("magma", n_colors=n_colors).as_hex()
    # Same sampling seaborn uses for continuous cmaps: trim both extremes.
    cmap = mpl.colormaps["magma"] if hasattr(mpl, "colormaps") else plt.get_cmap("magma")
    stops = np.linspace(0, 1, n_colors + 2)[1:-1]
    return [mpl.colors.to_hex(cmap(s)) for s in stops]

magma_colors = _magma_hexes(8)
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
        return r"$10^{%d}$" % int(round(exponent))
    return r"$%d$" % n


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
    searched = "\n".join(f"  {c}" for c in DEFAULT_PICKLES)
    raise FileNotFoundError(
        f"Could not find the SANDIA sweep pickle. Looked in (repo root = {REPO_ROOT}):\n"
        f"{searched}\nPass the path as an argument instead."
    )


def score_from_populations(probs) -> np.ndarray:
    """Calibration score = P(|11>), for a (n_iter, 4) population array."""
    probs = np.asarray(probs, dtype=float)
    return np.clip(probs[:, TARGET_POPULATION_INDEX], 0.0, 1.0)


def infid_trace(probs, floor: float, max_iter: int) -> np.ndarray:
    values = np.maximum(1.0 - score_from_populations(probs), floor)
    if len(values) >= max_iter:
        return values[:max_iter]
    return np.concatenate([values, np.full(max_iter - len(values), values[-1])])


def infid_matrix(group: pd.DataFrame, n_shots: int) -> np.ndarray:
    if TRACE == "exact":
        if ("recommended_populations_exact" not in group
                or group["recommended_populations_exact"].isna().all()):
            raise KeyError(
                "This pickle has no `recommended_populations_exact` column "
                "(hardware run, or produced before that column existed). "
                "Re-run with --trace=sampled.")
        column, floor = "recommended_populations_exact", EXACT_INFLOOR
    else:
        column = "recommended_populations"
        floor = max(FLOOR_SHOTS_FRACTION / n_shots, ABS_INFLOOR)
    return np.vstack([infid_trace(row, floor, MAX_ITER) for row in group[column]])


def mu_infid_matrix(group: pd.DataFrame, n_shots: int) -> np.ndarray:
    """1 - mu_opt: the GP's posterior mean at its own recommendation (no shot noise)."""
    floor = EXACT_INFLOOR if TRACE == "exact" else max(FLOOR_SHOTS_FRACTION / n_shots,
                                                       ABS_INFLOOR)
    rows = []
    for mu in group["mu_opt"]:
        v = np.maximum(1.0 - np.clip(np.asarray(mu, dtype=float), 0.0, 1.0), floor)
        if len(v) < MAX_ITER:
            v = np.concatenate([v, np.full(MAX_ITER - len(v), v[-1])])
        rows.append(v[:MAX_ITER])
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fuzzy_gradient_fill(ax, xs, mat, color, zorder=2):
    n_layers = 50
    quantiles = np.linspace(0.05, 0.45, n_layers)
    base_alpha = 0.5 / n_layers
    for q in quantiles:
        lower = np.nanquantile(mat, q, axis=0)
        upper = np.nanquantile(mat, 1.0 - q, axis=0)
        ax.fill_between(xs, lower, upper, color=color, alpha=base_alpha,
                        linewidth=0, edgecolor="none", zorder=zorder)


def style_log_axis(ax):
    y_limits, _, y_exponents = _y_style()
    ax.set_xlim(0, MAX_ITER)
    ax.set_ylim(*y_limits)
    ax.set_yscale("log")

    step = 15 if MAX_ITER <= 90 else 20
    xticks = list(range(0, MAX_ITER + 1, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)
    )

    def custom_fmt(x, pos):
        if x > 0:
            exp = int(np.round(np.log10(x)))
            if exp in y_exponents:
                return r"$10^{%d}$" % exp
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
        q50 = np.nanquantile(mat, 0.50, axis=0)
        c = colors[n]

        _fuzzy_gradient_fill(ax, xs, mat, color=c)
        line, = ax.plot(xs, q50, color=c, zorder=4)
        handles.append(line)
        labels.append(n_legend(n))

        if SHOW_MU_OPT:
            mu50 = np.nanquantile(mu_infid_matrix(group, n), 0.50, axis=0)
            ax.plot(xs, mu50, color=c, linestyle=(0, (1, 1)), linewidth=1.0, zorder=3)

        if TRACE == "sampled":
            # sampling-resolution floor: below 1/(2N) a measurement cannot resolve
            ax.axhline(max(FLOOR_SHOTS_FRACTION / n, ABS_INFLOOR), color=c,
                       alpha=0.98, linestyle="--", linewidth=1.5, zorder=1)

        print(f"  N={n}: {mat.shape[0]} trials, final median {q50[-1]:.6g}, "
              f"best {np.nanmin(mat[:, -1]):.6g}")

    style_log_axis(ax)
    return handles, labels


def add_side_title(ax, leg, text):
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

    print(f"Panel — shot-budget trend (SANDIA emulator, trace={TRACE}):")
    handles, labels = draw_n_trend(ax, df)

    ax.set_ylabel(r"$1 - Q(\bm{x}_n^*)$" if USE_TEX else r"$1 - Q(x_n^*)$")
    ax.set_xlabel(r"$n$")
    y_limits, y_ticks, _ = _y_style()
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)

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
    global MAX_ITER, TRACE
    argv = sys.argv if argv is None else argv
    for a in argv[1:]:
        if a.startswith("--trace="):
            TRACE = a.split("=", 1)[1]
    argv = [a for a in argv if not a.startswith("--")]

    pickle_path = resolve_pickle(argv)
    print(f"Reading {pickle_path}")
    df = pd.read_pickle(pickle_path)

    metadata = df.attrs.get("metadata", {})
    if metadata:
        MAX_ITER = int(metadata.get("n_iter", MAX_ITER))
        print(f"  score={metadata.get('score', 'p11')}, n_iter={MAX_ITER}, "
              f"N_LIST={metadata.get('N_LIST')}, trials={len(df)}, "
              f"emulator={metadata.get('emulator')}, "
              f"sampled_shots={metadata.get('emulator_sample_shots')}")

    fig = build_figure(df)
    show_and_save(fig, f"figure_bo_sweep_sandia_emulator_{TRACE}.pdf")
    return fig


if __name__ == "__main__":
    main()
