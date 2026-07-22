#!/usr/bin/env python3
"""
make_optimal_inputs_table.py

Reads the same trace directories as plot_figure_ab_vertical.py and prints
two tables (scale groups and N groups) showing, for the last recorded
iteration of each seed:

    median  [5th pct, 95th pct]

for each of the three normalised inputs u1 (Δf_cl), u2 (Δf_sb), u3 (ΔA).
"""

import glob
import csv
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (mirror plot_figure_ab_vertical.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"

SCALE_GROUPS = [
    {"label": "scale = 0.1",
     "dir": DATA_DIR / "traces_freqspan10_bound010_NInf_nostop100_stream_100seeds"},
    {"label": "scale = 0.5",
     "dir": DATA_DIR / "traces_freqspan10_bound050_NInf_nostop100_stream_100seeds"},
    {"label": "scale = 1.0",
     "dir": DATA_DIR / "traces_freqspan10_bound100_NInf_nostop100_stream_100seeds"},
]

N_GROUPS = [
    {"label": "N = 100",   "dir": DATA_DIR / "traces_freqspan10_bound050_N100_nostop100_stream_100seeds"},
    {"label": "N = 1000",  "dir": DATA_DIR / "traces_freqspan10_bound050_N1000_nostop100_stream_100seeds"},
    {"label": "N = 10k",   "dir": DATA_DIR / "traces_freqspan10_bound050_N10000_nostop100_stream_100seeds"},
    {"label": "N = 100k",  "dir": DATA_DIR / "traces_freqspan10_bound050_N100000_nostop100_stream_100seeds"},
    {"label": "N = Inf",   "dir": DATA_DIR / "traces_freqspan10_bound050_NInf_nostop100_stream_100seeds"},
]

INPUT_COLS  = ["x_rec_u1", "x_rec_u2", "x_rec_u3"]
INPUT_NAMES = [r"Δf_cl (kHz)", r"Δf_sb (kHz)", r"A/A_opt"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def last_row_inputs(path: str) -> np.ndarray:
    """Return x_rec_u1/u2/u3 from the last iteration row of one trace CSV."""
    best_iter = -1
    best_vals = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = int(float(row["iter"]))
            if it > best_iter:
                best_iter = it
                best_vals = np.array([float(row[c]) for c in INPUT_COLS])
    return best_vals


def group_stats(directory: Path):
    """
    Returns array of shape (n_seeds, 3) with the last-iteration inputs
    for every seed found in `directory`.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    paths = sorted(glob.glob(str(directory / "trace_seed*_ucb.csv")))
    if not paths:
        return None

    rows = []
    for p in paths:
        v = last_row_inputs(p)
        if v is not None:
            rows.append(v)
    if not rows:
        return None
    mat = np.array(rows)          # (n_seeds, 3)
    mat[:, 0] = 10.0 * mat[:, 0]          # u1 → Δf_cl (kHz)
    mat[:, 1] = 10.0 * mat[:, 1]          # u2 → Δf_sb (kHz)
    mat[:, 2] = 1.0 + 0.2 * mat[:, 2]    # u3 → A/A_opt
    return mat


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_sci(v):
    """Scientific notation with explicit +/- sign on the coefficient."""
    s = f"{v:.2e}"
    # ensure a space is reserved for positive values so columns align
    return f" {s}" if v >= 0 else s


def fmt(med, lo, hi):
    return f"{fmt_sci(med)}  [{fmt_sci(lo)}, {fmt_sci(hi)}]"


def print_table(groups, title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # Header
    col_w = 36
    header = f"{'Group':<18}" + "".join(f"{n:<{col_w}}" for n in INPUT_NAMES)
    print(header)
    print("-" * (18 + col_w * len(INPUT_NAMES)))

    for g in groups:
        mat = group_stats(g["dir"])
        if mat is None:
            print(f"{g['label']:<18}  [directory not found]")
            continue

        n = mat.shape[0]
        med = np.nanmedian(mat, axis=0)
        lo  = np.nanpercentile(mat, 5,  axis=0)
        hi  = np.nanpercentile(mat, 95, axis=0)

        row = f"{g['label']:<18}"
        for j in range(len(INPUT_COLS)):
            row += f"{fmt(med[j], lo[j], hi[j]):<{col_w+4}}"
        print(row + f"  (n={n})")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_table(SCALE_GROUPS, "Scale groups  (N = Inf,  scale = 0.1 / 0.5 / 1.0)")
    print_table(N_GROUPS,     "Shot-count groups  (scale = 0.5,  N = 100 … Inf)")
