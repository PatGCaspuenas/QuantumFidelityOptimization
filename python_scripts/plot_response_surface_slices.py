"""
Plot 1D response surface slices — all Q models together.

Produces 5 figures, each with a 5×5 grid of subplots (25 total):
  Fig 1  — FCL  slice  : rows=u_fsb,  cols=u_A,    u_phi=0
  Fig 2  — FSB  slice  : rows=u_fcl,  cols=u_A,    u_phi=0
  Fig 3  — A    slice  : rows=u_fcl,  cols=u_fsb,  u_phi=0
  Fig 4  — PHI  slice  : rows=u_fcl,  cols=u_fsb,  u_A=0
  Fig 5  — FCL  slice  : rows=u_phi,  cols=u_A,    u_fsb=0  (phase×amp interaction)

Usage:
    python scripts/plot_response_surface_slices.py [path/to/csv]
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import seaborn as sns
    def _husl_colors(n):
        return sns.color_palette("husl", n)
except ImportError:
    def _husl_colors(n):
        return plt.cm.hsv(np.linspace(0, 0.85, n))

# ── Config ───────────────────────────────────────────────────────────────────

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "data", "response_surface_all_q_slices.csv"
)
OUT_DIR = os.path.join(os.path.dirname(CSV_PATH))

FIXED_VALS = [-1.0, -0.5, 0.0, 0.5, 1.0]
EPS = 1e-6

Q_COLS = [
    "Q_varMS_2ms",
    "Q_varMS_3ms",
    "Q_varMS_balance_3ms",
    "Q_jacobian",
    "Q_det",
    "Q_noisy_1ms"
]
Q_LABELS = [
    "Q_varMS 2ms",
    "Q_varMS 3ms",
    "Q_varMS balance 3ms",
    "Q_jacobian",
    "Q_det",
    "Q_noisy"
]

# Line styles: Q_det solid, Q_noisy dashed, others varied so overlaps are distinguishable
LINE_STYLES = ["-", "-", "-.", ":", "-", "--"]

COLORS = _husl_colors(len(Q_COLS))

PARAM_LABEL = {
    "u_fcl": r"$u_{f_{cl}}$",
    "u_fsb": r"$u_{f_{sb}}$",
    "u_A":   r"$u_A$",
    "u_phi": r"$u_\phi$",
}

FIGURES = [
    dict(
        title=r"Active: $f_{cl}$   |   rows: $u_{f_{sb}}$,  cols: $u_A$,  $u_\phi=0$",
        slice_name="fcl", active="u_fcl",
        row="u_fsb", col="u_A", filter_u="u_phi", filter_val=0.0,
        fname="response_surface_fcl_fsb_A.png",
    ),
    dict(
        title=r"Active: $f_{sb}$   |   rows: $u_{f_{cl}}$,  cols: $u_A$,  $u_\phi=0$",
        slice_name="fsb", active="u_fsb",
        row="u_fcl", col="u_A", filter_u="u_phi", filter_val=0.0,
        fname="response_surface_fsb_fcl_A.png",
    ),
    dict(
        title=r"Active: $A$   |   rows: $u_{f_{cl}}$,  cols: $u_{f_{sb}}$,  $u_\phi=0$",
        slice_name="A", active="u_A",
        row="u_fcl", col="u_fsb", filter_u="u_phi", filter_val=0.0,
        fname="response_surface_A_fcl_fsb.png",
    ),
    dict(
        title=r"Active: $\phi$   |   rows: $u_{f_{cl}}$,  cols: $u_{f_{sb}}$,  $u_A=0$",
        slice_name="phi", active="u_phi",
        row="u_fcl", col="u_fsb", filter_u="u_A", filter_val=0.0,
        fname="response_surface_phi_fcl_fsb.png",
    ),
    dict(
        title=r"Active: $f_{cl}$   |   rows: $u_\phi$,  cols: $u_A$,  $u_{f_{sb}}=0$  (phase $\times$ amp)",
        slice_name="fcl", active="u_fcl",
        row="u_phi", col="u_A", filter_u="u_fsb", filter_val=0.0,
        fname="response_surface_fcl_phi_A.png",
    ),
]

# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(df, cfg):
    slice_name = cfg["slice_name"]
    active     = cfg["active"]
    row_key    = cfg["row"]
    col_key    = cfg["col"]
    filter_u   = cfg["filter_u"]
    filter_val = cfg["filter_val"]

    # Filter to this slice and the chosen fixed-param value
    mask = (df["slice"] == slice_name) & (np.abs(df[filter_u] - filter_val) < EPS)
    base = df[mask].copy()

    fig, axes = plt.subplots(
        5, 5, figsize=(26, 26),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.45, wspace=0.25),
    )
    fig.suptitle(cfg["title"], fontsize=13, fontweight="bold", y=0.995)

    for ri, rv in enumerate(FIXED_VALS):
        for ci, cv in enumerate(FIXED_VALS):
            ax = axes[ri, ci]

            sub = base[
                (np.abs(base[row_key] - rv) < EPS) &
                (np.abs(base[col_key] - cv) < EPS)
            ].sort_values(active)

            for j, (qcol, qlabel) in enumerate(zip(Q_COLS, Q_LABELS)):
                if sub.empty:
                    continue
                ax.plot(
                    sub[active], sub[qcol],
                    color=COLORS[j], lw=1.4, alpha=0.88,
                    linestyle=LINE_STYLES[j],
                    label=qlabel,
                )

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-1.05, 1.05)
            ax.axhline(1.0, color="k", lw=0.4, ls="--", alpha=0.3)
            ax.tick_params(labelsize=6, labelbottom=True, labelleft=True)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.grid(True, lw=0.3, alpha=0.4)

            # Subplot title: all fixed params including the slice filter
            title_str = (
                f"{PARAM_LABEL[col_key]}={cv:+.1f}, "
                f"{PARAM_LABEL[row_key]}={rv:+.1f}, "
                f"{PARAM_LABEL[filter_u]}=0"
            )
            ax.set_title(title_str, fontsize=6.5, pad=3)

            # x/y axis labels on every subplot
            ax.set_xlabel(PARAM_LABEL[active], fontsize=7, labelpad=2)
            ax.set_ylabel("Q", fontsize=7, labelpad=2)

    # Shared legend below figure
    legend_handles = [
        Line2D([0], [0], color=COLORS[j], lw=1.8,
               linestyle=LINE_STYLES[j], label=Q_LABELS[j])
        for j in range(len(Q_COLS))
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    return fig


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows loaded. Slices: {sorted(df['slice'].unique())}")

    os.makedirs(OUT_DIR, exist_ok=True)

    for i, cfg in enumerate(FIGURES, 1):
        print(f"[{i}/5] {cfg['fname']} ...")
        fig = make_figure(df, cfg)
        out = os.path.join(OUT_DIR, cfg["fname"])
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"       saved → {out}")

    print("Done.")


if __name__ == "__main__":
    main()
