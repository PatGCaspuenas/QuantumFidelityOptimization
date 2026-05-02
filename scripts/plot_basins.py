import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import sys, os

csv = os.path.join(os.path.dirname(__file__), "data", "basin_probe.csv")
df  = pd.read_csv(csv)

fp = df[df["basin"] == "false"].copy()
tp = df[df["basin"] == "true"].copy()

def make_grid(sub, xcol, ycol, zcol):
    xs = np.sort(sub[xcol].unique())
    ys = np.sort(sub[ycol].unique())
    Z  = np.full((len(ys), len(xs)), np.nan)
    for _, row in sub.iterrows():
        xi = np.searchsorted(xs, row[xcol])
        yi = np.searchsorted(ys, row[ycol])
        Z[yi, xi] = row[zcol]
    return xs, ys, Z

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                        width_ratios=[1, 1, 0.05])

norm_bal = Normalize(vmin=0.0, vmax=1.0)
norm_det = Normalize(vmin=0.0, vmax=1.0)
cmap_bal = "Blues"
cmap_det = "Reds"

panels = [
    # (dataframe, xcol, ycol, zcol, xlabel, ylabel, title)
    (fp, "u2", "u3", "Q_bal",
     r"$u_2$ (sideband freq)", r"$u_3$ (amplitude)",
     r"False-peak basin — $Q_\mathrm{balance}$  ($u_1=0$)"),
    (fp, "u2", "u3", "Q_det",
     r"$u_2$ (sideband freq)", r"$u_3$ (amplitude)",
     r"False-peak basin — $Q_\mathrm{det}$  ($u_1=0$)"),
    (tp, "u2", "u3", "Q_bal",
     r"$u_2$ (sideband freq)", r"$u_3$ (amplitude)",
     r"True-peak basin — $Q_\mathrm{balance}$  ($u_1=0$)"),
    (tp, "u2", "u3", "Q_det",
     r"$u_2$ (sideband freq)", r"$u_3$ (amplitude)",
     r"True-peak basin — $Q_\mathrm{det}$  ($u_1=0$)"),
]

positions = [(0,0), (0,1), (1,0), (1,1)]

for (sub, xc, yc, zc, xl, yl, title), (row, col) in zip(panels, positions):
    ax = fig.add_subplot(gs[row, col])
    xs, ys, Z = make_grid(sub, xc, yc, zc)

    norm = norm_bal if "bal" in zc else norm_det
    cmap = cmap_bal if "bal" in zc else cmap_det
    im   = ax.pcolormesh(xs, ys, Z, cmap=cmap, norm=norm, shading="nearest")

    # mark the maximum
    best_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.plot(xs[best_idx[1]], ys[best_idx[0]], "w*", ms=12,
            label=f"max={Z[best_idx]:.4f}")
    ax.legend(fontsize=8, loc="upper right",
              framealpha=0.7, handlelength=0)

    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)

    # shared colorbar column
    cax = fig.add_subplot(gs[row, 2])
    if col == 1:
        plt.colorbar(im, cax=cax)
        cax.tick_params(labelsize=8)
    else:
        cax.set_visible(False)

# global annotation
fp_best = fp.loc[fp["Q_bal"].idxmax()]
tp_best = tp.loc[tp["Q_bal"].idxmax()]
diff    = tp_best["Q_bal"] - fp_best["Q_bal"]
sign    = "+" if diff >= 0 else ""
fig.text(
    0.5, 0.01,
    f"False-peak max $Q_{{\\mathrm{{balance}}}}$ = {fp_best['Q_bal']:.4f}  "
    f"($Q_{{\\mathrm{{det}}}}={fp_best['Q_det']:.4f}$)     "
    f"True-peak max $Q_{{\\mathrm{{balance}}}}$ = {tp_best['Q_bal']:.4f}  "
    f"($Q_{{\\mathrm{{det}}}}={tp_best['Q_det']:.4f}$)     "
    f"Δ = {sign}{diff:.4f}",
    ha="center", va="bottom", fontsize=9,
    color="darkred" if diff < 0 else "darkblue"
)

out = os.path.join(os.path.dirname(__file__), "data", "basin_probe.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
