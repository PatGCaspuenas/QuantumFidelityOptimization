"""
Visualize the deterministic (shot-free) model bias of the Jacobian GP surrogate
vs the full deterministic simulator (Q_det).

Data:
  jacobian_probe_coarse.csv  — 29×29×41 grid (u1,u2,u3), Q_jac, Q_det
  jacobian_probe_lines.csv   — 200-pt line scans along each axis through the peak

No shots were taken, so bias = pure linearization error of the Jacobian model.
With N=400 shots, shot noise std ≈ sqrt(Q(1-Q)/N) ~ 0.025 is added on top.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

DATA_DIR = "scripts/data"
OUT      = "scripts/plots"

import os; os.makedirs(OUT, exist_ok=True)

coarse = pd.read_csv(f"{DATA_DIR}/jacobian_probe_coarse.csv")
lines  = pd.read_csv(f"{DATA_DIR}/jacobian_probe_lines.csv")

THRESH = 0.999
bias   = coarse.Q_det - coarse.Q_jac   # always positive (jac underestimates)

# ── find the best u3 slice ─────────────────────────────────────────────────────
best_idx = coarse.Q_jac.idxmax()
best_u3  = coarse.loc[best_idx, "u3"]
best_u2  = coarse.loc[best_idx, "u2"]
best_u1  = coarse.loc[best_idx, "u1"]
slice3d  = coarse[np.isclose(coarse.u3, best_u3)]

print(f"Peak at u1={best_u1:.4f}, u2={best_u2:.4f}, u3={best_u3:.4f}")
print(f"  Q_jac={coarse.loc[best_idx,'Q_jac']:.6f}  Q_det={coarse.loc[best_idx,'Q_det']:.6f}")
print(f"Bias stats: mean={bias.mean():.4f}  std={bias.std():.4f}  min={bias.min():.4f}  max={bias.max():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Bias overview (histogram + scatter + CDF)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# — histogram
ax = axes[0]
ax.hist(bias, bins=80, color="steelblue", edgecolor="none", density=True)
ax.axvline(bias.mean(), color="tomato", lw=1.5, label=f"mean={bias.mean():.3f}")
ax.axvline(bias.median(), color="orange", lw=1.5, ls="--", label=f"median={bias.median():.3f}")
ax.set_xlabel("Q_det − Q_jac  (model bias)")
ax.set_ylabel("density")
ax.set_title("Histogram of Jacobian bias\n(all 34 k grid points, no shot noise)")
ax.legend(fontsize=9)

# — scatter: bias vs Q_det (colour = Q_jac)
ax = axes[1]
sc = ax.scatter(coarse.Q_det, bias, c=coarse.Q_jac, s=0.5, cmap="viridis",
                vmin=0.7, vmax=1.0, rasterized=True)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(THRESH, color="tomato", lw=1, ls="--", label=f"threshold={THRESH}")
ax.set_xlabel("Q_det  (true fidelity)")
ax.set_ylabel("Q_det − Q_jac  (bias)")
ax.set_title("Bias vs true fidelity\n(colour = Q_jac)")
plt.colorbar(sc, ax=ax, label="Q_jac")
ax.legend(fontsize=9)

# — CDF of Q_jac and Q_det
ax = axes[2]
for col, label, color in [("Q_jac","Q_jac (jacobian)","steelblue"),
                            ("Q_det","Q_det (full sim)","tomato")]:
    vals = np.sort(coarse[col])
    cdf  = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, cdf, color=color, lw=1.5, label=label)
ax.axvline(THRESH, color="k", lw=1, ls="--", label=f"threshold={THRESH}")
# fraction above threshold
frac_jac = (coarse.Q_jac > THRESH).mean()
frac_det = (coarse.Q_det > THRESH).mean()
ax.text(0.02, 0.97, f"Frac > {THRESH}:\n  Q_jac: {frac_jac:.2%}\n  Q_det: {frac_det:.2%}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
ax.set_xlabel("Fidelity value")
ax.set_ylabel("CDF")
ax.set_title("CDF of Q_jac vs Q_det\nacross all grid points")
ax.legend(fontsize=9)

fig.suptitle("Jacobian model bias (deterministic, no shot noise)", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/jacobian_bias_overview.png", dpi=150, bbox_inches="tight")
print("Saved jacobian_bias_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Line scans — Q_jac vs Q_det along each axis
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

axis_labels = {"u1": "u1 (fcl offset)",
               "u2": "u2 (fsb offset)",
               "u3": "u3 (A offset)"}

for row, ax_name in enumerate(["u1", "u2", "u3"]):
    sub    = lines[lines.axis == ax_name].sort_values("u_scan")
    u      = sub.u_scan.values
    q_jac  = sub.Q_jac.values
    q_det  = sub.Q_det.values
    b      = q_det - q_jac

    # left: Q profiles + threshold
    ax = axes[row, 0]
    ax.plot(u, q_det,  color="tomato",    lw=2,   label="Q_det  (full sim)")
    ax.plot(u, q_jac,  color="steelblue", lw=2,   label="Q_jac (jacobian)")
    ax.fill_between(u, q_jac, q_det, alpha=0.15, color="orange", label="bias region")
    ax.axhline(THRESH, color="k", lw=1, ls="--", label=f"threshold={THRESH}")
    ax.set_xlabel(axis_labels[ax_name])
    ax.set_ylabel("Fidelity")
    ax.set_title(f"Line scan along {ax_name}: Q profiles")
    ax.legend(fontsize=8)
    ax.set_ylim(max(0.88, q_jac.min() - 0.01), 1.005)

    # right: bias along axis
    ax = axes[row, 1]
    ax.plot(u, b, color="darkorange", lw=2)
    ax.axhline(b.mean(), color="gray", lw=1, ls="--", label=f"mean bias = {b.mean():.4f}")
    ax.axhline(0, color="k", lw=0.5)
    # mark shot noise level for N=400
    q_peak = q_det.max()
    shot_std = np.sqrt(q_peak * (1 - q_peak) / 400)
    ax.axhline(shot_std, color="steelblue", lw=1, ls=":",
               label=f"shot std (N=400) = {shot_std:.4f}")
    ax.set_xlabel(axis_labels[ax_name])
    ax.set_ylabel("Q_det − Q_jac")
    ax.set_title(f"Line scan along {ax_name}: bias")
    ax.legend(fontsize=8)

fig.suptitle("Line scans through the peak: Jacobian bias (no shots)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/jacobian_bias_lines.png", dpi=150, bbox_inches="tight")
print("Saved jacobian_bias_lines.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: 2D heat maps at best u3 slice
# ══════════════════════════════════════════════════════════════════════════════
pivot_det  = slice3d.pivot(index="u2", columns="u1", values="Q_det")
pivot_jac  = slice3d.pivot(index="u2", columns="u1", values="Q_jac")
pivot_bias = pivot_det - pivot_jac

u1_vals = pivot_det.columns.values
u2_vals = pivot_det.index.values
ext = [u1_vals.min(), u1_vals.max(), u2_vals.min(), u2_vals.max()]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Q_det
ax = axes[0]
im = ax.imshow(pivot_det.values, origin="lower", extent=ext, aspect="auto",
               cmap="viridis", vmin=0.8, vmax=1.0)
ax.contour(u1_vals, u2_vals, pivot_det.values, levels=[THRESH],
           colors="white", linewidths=1.5)
ax.plot(best_u1, best_u2, "r*", ms=10, label="Q_jac peak")
plt.colorbar(im, ax=ax)
ax.set_xlabel("u1"); ax.set_ylabel("u2")
ax.set_title(f"Q_det  (u3={best_u3:.3f})\nwhite contour = {THRESH}")
ax.legend(fontsize=8)

# Q_jac
ax = axes[1]
im = ax.imshow(pivot_jac.values, origin="lower", extent=ext, aspect="auto",
               cmap="viridis", vmin=0.8, vmax=1.0)
ax.contour(u1_vals, u2_vals, pivot_jac.values, levels=[THRESH],
           colors="white", linewidths=1.5)
ax.plot(best_u1, best_u2, "r*", ms=10, label="Q_jac peak")
plt.colorbar(im, ax=ax)
ax.set_xlabel("u1"); ax.set_ylabel("u2")
ax.set_title(f"Q_jac  (u3={best_u3:.3f})\nwhite contour = {THRESH}")
ax.legend(fontsize=8)

# Bias = Q_det - Q_jac
ax = axes[2]
vmax_b = pivot_bias.values.max()
im = ax.imshow(pivot_bias.values, origin="lower", extent=ext, aspect="auto",
               cmap="YlOrRd", vmin=0, vmax=vmax_b)
ax.plot(best_u1, best_u2, "b*", ms=10, label="Q_jac peak")
plt.colorbar(im, ax=ax, label="Q_det − Q_jac")
ax.set_xlabel("u1"); ax.set_ylabel("u2")
ax.set_title(f"Bias = Q_det − Q_jac  (u3={best_u3:.3f})")
ax.legend(fontsize=8)

fig.suptitle(f"2D slice at u3={best_u3:.3f} (best Jacobian slice)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/jacobian_bias_2d_slice.png", dpi=150, bbox_inches="tight")
print("Saved jacobian_bias_2d_slice.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: relative bias = bias / Q_det  + bias at high-Q region
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

rel_bias = bias / coarse.Q_det

ax = axes[0]
high = coarse.Q_det > 0.95
ax.scatter(coarse.Q_det[~high], rel_bias[~high], s=0.5, color="steelblue",
           alpha=0.3, label="Q_det ≤ 0.95", rasterized=True)
ax.scatter(coarse.Q_det[high],  rel_bias[high],  s=2,   color="tomato",
           label=f"Q_det > 0.95 ({high.sum()} pts)", rasterized=True)
ax.axvline(THRESH, color="k", lw=1, ls="--")
ax.set_xlabel("Q_det")
ax.set_ylabel("(Q_det − Q_jac) / Q_det")
ax.set_title("Relative bias vs true fidelity")
ax.legend(fontsize=9)

# zoom into Q_det > 0.99 region only
ax = axes[1]
near_peak = coarse[coarse.Q_det > 0.99].copy()
near_peak["bias"] = near_peak.Q_det - near_peak.Q_jac
near_peak["rel"]  = near_peak["bias"] / near_peak.Q_det
ax.scatter(near_peak.Q_det, near_peak["bias"], s=3, c=near_peak.Q_jac,
           cmap="plasma", vmin=0.96, vmax=1.0)
ax.axvline(THRESH, color="k", lw=1, ls="--", label=f"threshold={THRESH}")
ax.axhline(0, color="k", lw=0.5)
# shot noise reference
q_ref = 0.999
sn = np.sqrt(q_ref * (1 - q_ref) / 400)
ax.axhline(sn, color="steelblue", lw=1.5, ls=":", label=f"shot std N=400 = {sn:.4f}")
ax.set_xlabel("Q_det")
ax.set_ylabel("Q_det − Q_jac  (bias)")
ax.set_title("Bias zoomed into Q_det > 0.99\n(colour = Q_jac)")
ax.legend(fontsize=9)

fig.suptitle("Jacobian bias in the high-fidelity regime", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/jacobian_bias_highQ.png", dpi=150, bbox_inches="tight")
print("Saved jacobian_bias_highQ.png")

plt.show()
print("\nDone. Plots in", OUT)
