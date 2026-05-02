"""
Plot GP fit quality study results.
Reads: scripts/data/gp_fit_quality_3d_2ms.csv

Produces 4 figures (saved to scripts/data/):
  1. RMSE vs n_pts          — one line per N_shots, fixed mode
  2. RMSE vs N_total        — fixed vs adaptive comparison
  3. Coverage-95 vs n_pts   — calibration check (dashed line at 0.95)
  4. NLPD vs n_pts          — proper scoring rule
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_FILE = os.path.join(DATA_DIR, "gp_fit_quality_3d_2ms.csv")

df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} rows from {CSV_FILE}")
print(df.groupby(["mode", "N_shots"])["n_pts"].value_counts().unstack().to_string())
# Warn if test coverage dropped substantially (training pts near test grid)
low_cov = df[df["n_valid"] < 0.8 * df["n_valid"].max()]
if not low_cov.empty:
    print(f"\nNote: {len(low_cov)} configs have n_valid < 80% of max "
          f"(min n_valid = {df['n_valid'].min()}) — large n_pts excludes nearby test pts.")

N_SHOTS_ALL = sorted(df["N_shots"].unique())
N_PTS_ALL   = sorted(df["n_pts"].unique())

# ── Colour / style maps ───────────────────────────────────────────────────────
PALETTE   = plt.get_cmap("tab10")
N_COLORS  = {N: PALETTE(i) for i, N in enumerate(N_SHOTS_ALL)}
MODE_STYLE = {"fixed": "-", "adaptive": "--"}
MODE_ALPHA = {"fixed": 0.85, "adaptive": 0.7}

def _agg(sub, col):
    """Return (x, mean, sem) grouped by n_pts for a metric column."""
    g = sub.groupby("n_pts")[col]
    m = g.mean(); s = g.sem()
    return m.index.values, m.values, s.values

def _agg_by_N_total(sub, col):
    """Return (N_total_mean, mean, sem) grouped by N_total for a metric."""
    g = sub.groupby("N_total")[col]
    m = g.mean(); s = g.sem()
    return m.index.values, m.values, s.values


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — RMSE vs n_pts: raw (solid) and noise-corrected (dashed)
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fix_df = df[df["mode"] == "fixed"]
for ax, metric, title_suffix in zip(axes1,
                                    ["rmse",      "rmse_corr"],
                                    ["raw RMSE",  "RMSE corrected for Q_ref noise"]):
    for N in N_SHOTS_ALL:
        sub = fix_df[fix_df["N_shots"] == N]
        x, m, s = _agg(sub, metric)
        c = N_COLORS[N]
        # Band = ±1 SEM across seeds; also overlay per-run rmse_se as error bars
        ax.plot(x, m, "-o", color=c, markersize=4, label=f"N={N}")
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)
        if metric == "rmse":
            # Add within-run SE bars (mean rmse_se across seeds)
            _, m_se, _ = _agg(sub, "rmse_se")
            ax.errorbar(x, m, yerr=m_se, fmt="none", color=c, capsize=3, alpha=0.5)
    ax.set_xlabel("Training points (n)")
    ax.set_title(title_suffix)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Shots/pt", fontsize=8)

axes1[0].set_ylabel("RMSE vs ground truth")
fig1.suptitle("GP fit quality — fixed N shots, Q_varMS(m=2) 3D\n"
              "Band = ±1 SEM across seeds; bars = within-run SE")
fig1.tight_layout()
fig1.savefig(os.path.join(DATA_DIR, "gp_fit_rmse_vs_npts.png"), dpi=150)
print("Saved gp_fit_rmse_vs_npts.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — RMSE vs N_total (fixed vs adaptive, one colour per N_shots)
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
for mode in ["fixed", "adaptive"]:
    mode_df = df[df["mode"] == mode]
    for N in N_SHOTS_ALL:
        sub = mode_df[mode_df["N_shots"] == N]
        if sub.empty:
            continue
        x, m, s = _agg_by_N_total(sub, "rmse")
        c   = N_COLORS[N]
        ls  = MODE_STYLE[mode]
        alp = MODE_ALPHA[mode]
        label = f"N={N}, {mode}" if mode == "fixed" else f"N={N}, adaptive"
        ax2.plot(x, m, ls, color=c, alpha=alp, markersize=3,
                 label=label if N == N_SHOTS_ALL[0] or mode == "adaptive" else None)
        ax2.fill_between(x, m - s, m + s, color=c, alpha=0.08)

ax2.set_xlabel("Total shots (n × N)")
ax2.set_ylabel("RMSE vs ground truth")
ax2.set_title("GP fit quality — RMSE vs total shot budget\nsolid=fixed N, dashed=adaptive N")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
ax2.grid(True, which="both", alpha=0.3)

# Custom legend: colours = N_shots, linestyle = mode
import matplotlib.lines as mlines
handles = [mlines.Line2D([], [], color=N_COLORS[N], label=f"N={N}") for N in N_SHOTS_ALL]
handles += [mlines.Line2D([], [], color="gray", ls="-",  label="fixed"),
            mlines.Line2D([], [], color="gray", ls="--", label="adaptive")]
ax2.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
fig2.tight_layout()
fig2.savefig(os.path.join(DATA_DIR, "gp_fit_rmse_vs_Ntotal.png"), dpi=150)
print("Saved gp_fit_rmse_vs_Ntotal.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Coverage-95: raw GP σ vs adjusted σ_total = √(σ_GP²+σ_ref²)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, metric, title in zip(axes3,
                              ["cov95",     "cov95_adj"],
                              ["GP σ only", "Adjusted σ_total = √(σ_GP²+σ_ref²)"]):
    mode_df = df[df["mode"] == "fixed"]
    for N in N_SHOTS_ALL:
        sub = mode_df[mode_df["N_shots"] == N]
        if sub.empty:
            continue
        x, m, s = _agg(sub, metric)
        c = N_COLORS[N]
        ax.plot(x, m, "-o", color=c, markersize=4, label=f"N={N}")
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)
    ax.axhline(0.95, color="k", lw=1.2, ls=":", label="ideal 95%")
    ax.set_xlabel("Training points (n)")
    ax.set_title(title)
    ax.legend(title="Shots/pt", fontsize=7, loc="lower right")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.4)

axes3[0].set_ylabel("Coverage of 95% PI  (fixed N, mean ± SEM across seeds)")
fig3.suptitle("GP calibration check — adjusted coverage accounts for Q_ref shot noise\n"
              "Q_varMS(m=2), 3D")
fig3.tight_layout()
fig3.savefig(os.path.join(DATA_DIR, "gp_fit_coverage95.png"), dpi=150)
print("Saved gp_fit_coverage95.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — CRPS and adjusted NLPD vs n_pts (proper scoring rules)
# ═══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
for ax, metric, ylabel, title in zip(
        axes4,
        ["crps",     "nlpd_adj"],
        ["CRPS (lower = better)", "Adjusted NLPD (lower = better)"],
        ["CRPS — uses σ_total, proper scoring rule",
         "NLPD adjusted for Q_ref shot noise"]):
    fix_df = df[df["mode"] == "fixed"]
    for N in N_SHOTS_ALL:
        sub = fix_df[fix_df["N_shots"] == N]
        if sub.empty:
            continue
        x, m, s = _agg(sub, metric)
        c = N_COLORS[N]
        ax.plot(x, m, "-o", color=c, markersize=4, label=f"N={N}")
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)
    ax.set_xlabel("Training points (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Shots/pt", fontsize=7, loc="upper right")
    ax.grid(alpha=0.4)

fig4.suptitle("GP calibration — proper scoring rules, fixed N shots\nQ_varMS(m=2), 3D")
fig4.tight_layout()
fig4.savefig(os.path.join(DATA_DIR, "gp_fit_scoring.png"), dpi=150)
print("Saved gp_fit_scoring.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Mean posterior σ vs n_pts (how uncertainty shrinks)
# ═══════════════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(7, 4.5))
fix_df = df[df["mode"] == "fixed"]
for N in N_SHOTS_ALL:
    sub = fix_df[fix_df["N_shots"] == N]
    x, m, s = _agg(sub, "mean_sigma")
    c = N_COLORS[N]
    ax5.plot(x, m, "-o", color=c, markersize=4, label=f"N={N}")
    ax5.fill_between(x, m - s, m + s, color=c, alpha=0.15)

# RMSE overlay (dashed, same colours) to compare with uncertainty
for N in N_SHOTS_ALL:
    sub = fix_df[fix_df["N_shots"] == N]
    x, m, s = _agg(sub, "rmse")
    ax5.plot(x, m, "--", color=N_COLORS[N], alpha=0.5, markersize=3)

ax5.set_xlabel("Training points (n)")
ax5.set_ylabel("Value")
ax5.set_title("GP posterior σ (solid) vs actual RMSE (dashed)\nQ_varMS(m=2), 3D, fixed N")
ax5.legend(title="Shots/pt", fontsize=8)
ax5.set_yscale("log")
ax5.yaxis.set_minor_formatter(mticker.NullFormatter())
ax5.grid(True, which="both", alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(DATA_DIR, "gp_fit_sigma_vs_rmse.png"), dpi=150)
print("Saved gp_fit_sigma_vs_rmse.png")

plt.show()
