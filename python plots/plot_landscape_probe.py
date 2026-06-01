import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── colour / style ───────────────────────────────────────────────────────────
C_BAL = "#2196F3"   # blue  – Q_balance
C_DET = "#F44336"   # red   – Q_det
LW = 2.0

# ── raw data ─────────────────────────────────────────────────────────────────

# FALSE PEAK  (u2 ≈ -0.8, u3 ≈ +1.0)
fp_u1 = dict(
    x     = [-0.20,-0.15,-0.10,-0.05, 0.00, 0.05, 0.10, 0.15, 0.20],
    Q_bal = [ 0.6396, 0.8133, 0.9032, 0.9589, 0.9780, 0.9701, 0.9324, 0.8228, 0.6176],
    Q_det = [ 0.4800, 0.4835, 0.4876, 0.4924, 0.5000, 0.5012, 0.5046, 0.5065, 0.5063],
    label = r"$u_1$  (fixed $u_2{=}{-}0.8,\;u_3{=}1.0$)",
)
fp_u2 = dict(
    x     = [-1.00,-0.95,-0.90,-0.85,-0.80,-0.75,-0.70,-0.65,-0.60,-0.55,-0.50],
    Q_bal = [ 0.8239, 0.8330, 0.8044, 0.8966, 0.9765, 0.9141, 0.8179, 0.8247, 0.8048, 0.7714, 0.5308],
    Q_det = [ 0.4978, 0.4982, 0.4989, 0.4995, 0.5000, 0.4996, 0.4990, 0.4984, 0.4978, 0.4981, 0.4990],
    label = r"$u_2$  (fixed $u_1{=}0,\;u_3{=}1.0$)",
)
fp_u3 = dict(
    x     = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    Q_bal = [0.9728,0.9807,0.9870,0.9846,0.9883,0.9957,0.9936,0.9877,0.9890,0.9846,0.9761],
    Q_det = [0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.5000],
    label = r"$u_3$  (fixed $u_1{=}0,\;u_2{=}{-}0.8$)",
)

# TRUE PEAK  (u1 ≈ 0, u2 ≈ 0, u3 ≈ 0)
tp_u1 = dict(
    x     = [-0.10,-0.08,-0.06,-0.04,-0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10],
    Q_bal = [ 0.8699, 0.8293, 0.7638, 0.7351, 0.7041, 0.6994, 0.6997, 0.7137, 0.7307, 0.7577, 0.7588],
    Q_det = [ 0.9136, 0.9452, 0.9691, 0.9854, 0.9946, 0.9974, 0.9947, 0.9874, 0.9767, 0.9634, 0.9483],
    label = r"$u_1$  (fixed $u_2{=}0,\;u_3{=}{-}0.3$)",
)
tp_u2 = dict(
    x     = [-0.10,-0.08,-0.06,-0.04,-0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10],
    Q_bal = [ 0.6270, 0.6036, 0.6878, 0.8563, 0.9233, 0.6982, 0.5118, 0.4100, 0.3695, 0.3874, 0.4390],
    Q_det = [ 0.8958, 0.9355, 0.9662, 0.9867, 0.9969, 0.9974, 0.9895, 0.9750, 0.9558, 0.9338, 0.9106],
    label = r"$u_2$  (fixed $u_1{=}0,\;u_3{=}{-}0.3$)",
)
tp_u3 = dict(
    x     = [-0.50,-0.45,-0.40,-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05, 0.00, 0.05, 0.10],
    Q_bal = [ 0.5210, 0.5620, 0.6022, 0.6538, 0.6926, 0.7416, 0.7846, 0.8299, 0.8839, 0.9177, 0.9654, 0.9814, 0.9388],
    Q_det = [ 0.9932, 0.9944, 0.9955, 0.9965, 0.9974, 0.9981, 0.9987, 0.9992, 0.9996, 0.9999, 1.0000, 1.0000, 0.9999],
    label = r"$u_3$  (fixed $u_1{=}0,\;u_2{=}0$)",
)

# ── layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

row_titles = [
    r"False peak region  ($u_2{\approx}{-}0.8,\;u_3{\approx}{+}1.0$,  $Q_\mathrm{det}\approx0.50$)",
    r"True peak region   ($u_1{\approx}0,\;u_2{\approx}0,\;u_3{\approx}0$,  $Q_\mathrm{det}\approx1.00$)",
]
datasets = [
    [fp_u1, fp_u2, fp_u3],
    [tp_u1, tp_u2, tp_u3],
]

axes = []
for row in range(2):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])
        d  = datasets[row][col]

        ax.plot(d["x"], d["Q_bal"], color=C_BAL, lw=LW, marker="o", ms=4,
                label=r"$Q_\mathrm{balance}$")
        ax.plot(d["x"], d["Q_det"], color=C_DET, lw=LW, marker="s", ms=4,
                label=r"$Q_\mathrm{det}$")

        # highlight the best-Q_det point
        best_idx = int(np.argmax(d["Q_det"]))
        ax.axvline(d["x"][best_idx], color="gray", lw=1.0, ls="--", alpha=0.6)

        ax.set_xlabel(d["label"], fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Score", fontsize=9)
        if col == 2 and row == 0:
            ax.legend(fontsize=8, loc="lower right")
        axes.append(ax)

# row labels
for row, title in enumerate(row_titles):
    fig.text(0.5, 0.96 - row * 0.50, title,
             ha="center", va="top", fontsize=10, fontweight="bold",
             color=["#B71C1C", "#1A237E"][row])

# global note
fig.text(0.5, 0.01,
         "Dashed vertical line = best $Q_\\mathrm{det}$ along each slice.  "
         "False peak: $Q_\\mathrm{balance}$ peaks where $Q_\\mathrm{det}\\approx0.5$.  "
         "True peak: $Q_\\mathrm{balance}<Q_\\mathrm{balance}^{\\mathrm{false}}$ everywhere.",
         ha="center", va="bottom", fontsize=8, color="gray")

out = "scripts/data/landscape_probe.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
