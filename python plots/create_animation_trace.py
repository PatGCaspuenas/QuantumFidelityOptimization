"""
create_animation_trace.py

1x3 animation of BO trace data:
  ax1 — 3D scatter of sampled points + current y_rec_cur star (log-z)
  ax2 — GP posterior on u2 slice (u1=u3=0) + inset zoom near peak
  ax3 — GP posterior on u1 slice (u2=u3=0) + inset zoom near peak
  Gradient progress bar ($n$) above. Shared legend below.

Usage:
  python "python plots/create_animation_trace.py"
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import seaborn as sns
    _husl8 = sns.color_palette("husl", 8)
except ImportError:
    _husl8 = [(0.90, 0.20, 0.20), (0.90, 0.60, 0.10), (0.50, 0.90, 0.20),
              (0.10, 0.80, 0.40), (0.10, 0.70, 0.80), (0.10, 0.30, 0.90),
              (0.50, 0.10, 0.90), (0.90, 0.10, 0.60)]

# ============================================================
# CONFIG
# ============================================================
TRACE_FILE         = "data/QvarMS_N50_v2_0999_trace_trace.txt"
SLICES_FILE        = "data/QvarMS_N50_v2_0999_trace_gp_slices.txt"
SURFACE_FILE       = "data/response_surface_1d_slices_N1000.csv"
OUTPUT_FILE        = "data/animation_trace.gif"
FPS                = 8
FIDELITY_THRESHOLD = 0.999
Y_MIN_LOG          = 1e-2    # lower bound for log axes
ZOOM_YMIN          = 0.995
ZOOM_YMAX          = 1.005
ZOOM_XHALF         = 0.4     # inset x range: [-ZOOM_XHALF, +ZOOM_XHALF]

# ============================================================
# STYLING
# ============================================================
DARK_BG    = '#212121'
TEXT_COLOR = '#eeeeee'

mpl.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   DARK_BG,
    'axes.edgecolor':   TEXT_COLOR,
    'axes.labelcolor':  TEXT_COLOR,
    'axes.titlecolor':  TEXT_COLOR,
    'legend.edgecolor': TEXT_COLOR,
    'legend.facecolor': DARK_BG,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
    'grid.color':       '#444444',
    'text.color':       TEXT_COLOR,
    'font.size':        14,
    'legend.fontsize':  14,
    'axes.linewidth':   1.2,
    'font.family':      'serif',
    'mathtext.fontset': 'cm',
})

COL_DET    = '#d0d0d0'      # grey          — Q_det (deterministic reference)
COL_NOISY  = _husl8[5]     # blue          — Q_varMS static curve
COL_GP     = _husl8[1]     # orange        — GP posterior
COL_ACQ    = _husl8[0]     # red           — sampled points
COL_REC    = _husl8[3]     # green         — y_rec_cur stars
COL_THRESH = _husl8[7]     # magenta       — fidelity threshold

# Colormaps for iteration-based shading (dark = early, bright = late)
def _dark(c, factor=0.15):
    arr = np.array(mpl.colors.to_rgb(c))
    return tuple(arr * factor)

cmap_acq = LinearSegmentedColormap.from_list('acq', [_dark(COL_ACQ), COL_ACQ])
cmap_rec = LinearSegmentedColormap.from_list('rec', [_dark(COL_REC), COL_REC])

# ============================================================
# PARSERS
# ============================================================
def parse_trace(path):
    """Returns (init_arr [N_init x 5], acq_arr [N_iters x 12]).
    Rows shorter than 12 cols are nan-padded.
    """
    init_rows, acq_rows = [], []
    section = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                if 'SECTION: INITIAL_TRAINING' in line:
                    section = 'init'
                elif 'SECTION: ACQ_REC' in line:
                    section = 'acq'
                elif 'SECTION: FINAL_TRAINING' in line:
                    section = 'final'
                continue
            if not line.strip():
                continue
            parts = line.split('\t')
            if section == 'init' and len(parts) == 5:
                init_rows.append([float(x) for x in parts])
            elif section == 'acq' and len(parts) >= 10:
                row = [float(x) for x in parts[:12]]
                row += [float('nan')] * (12 - len(row))
                acq_rows.append(row)
    init_arr = np.array(init_rows) if init_rows else np.empty((0, 5))
    acq_arr  = np.array(acq_rows)  if acq_rows  else np.empty((0, 12))
    return init_arr, acq_arr


def parse_slices(path):
    """Returns two dicts keyed by iteration: {it: {'coord', 'mu', 'sigma'}}."""
    u1_data, u2_data = {}, {}
    section = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                if 'SECTION: GP_SLICE_U1' in line:
                    section = 'u1'
                elif 'SECTION: GP_SLICE_U2' in line:
                    section = 'u2'
                continue
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                continue
            it    = int(float(parts[0]))
            coord = float(parts[1])
            mu    = float(parts[2])
            sig   = float(parts[3])
            target = u1_data if section == 'u1' else u2_data
            if it not in target:
                target[it] = {'coord': [], 'mu': [], 'sigma': []}
            target[it]['coord'].append(coord)
            target[it]['mu'].append(mu)
            target[it]['sigma'].append(sig)
    for d in [u1_data, u2_data]:
        for it in d:
            d[it] = {k: np.array(v) for k, v in d[it].items()}
    return u1_data, u2_data

# ============================================================
# LOAD DATA
# ============================================================
print("Loading trace data ...")
init_pts, acq_rec = parse_trace(TRACE_FILE)
gp_u1, gp_u2     = parse_slices(SLICES_FILE)

print("Loading 1-D response surface slices ...")
surf = pd.read_csv(SURFACE_FILE)

_s2       = surf[surf['slice'] == 'u1u3=0'].sort_values('u_val')
s2_u      = _s2['u_val'].values
s2_qdet   = _s2['Q_det'].values
s2_qvarMS = _s2['Q_varMS'].values
u2_star   = float(s2_u[np.argmax(s2_qdet)])

_s3       = surf[surf['slice'] == 'u2u3=0'].sort_values('u_val')
s3_u      = _s3['u_val'].values
s3_qdet   = _s3['Q_det'].values
s3_qvarMS = _s3['Q_varMS'].values
u1_star   = float(s3_u[np.argmax(s3_qdet)])

iters    = np.unique(acq_rec[:, 0].astype(int))
n_frames = len(iters)
n_init   = len(init_pts)

init_u = init_pts[:, :3]
init_y = init_pts[:,  3]

print(f"Frames: {n_frames}, iterations: {iters[0]}--{iters[-1]}")

# ============================================================
# FIGURE LAYOUT
# ============================================================
fig = plt.figure(figsize=(18, 5.2))

# 1x3 subplots; ax1 slightly narrower to give room for 3-D z-axis label
gs = fig.add_gridspec(
    1, 3, width_ratios=[0.85, 1, 1],
    wspace=0.55, left=0.09, right=0.97, top=0.78, bottom=0.22,
)
ax1 = fig.add_subplot(gs[0], projection='3d')
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Inset zoom axes — created as figure-level axes so ax.cla() never destroys them.
# Position is bottom-right of ax2/ax3 in figure coordinates.
fig.canvas.draw()   # force layout so get_position() is accurate
def _inset_fig_bbox(ax, rel_x=0.54, rel_y=0.03, rel_w=0.44, rel_h=0.40):
    p = ax.get_position()
    return [p.x0 + rel_x * p.width, p.y0 + rel_y * p.height,
            rel_w * p.width,         rel_h * p.height]

ax2_inset = fig.add_axes(_inset_fig_bbox(ax2))
ax3_inset = fig.add_axes(_inset_fig_bbox(ax3))

# ---- Progress bar --------------------------------------------------
ax_prog = fig.add_axes([0.12, 0.89, 0.82, 0.020])
_prog_init = np.zeros((1, n_frames))
prog_img = ax_prog.imshow(
    _prog_init, aspect='auto', cmap='gray', vmin=0, vmax=1,
    extent=[0.5, n_frames + 0.5, 0, 1], interpolation='nearest',
)
ax_prog.set_xlim(0.5, n_frames + 0.5)
ax_prog.set_ylim(0, 1)
ax_prog.set_yticks([])
ax_prog.set_xticks([1, n_frames])
ax_prog.set_xticklabels(['1', str(iters[-1])], fontsize=9, color=TEXT_COLOR)
ax_prog.xaxis.tick_top()
ax_prog.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=3)
for sp in ax_prog.spines.values():
    sp.set_edgecolor(TEXT_COLOR)
fig.text(0.10, 0.899, r'$n$', ha='right', va='center', fontsize=13, color=TEXT_COLOR)

# ---- Shared static legend at the bottom ----------------------------
_legend_handles = [
    Line2D([0], [0], color=COL_DET,   lw=1.5, ls='--',
           label=r'$Q_\mathrm{det}$'),
    Line2D([0], [0], color=COL_NOISY, lw=1.5, ls='-',
           label=r'$Q_\mathrm{varMS}$ ($N\!=\!1000$)'),
    Patch(facecolor=COL_GP, alpha=0.5,
          label=r'GP $\mu_n\pm\sigma_n$'),
    Line2D([0], [0], color=COL_GP, lw=2,
           label=r'GP $\mu_n$'),
    Line2D([0], [0], color=COL_ACQ, lw=0, marker='o', markersize=6,
           markerfacecolor=COL_ACQ, label=r'$y_n$ (sampled)'),
    Line2D([0], [0], color=COL_REC, lw=0, marker='*', markersize=9,
           markerfacecolor=COL_REC, label=r'$y^*_n$ (rec.)'),
    Line2D([0], [0], color=COL_THRESH, lw=1.2, ls='--',
           label=rf'$Q_\mathrm{{thresh}}={FIDELITY_THRESHOLD}$'),
]
fig.legend(
    handles=_legend_handles,
    loc='lower center', ncol=len(_legend_handles),
    bbox_to_anchor=(0.5, 0.00), fontsize=10,
    framealpha=0.35, handlelength=1.6, columnspacing=1.0,
)

# ============================================================
# HELPERS
# ============================================================
Z_LOG_MIN = -2
Z_LOG_MAX =  0


def style_3d(ax):
    pane = (0.13, 0.13, 0.13, 1.0)
    for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a.set_pane_color(pane)
        a._axinfo["grid"].update({"color": (0.35, 0.35, 0.35, 1)})
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(Z_LOG_MIN, Z_LOG_MAX)
    ax.set_xlabel(r'$\Delta f_\mathrm{cl}$', fontsize=12, labelpad=4)
    ax.set_ylabel(r'$\Delta f_\mathrm{sb}$', fontsize=12, labelpad=4)
    ax.set_zlabel(r'$Q$', fontsize=12, labelpad=6)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=25, azim=-50)
    ax.tick_params(labelsize=9)
    z_ticks = [t for t in [-2, -1, 0] if Z_LOG_MIN - 0.1 <= t <= Z_LOG_MAX + 0.1]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f'$10^{{{t}}}$' for t in z_ticks], fontsize=9)


def _draw_static_curves(ax, u_arr, qdet_arr, qvarMS_arr, x_star, y_lo):
    ax.plot(u_arr, np.clip(qdet_arr,   y_lo, None),
            color=COL_DET,   lw=1.5, ls='--', zorder=2)
    ax.plot(u_arr, np.clip(qvarMS_arr, y_lo, None),
            color=COL_NOISY, lw=1.5, ls='-',  zorder=2)
    ax.axvline(x_star,             color=COL_DET,    lw=0.8, ls=':',  alpha=0.50, zorder=1)
    ax.axhline(FIDELITY_THRESHOLD, color=COL_THRESH, lw=1.0, ls='--', alpha=0.80, zorder=1)


def draw_2d_static(ax, u_arr, qdet_arr, qvarMS_arr, x_label, title, x_star):
    ax.set_facecolor(DARK_BG)
    _draw_static_curves(ax, u_arr, qdet_arr, qvarMS_arr, x_star, Y_MIN_LOG)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(Y_MIN_LOG, 1.3)
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('$Q$', fontsize=14)
    ax.set_title(title, fontsize=14, pad=6)
    ax.tick_params(direction='in', length=4, which='both')
    ax.grid(True, which='both', alpha=0.25, lw=0.6)


def draw_inset(ax_ins, u_arr, qdet_arr, qvarMS_arr, x_star,
               gp_slice, acc, u_rec_col, valid_mask, iter_rec_w):
    ax_ins.cla()
    ax_ins.set_facecolor('#2a2a2a')
    for sp in ax_ins.spines.values():
        sp.set_edgecolor('#777777')

    _draw_static_curves(ax_ins, u_arr, qdet_arr, qvarMS_arr, x_star, ZOOM_YMIN - 0.01)

    if gp_slice is not None:
        sl = gp_slice
        ax_ins.fill_between(
            sl['coord'],
            np.clip(sl['mu'] - sl['sigma'], ZOOM_YMIN - 0.01, None),
            np.clip(sl['mu'] + sl['sigma'], None, ZOOM_YMAX + 0.01),
            alpha=0.28, color=COL_GP, zorder=3,
        )
        ax_ins.plot(sl['coord'], sl['mu'], color=COL_GP, lw=1.5, zorder=4)

    if acc is not None and valid_mask.any():
        y_stars = acc[valid_mask, 11]
        in_range = (y_stars >= ZOOM_YMIN) & (y_stars <= ZOOM_YMAX)
        if in_range.any():
            ax_ins.scatter(
                acc[valid_mask][in_range, u_rec_col],
                y_stars[in_range],
                c=iter_rec_w[in_range], cmap=cmap_rec, vmin=0, vmax=1,
                s=22, marker='*', zorder=6,
            )

    ax_ins.set_xlim(-ZOOM_XHALF, ZOOM_XHALF)
    ax_ins.set_ylim(ZOOM_YMIN, ZOOM_YMAX)
    ax_ins.tick_params(labelsize=7, direction='in', length=3, colors=TEXT_COLOR)
    ax_ins.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
    ax_ins.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, prune='both'))


# ============================================================
# UPDATE
# ============================================================
def update(frame_idx):
    it   = iters[frame_idx]
    mask = acq_rec[:, 0] <= it
    acc  = acq_rec[mask]
    n_acq = len(acc)

    all_u1 = np.concatenate([init_u[:, 0], acc[:, 1] if n_acq else np.empty(0)])
    all_u2 = np.concatenate([init_u[:, 1], acc[:, 2] if n_acq else np.empty(0)])
    all_y  = np.concatenate([init_y,        acc[:, 4] if n_acq else np.empty(0)])

    x_rec         = acc[-1, 5:8] if n_acq else np.zeros(3)
    y_rec_cur_val = float(acc[-1, 11]) if (n_acq and not np.isnan(acc[-1, 11])) else Y_MIN_LOG

    # Iteration-based color weights: 0 = dark (early), 1 = bright (late)
    iter_init_w = np.zeros(n_init)
    iter_acq_w  = acc[:, 0] / max(iters[-1], 1) if n_acq else np.empty(0)
    iter_all_w  = np.concatenate([iter_init_w, iter_acq_w])

    valid_rec  = (~np.isnan(acc[:, 11])) if n_acq else np.zeros(0, dtype=bool)
    iter_rec_w = acc[valid_rec, 0] / max(iters[-1], 1) if n_acq and valid_rec.any() else np.empty(0)

    # ---- ax1: 3D scatter (log-z) --------------------------------
    ax1.cla()
    style_3d(ax1)
    z_all = np.log10(np.clip(all_y, 10**Z_LOG_MIN, 1.0))
    z_rec = np.log10(np.clip(y_rec_cur_val, 10**Z_LOG_MIN, 1.0))
    ax1.scatter(all_u1, all_u2, z_all,
                c=np.clip(all_y, 0, 1), cmap='plasma', vmin=0, vmax=1,
                s=18, depthshade=True, alpha=0.85)
    ax1.scatter([x_rec[0]], [x_rec[1]], [z_rec],
                color=COL_REC, s=160, marker='*', depthshade=False, zorder=10)
    z_thr = np.log10(FIDELITY_THRESHOLD)
    if Z_LOG_MIN <= z_thr <= Z_LOG_MAX + 0.05:
        xx, yy = np.meshgrid([-1, 1], [-1, 1])
        ax1.plot_surface(xx, yy, np.full_like(xx, z_thr),
                         alpha=0.15, color=COL_THRESH, zorder=0)

    # ---- ax2: u2 slice (u1=u3=0) --------------------------------
    ax2.cla()
    draw_2d_static(ax2, s2_u, s2_qdet, s2_qvarMS,
                   r'$f_\mathrm{sb}$',
                   r'Slice $\Delta\Omega = \Delta f_\mathrm{cl} = 0$',
                   u2_star)
    if it in gp_u2:
        sl = gp_u2[it]
        ax2.fill_between(sl['coord'],
                         np.clip(sl['mu'] - sl['sigma'], Y_MIN_LOG, None),
                         np.clip(sl['mu'] + sl['sigma'], Y_MIN_LOG, None),
                         alpha=0.28, color=COL_GP, zorder=3)
        ax2.plot(sl['coord'], np.clip(sl['mu'], Y_MIN_LOG, None),
                 color=COL_GP, lw=2, zorder=4)
    ax2.scatter(all_u2, np.clip(all_y, Y_MIN_LOG, None),
                c=iter_all_w, cmap=cmap_acq, vmin=0, vmax=1,
                s=18, alpha=0.85, zorder=5)
    if n_acq and valid_rec.any():
        ax2.scatter(acc[valid_rec, 6], np.clip(acc[valid_rec, 11], Y_MIN_LOG, None),
                    c=iter_rec_w, cmap=cmap_rec, vmin=0, vmax=1,
                    s=30, marker='*', zorder=6)
        ax2.axvline(x_rec[1], color=COL_REC, lw=1.2, ls='--', alpha=0.7, zorder=7)

    draw_inset(ax2_inset, s2_u, s2_qdet, s2_qvarMS, u2_star,
               gp_u2.get(it), acc if n_acq else None,
               6, valid_rec, iter_rec_w)

    # ---- ax3: u1 slice (u2=u3=0) --------------------------------
    ax3.cla()
    draw_2d_static(ax3, s3_u, s3_qdet, s3_qvarMS,
                   r'$f_\mathrm{cl}$',
                   r'Slice $\Delta\Omega = \Delta f_\mathrm{sb} = 0$',
                   u1_star)
    if it in gp_u1:
        sl = gp_u1[it]
        ax3.fill_between(sl['coord'],
                         np.clip(sl['mu'] - sl['sigma'], Y_MIN_LOG, None),
                         np.clip(sl['mu'] + sl['sigma'], Y_MIN_LOG, None),
                         alpha=0.28, color=COL_GP, zorder=3)
        ax3.plot(sl['coord'], np.clip(sl['mu'], Y_MIN_LOG, None),
                 color=COL_GP, lw=2, zorder=4)
    ax3.scatter(all_u1, np.clip(all_y, Y_MIN_LOG, None),
                c=iter_all_w, cmap=cmap_acq, vmin=0, vmax=1,
                s=18, alpha=0.85, zorder=5)
    if n_acq and valid_rec.any():
        ax3.scatter(acc[valid_rec, 5], np.clip(acc[valid_rec, 11], Y_MIN_LOG, None),
                    c=iter_rec_w, cmap=cmap_rec, vmin=0, vmax=1,
                    s=30, marker='*', zorder=6)
        ax3.axvline(x_rec[0], color=COL_REC, lw=1.2, ls='--', alpha=0.7, zorder=7)
    ax3.set_ylabel('')
    ax3.set_yticklabels([])

    draw_inset(ax3_inset, s3_u, s3_qdet, s3_qvarMS, u1_star,
               gp_u1.get(it), acc if n_acq else None,
               5, valid_rec, iter_rec_w)

    # ---- Progress bar: gray gradient fill -----------------------
    prog_arr = np.zeros((1, n_frames))
    if frame_idx >= 0:
        prog_arr[0, :frame_idx + 1] = np.linspace(1.0 / n_frames, 1.0, n_frames)[:frame_idx + 1]
    prog_img.set_data(prog_arr)

    return []


# ============================================================
# BUILD AND SAVE
# ============================================================
ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 // FPS, blit=False)

print(f"Saving {n_frames}-frame animation -> {OUTPUT_FILE}")
ani.save(OUTPUT_FILE, writer='pillow', fps=FPS, dpi=90)
print("Done.")
plt.show()
