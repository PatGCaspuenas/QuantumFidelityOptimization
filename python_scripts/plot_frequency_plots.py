import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.ndimage import gaussian_filter
import os
import matplotlib.colors as mc

import colorsys


# ── settings ────────────────────────────────────────────────────────────────
n_bins   = 30    # histogram grid resolution
sigma    = 1.2   # Gaussian smoothing sigma
bg_color = 'white'

my_husl_blue = sns.husl_palette(n_colors=1, h=0.72, s=0.9, l=0.6)[0]
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
# Now you can use it exactly as you did before:
vibrant_blue = adjust_lightness(my_husl_blue, 0.9)

n_initial = 1    # Drop the first initial points if identical across runs

slices_config = [
    {'name': 'u1 - u2', 'file': 'data/slices/slice_u1u2.csv', 'x_col': 'u1', 'y_col': 'u2'},
    {'name': 'u1 - u3', 'file': 'data/slices/slice_u1u3.csv', 'x_col': 'u1', 'y_col': 'u3'},
    {'name': 'u2 - u3', 'file': 'data/slices/slice_u2u3.csv', 'x_col': 'u2', 'y_col': 'u3'}
]

label_map = {
    'u1': r'$\Delta f_{cl}$', 
    'u2': r'$\Delta f_{sb}$', 
    'u3': r'$\Delta \Omega$'
}

row_labels = [r'estimated max' + '\n' + r'($\mathbf{x}_n^*$)', 
              r'samples' + '\n' + r'($\mathbf{x}_{n+1}$)']
keys = ['x_rec', 'x_acq'] 

clevels_base = np.linspace(0.1, 0.9, 5) 
thick_level = [0.999]  

# ── Pass 1: load data and compute per-run visit counts ───────────────────────
ucb_files = glob.glob('data/traces/trace_seed*_ucb.csv')
if not ucb_files:
    print("No UCB trace files found.")
dfs = [pd.read_csv(f) for f in ucb_files] if ucb_files else []
n_runs = max(len(dfs), 1)

edges = np.linspace(-1, 1, n_bins + 1)

all_densities = {
    'x_rec': [None, None, None],
    'x_acq': [None, None, None]
}

for col_idx, s_conf in enumerate(slices_config):
    x_axis = s_conf['x_col']
    y_axis = s_conf['y_col']
    
    for key in keys:
        H_runs = np.zeros((n_bins, n_bins))
        for df in dfs:
            pts_x = df[f'{key}_{x_axis}'].values[n_initial:]
            pts_y = df[f'{key}_{y_axis}'].values[n_initial:]
            H_r, _, _ = np.histogram2d(pts_x, pts_y, bins=[edges, edges])
            H_runs += (H_r > 0).astype(float)
            
        H_smooth = gaussian_filter(H_runs.T.astype(float), sigma=sigma)
        all_densities[key][col_idx] = H_smooth

# ── Pass 2: plot ─────────────────────────────────────────────────────────────
vmin_counts = 1.0
vmax_counts = max(100.0, float(n_runs))
shared_norm = LogNorm(vmin=vmin_counts, vmax=vmax_counts)

fig, axes = plt.subplots(2, 3, subplot_kw=dict(box_aspect=1),
                         figsize=(10, 6), layout='compressed')

for i, (key, row_label) in enumerate(zip(keys, row_labels)):
    for j, s_conf in enumerate(slices_config):
        ax_j = axes[i, j]

        x_col = s_conf['x_col']
        y_col = s_conf['y_col']
        extent = [-1, 1, -1, 1]

        sub_cmap = LinearSegmentedColormap.from_list('freq_blue', [bg_color, vibrant_blue])
        sub_cmap.set_under(bg_color)
        ax_j.set_facecolor(bg_color)

        H_smooth = all_densities[key][j]
        if H_smooth is not None:
            ax_j.imshow(H_smooth, origin='lower', aspect='auto',
                        extent=extent, cmap=sub_cmap, norm=shared_norm, zorder=1)

        slice_file = s_conf['file']
        if os.path.exists(slice_file):
            df_slice = pd.read_csv(slice_file, comment='#')
            n_grid = int(np.sqrt(len(df_slice)))
            
            X_map = df_slice[x_col].values.reshape(n_grid, n_grid)
            Y_map = df_slice[y_col].values.reshape(n_grid, n_grid)
            Z_map = df_slice['q'].values.reshape(n_grid, n_grid)
            
            cs_base = ax_j.contour(X_map, Y_map, Z_map, levels=clevels_base, 
                                   colors='#777777', linewidths=0.6, alpha=0.9, zorder=2)
            ax_j.clabel(cs_base, inline=True, fontsize=8, fmt='%.1f')
            
            cs_thick = ax_j.contour(X_map, Y_map, Z_map, levels=thick_level, 
                                    colors='#333333', linewidths=2.0, zorder=3)
            ax_j.clabel(cs_thick, inline=True, fontsize=9, fmt='%.3f')

        ax_j.set_xlim([-1, 1])
        ax_j.set_ylim([-1, 1])
        
        ticks = [-1, -0.5, 0, 0.5, 1]
        ax_j.set_xticks(ticks)
        ax_j.set_yticks(ticks)
        
        # All axis labels always shown
        ax_j.set_xlabel(label_map[x_col], fontsize=12)
        ax_j.set_ylabel(label_map[y_col], fontsize=12)
        
        # Inner/Outer Tick Logic
        ax_j.tick_params(axis='both', direction='in', length=5, labelsize=12)
        
        # Hide tick LABELS everywhere except bottom-left (row i=1, col j=0)
        if not (i == 1 and j == 0):
            ax_j.tick_params(labelbottom=False, labelleft=False)

        ax_j.grid(True, which='both', ls=':', alpha=0.5, color='gray')

        # Add Row Label on the far right of the 3rd column
        if j == 2:
            ax_right = ax_j.twinx()
            ax_right.set_ylabel(row_label, fontsize=12, rotation=-90, labelpad=40)
            ax_right.set_yticks([]) 
            for spine in ax_right.spines.values():
                spine.set_visible(False)

# ── Shared colorbar ───────────────────────────────────────────────

fig.align_ylabels(axes[:, 0])
fig.align_xlabels(axes[1, :])

gray_cmap = LinearSegmentedColormap.from_list('blue_freq', [bg_color, vibrant_blue])
gray_cmap.set_under(bg_color)
sm = plt.cm.ScalarMappable(cmap=gray_cmap, norm=shared_norm)
sm.set_array([])

cb = fig.colorbar(sm, ax=axes, location='bottom', shrink=0.35, aspect=30, pad=0.04,
                  ticks=[1, 10, 100])
cb.ax.set_xticklabels(['0.01', '0.1', '1']) 
cb.set_label('Sampling Frequency', rotation=0, labelpad=5, fontsize=12)
cb.ax.tick_params(labelsize=12)

fig.savefig('ucb_slices_heatmap_final.png', bbox_inches='tight', dpi=300)
plt.show()