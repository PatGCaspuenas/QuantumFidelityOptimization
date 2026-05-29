import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mc
import colorsys
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
import os

# --- Helper for Colors ---
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def create_color_list(start_color, end_color, n_colors=10):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])
    n = max(n_colors, 2)
    return [cmap(i/(n-1)) for i in range(n)]

# --- Configuration ---
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

# Colors: HUSL Blue
my_husl_blue = sns.husl_palette(n_colors=1, h=0.72, s=0.9, l=0.6)[0]
color_main = adjust_lightness(my_husl_blue, 0.9)
edge_color = adjust_lightness(my_husl_blue, 0.6) 

c_start, c_end = "#ADD8E6", color_main 

# True Max Coordinates in the plotted contour frame.
# Jittered runs shift the simulated optimum internally, but these plots keep
# the marker at the contour center so the background landscape stays readable.
true_max = {'u1': 0.0, 'u2': 0.0, 'u3': 0.0}

# ── 1. Select Seed ───────────────────────────────────────────────────────────
trace_glob = os.environ.get('UCB_TRACE_GLOB', 'data/traces/trace_seed*_ucb.csv')
ucb_files = sorted(glob.glob(trace_glob))
if not ucb_files:
    raise ValueError(f"No UCB files found for glob: {trace_glob}")

selected_seed = os.environ.get('UCB_TRACE_SEED')
if selected_seed:
    matches = [f for f in ucb_files if f'trace_seed{selected_seed}_ucb.csv' in f]
    if not matches:
        raise ValueError(f"No UCB trace file found for seed {selected_seed}")
    selected_file = matches[0]
    df_selected = pd.read_csv(selected_file)
    output_file = os.environ.get('UCB_POINTS_OUT', f'ucb_seed{selected_seed}_points_aligned.png')
    print(f"Selected seed {selected_seed}: {selected_file}")
else:
    dfs = [pd.read_csv(f) for f in ucb_files]
    final_m_rec = [df['m_rec'].iloc[-1] for df in dfs]
    median_val = np.nanmedian(final_m_rec)
    median_idx = np.nanargmin(np.abs(np.array(final_m_rec) - median_val))
    selected_file = ucb_files[median_idx]
    df_selected = dfs[median_idx]
    output_file = os.environ.get('UCB_POINTS_OUT', 'ucb_median_points_aligned.png')
    print(f"Selected median final m_rec trace: {selected_file}")

n_initial = 1
df_selected = df_selected.iloc[n_initial:].dropna(subset=['x_rec_u1', 'x_acq_u1']).copy()
n_points = len(df_selected)
colors = create_color_list(c_start, c_end, n_colors=n_points)

# ── 2. Plotting ──────────────────────────────────────────────────────────────
# 1) Reduced width to 12.0 to bring columns closer natively
fig, axes = plt.subplots(2, 3, subplot_kw=dict(box_aspect=1),
                         figsize=(10, 6), layout='compressed')

# 1) Reduced wspace to 0.08 to tightly pack the columns
# plt.subplots_adjust(bottom=0.22, hspace=0.25, wspace=0.08)

zoom_delta = 0.1 
axins_list = []

for i, (key, row_label) in enumerate(zip(keys, row_labels)):
    for j, s_conf in enumerate(slices_config):
        ax_j = axes[i, j]
        
        x_col = s_conf['x_col']
        y_col = s_conf['y_col']
        
        # ── Draw Background Contours & Labels ──
        slice_file = s_conf['file']
        if os.path.exists(slice_file):
            df_slice = pd.read_csv(slice_file, comment='#')
            n_grid = int(np.sqrt(len(df_slice)))
            
            X_map = df_slice[x_col].values.reshape(n_grid, n_grid)
            Y_map = df_slice[y_col].values.reshape(n_grid, n_grid)
            Z_map = df_slice['q'].values.reshape(n_grid, n_grid)
            
            cs_base = ax_j.contour(X_map, Y_map, Z_map, levels=clevels_base, 
                                   colors='#777777', linewidths=0.6, alpha=0.9, zorder=1)
            ax_j.clabel(cs_base, inline=True, fontsize=8, fmt='%.1f')
            
            cs_thick = ax_j.contour(X_map, Y_map, Z_map, levels=thick_level, 
                                    colors='#333333', linewidths=2.0, zorder=1)
            ax_j.clabel(cs_thick, inline=True, fontsize=9, fmt='%.3f')
            
        # ── Plot True Max ──
        ax_j.plot(true_max[x_col], true_max[y_col], '*', color='white', 
                  markeredgecolor=edge_color, markersize=15, zorder=5)
        
        # ── Setup zoom inset for top row (x_rec) ──
        if i == 0:
            axins = ax_j.inset_axes([0.55, 0.55, 0.4, 0.4])
            if os.path.exists(slice_file):
                cs_ins = axins.contour(X_map, Y_map, Z_map, levels=clevels_base, colors='#777777', linewidths=0.5, zorder=1)
                axins.clabel(cs_ins, inline=True, fontsize=6, fmt='%.1f')
                cs_ins_thick = axins.contour(X_map, Y_map, Z_map, levels=thick_level, colors='#333333', linewidths=1.5, zorder=1)
            
            axins.plot(true_max[x_col], true_max[y_col], '*', color='white', 
                       markeredgecolor=edge_color, markersize=18, zorder=5)
            
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            ax_j.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.5)
            axins_list.append(axins)
            
        # ── Extract and Plot Points ──
        pts_x = df_selected[f'{key}_{x_col}'].values
        pts_y = df_selected[f'{key}_{y_col}'].values
        
        for k in range(n_points):
            if i == 0: 
                if k == n_points - 1:
                    face_color, pt_edge, z_level = 'red', edge_color, 20
                    main_size, inset_size = 12, 16
                else:
                    face_color, pt_edge, z_level = colors[k], edge_color, 10
                    main_size, inset_size = 8, 10
                    
                ax_j.plot(pts_x[k], pts_y[k], '*', c=face_color, markersize=main_size,
                          clip_on=True, zorder=z_level, markeredgecolor=pt_edge, markeredgewidth=0.8)
                axins_list[j].plot(pts_x[k], pts_y[k], '*', c=face_color, markersize=inset_size,
                                   clip_on=True, zorder=z_level, markeredgecolor=pt_edge, markeredgewidth=0.8)
            else: 
                ax_j.plot(pts_x[k], pts_y[k], 'o', c=colors[k], markersize=6,
                          clip_on=True, zorder=10, markeredgecolor=edge_color, markeredgewidth=0.6)

        # Finalize Zoom Bounds
        if i == 0:
            last_x, last_y = pts_x[-1], pts_y[-1]
            axins_list[j].set_xlim(last_x - zoom_delta, last_x + zoom_delta)
            axins_list[j].set_ylim(last_y - zoom_delta, last_y + zoom_delta)

        # ── Formatting ──
        ax_j.set_xlim([-1, 1])
        ax_j.set_ylim([-1, 1])
        ticks = [-1, -0.5, 0, 0.5, 1]
        ax_j.set_xticks(ticks)
        ax_j.set_yticks(ticks)
        
        if i == 1:
            ax_j.set_xlabel(label_map[x_col], fontsize=12)
        ax_j.set_ylabel(label_map[y_col], fontsize=12)
            
        ax_j.tick_params(axis='both', direction='in', length=5, labelsize=12)
        
        if not (i == 1 and j == 0):
            ax_j.tick_params(labelbottom=False, labelleft=False)

        ax_j.grid(True, which='both', ls=':', alpha=0.5, color='gray')

        if j == 2:
            ax_right = ax_j.twinx()
            ax_right.set_ylabel(row_label, fontsize=12, rotation=-90, labelpad=40)
            ax_right.set_yticks([]) 
            for spine in ax_right.spines.values():
                spine.set_visible(False)

# ── 3. Alignment, Colorbar & Legend ──────────────────────────────────────────

fig.align_ylabels(axes[:, 0])
fig.align_xlabels(axes[1, :])

# 1. Define the legend elements separately
legend_element_1 = Line2D([0], [0], marker='*', color='w', label='True Max',
                          markerfacecolor='white', markeredgecolor=edge_color, markersize=14, markeredgewidth=1.2)
legend_element_2 = Line2D([0], [0], marker='*', color='w', label='Last Estimated Max',
                          markerfacecolor='red', markeredgecolor=edge_color, markersize=12, markeredgewidth=1.2)

cmap = LinearSegmentedColormap.from_list("custom_cbar", [c_start, c_end])
colorbar_max = max(50, n_points)
mnorm = mpl.colors.Normalize(vmin=0, vmax=colorbar_max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mnorm)
sm.set_array([])

cb_ticks = [0, colorbar_max / 2, colorbar_max]
cb = fig.colorbar(sm, ax=axes, location='bottom', shrink=0.35, aspect=30, pad=0.05, ticks=cb_ticks)
cb.ax.set_title('Iteration ($n$)', fontsize=12, pad=10)
cb.ax.tick_params(labelsize=12)

# 2. Anchor the first legend to the left side of the colorbar (x = -0.08)
leg1 = cb.ax.legend(handles=[legend_element_1], loc='center right', bbox_to_anchor=(-0.08, 0.5), fontsize=12, frameon=False)
cb.ax.add_artist(leg1) # Required so the second legend doesn't overwrite the first

# 3. Anchor the second legend to the right side of the colorbar (x = 1.08)
cb.ax.legend(handles=[legend_element_2], loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=12, frameon=False)

fig.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"Saved {output_file}")
plt.show()
