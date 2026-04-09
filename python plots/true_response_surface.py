import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# User's Plotting Style
# ============================================================
sns.set_theme(font_scale=1, style='whitegrid', font='serif')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.rcParamsDefault['axes.prop_cycle'].by_key()['color'])

mpl.rcParams['lines.linewidth'] = 1.3
mpl.rcParams['axes.linewidth'] = 1.3
mpl.rcParams['legend.fontsize'] = 12

mpl.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'legend.edgecolor': 'black',
    'legend.facecolor': '#ffffff',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
})

# ============================================================
# Data Processing Helpers
# ============================================================
def get_mesh(df, target_col, plane, slice_val):
    sub = df[(df['plane'] == plane) & (np.isclose(df['slice_fixed_val'], slice_val))]
    
    res = int(np.sqrt(len(sub)))
    if res * res != len(sub):
        raise ValueError(f"Data for plane {plane} at {slice_val} is not a perfect square grid.")

    if plane == 'XY':
        sub = sub.sort_values(by=['u2_fsb', 'u1_fcl'])
    elif plane == 'XZ':
        sub = sub.sort_values(by=['u3_A', 'u1_fcl'])
    elif plane == 'YZ':
        sub = sub.sort_values(by=['u3_A', 'u2_fsb'])

    X = sub['u1_fcl'].values.reshape(res, res)
    Y = sub['u2_fsb'].values.reshape(res, res)
    Z = sub['u3_A'].values.reshape(res, res)
    C = sub[target_col].values.reshape(res, res)
    
    return X, Y, Z, C

# ============================================================
# Main Plotting Logic
# ============================================================
def plot_3d_slices(csv_filepath):
    csv_path = Path(csv_filepath)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    
    targets = ['Q_det', 'Q_varMS']
    planes = ['XY', 'XZ', 'YZ']
    slices = sorted(df['slice_fixed_val'].unique())

    ELEV = 20
    AZIM = 45

    for target in targets:
        fig = plt.figure(figsize=(25, 14), constrained_layout=True)
        
        # Determine dynamic vmin and label based on the target
        if target == 'Q_det':
            vmin_val = 1e-1
            cbar_label = r'$Q_{det}$ ($\log_{10}$ scale)'
        else:
            vmin_val = 1e-2
            cbar_label = r'$Q_{var,MS}$ ($N=1000$) ($\log_{10}$ scale)'
        
        norm = mpl.colors.LogNorm(vmin=vmin_val, vmax=1.0)
        cmap = 'turbo' 

        for i, main_plane in enumerate(planes):
            for j, val in enumerate(slices):
                idx = i * len(slices) + j + 1
                ax = fig.add_subplot(3, 5, idx, projection='3d')
                ax.grid(False)
                
                # Zoom out the 3D box inside the subplot frame to make room for labels
                try:
                    ax.set_box_aspect(None, zoom=0.85)
                except AttributeError:
                    ax.dist = 11

                # 1. Plot the center slices of the OTHER planes
                other_planes = [p for p in planes if p != main_plane]
                for p in other_planes:
                    try:
                        X_bg, Y_bg, Z_bg, C_bg = get_mesh(df, target, p, 0.0)
                        C_bg_clipped = np.clip(C_bg, vmin_val, 1.0)
                        ax.plot_surface(X_bg, Y_bg, Z_bg, facecolors=plt.get_cmap(cmap)(norm(C_bg_clipped)), 
                                        alpha=0.15, rstride=1, cstride=1, antialiased=False, 
                                        linewidth=0, shade=False)
                    except ValueError:
                        pass
                
                # 2. Plot the MAIN slice of interest
                X_main, Y_main, Z_main, C_main = get_mesh(df, target, main_plane, val)
                C_main_clipped = np.clip(C_main, vmin_val, 1.0)
                ax.plot_surface(X_main, Y_main, Z_main, facecolors=plt.get_cmap(cmap)(norm(C_main_clipped)), 
                                alpha=1.0, rstride=1, cstride=1, antialiased=False, 
                                linewidth=0, shade=False)

                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
                
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([-1, 0, 1])
                ax.set_zticks([-1, 0, 1])
                
                if i == 2 and j == 0:
                    ax.set_xlabel('$u_1$ (f_cl)', labelpad=5)
                    ax.set_ylabel('$u_2$ (f_sb)', labelpad=5)
                    ax.set_zlabel('$u_3$ (A)', labelpad=5) 
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])

                ax.view_init(elev=ELEV, azim=AZIM)

        # Horizontal colorbar at the top center
        # Lowered the Y position slightly (from 0.92 to 0.90) to leave room for the title above it
        cbar_ax = fig.add_axes([0.3, 0.90, 0.4, 0.025])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Plot the colorbar without the default side label
        cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        
        # Add the label as a title to the colorbar axis to put it cleanly on top
        cb.ax.set_title(cbar_label, pad=12, fontsize=14)

        # Dropped the top margin slightly (0.85 -> 0.82) so the plots don't clip the new top label
        plt.subplots_adjust(left=0.15, right=0.98, top=0.82, bottom=0.12, wspace=0.05, hspace=0.1)
        
        output_file = Path(__file__).parent / 'figures' / f'response_surface_3D_{target}_log.png'
        output_file.parent.mkdir(exist_ok=True)
        
        fig.savefig(output_file, dpi=300, pad_inches=0.4, transparent=False)
        print(f"Saved figure to {output_file}")
        
        plt.close(fig)

if __name__ == '__main__':
    csv_file = Path(__file__).parent / 'data' / 'response_surface_slices_N1000.csv'
    plot_3d_slices(csv_file)