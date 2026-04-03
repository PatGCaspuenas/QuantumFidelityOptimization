import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(font_scale=1, style='whitegrid', font='serif')
mpl.rcParams['mathtext.fontset'] = 'cm' # Computer Modern (LaTeX default)
mpl.rcParams['font.family'] = 'serif' # LaTeX-like font
mpl.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.rcParamsDefault['axes.prop_cycle'].by_key()['color'])

mpl.rcParams['lines.linewidth'] = 1.3
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['axes.linewidth'] = 1.3

mpl.rcParams['legend.loc'] = 'upper left'
mpl.rcParams['legend.fontsize'] = 12

mpl.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f5f3ee',         # Light beige background
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'legend.edgecolor': 'black',
    'legend.facecolor': '#ffffff',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'gray',
    'text.color': 'black',
})

df = pd.read_csv("data/compare_sigma_models.txt", sep=r'\s+')

# Color palette by N (shared across both subplots)
N_vals = sorted(df['N'].unique())
palette = sns.color_palette("husl", len(N_vals))
color_map = dict(zip(N_vals, palette))

fig, axes = plt.subplots(1, 2, figsize=(8, 5), subplot_kw={'box_aspect': 1})
fig.subplots_adjust(bottom=0.22)

# --- Plot 1: Heteroscedasticity (Q_mean vs Sigma Empirical) ---
for n_val in N_vals:
    subset = df[df['N'] == n_val]
    axes[0].scatter(subset['Q_mean'], subset['σ_empirical'],
                    color=color_map[n_val], s=30, zorder=3, clip_on=False)

# Theoretical curves over full [x_min, 1] range
x_min1 = df['Q_mean'].min()
x_vals = np.linspace(x_min1, 1, 200)
for n_val in N_vals:
    y_vals = np.sqrt(x_vals * (1 - x_vals) / n_val)
    axes[0].plot(x_vals, y_vals, color=color_map[n_val], alpha=0.6)

axes[0].set_xlim(x_min1, 1)
axes[0].set_ylim(df['σ_empirical'].min(), 0.1)
axes[0].margins(0)

# Subplot 1 legend: one generic sample point + one theoretical line
sample_handle = mlines.Line2D([], [], color='dimgray', marker='o', linestyle='None',
                               markersize=5, label='Samples')
theory_handle = mlines.Line2D([], [], color='dimgray', linestyle='-', alpha=0.6,
                               label='Theoretical')
axes[0].legend(handles=[sample_handle, theory_handle], loc='upper right')

axes[0].set_xlabel("$\\overline{Q}_{MS}$", fontsize=14)
axes[0].set_ylabel("$\\sigma_{empirical}$", fontsize=14)
axes[0].tick_params(axis='both', direction='in', length=5,
                    top=False, bottom=True, left=True, right=False, labelsize=14)
axes[0].grid(True, which='both', ls='-', alpha=0.5)

# --- Plot 2: Predicted Sigma vs Empirical Sigma (axes flipped: x=predicted, y=empirical) ---
for n_val in N_vals:
    subset = df[df['N'] == n_val]
    axes[1].scatter(subset['σ_simple'], subset['σ_empirical'],
                    color=color_map[n_val], alpha=0.8, s=30, marker='o', clip_on=False)
    axes[1].scatter(subset['σ_correct'], subset['σ_empirical'],
                    color=color_map[n_val], alpha=0.8, s=30, marker='^', clip_on=False)

# Both axes share the same [0, 0.1] range for a square equal plot
axes[1].set_xlim(0, 0.1)
axes[1].set_ylim(0, 0.1)
axes[1].margins(0)
axes[1].set_aspect('equal', adjustable='box')

# Ideal match diagonal
axes[1].plot([0, 0.1], [0, 0.1], 'k--', alpha=0.6, label="Ideal")

# Xticks match yticks (same range, explicitly synced)
axes[1].set_xticks(axes[1].get_yticks())
axes[1].set_xlim(0, 0.1)  # restore after get_yticks may expand limits

# Subplot 2 legend: marker shapes + ideal match line
simple_handle = mlines.Line2D([], [], color='dimgray', marker='o', linestyle='None',
                               markersize=5, label='Simple ($1/\\sqrt{N}$)')
correct_handle = mlines.Line2D([], [], color='dimgray', marker='^', linestyle='None',
                                markersize=5, label=r'Binomial ($\sqrt{Q(1-Q)/N}$)')
ideal_handle = mlines.Line2D([], [], color='black', linestyle='--', alpha=0.6,
                              label='Ideal')
axes[1].legend(handles=[simple_handle, correct_handle, ideal_handle], loc='upper right')

axes[1].tick_params(axis='both', direction='in', length=5,
                    top=False, bottom=True, left=True, right=False, labelsize=14)
axes[1].grid(True, which='both', ls='-', alpha=0.5)
axes[1].set_yticklabels([])

# --- Shared legend at the bottom: colors for N ---
color_handles = [mpatches.Patch(color=color_map[n_val], label=f'{n_val}')
                 for n_val in N_vals]
fig.legend(handles=color_handles, loc='lower center', ncol=len(N_vals),
           title='Number of shots $N$', bbox_to_anchor=(0.5, 0.01),
           frameon=True, fontsize=12, title_fontsize=12)

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.show()
