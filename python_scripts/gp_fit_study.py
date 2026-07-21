import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ====================================================================
# 1. Violin Plots (Error Trends)
# ====================================================================
# Load the main metrics dataset
df = pd.read_csv('data/gp_fit_quality_3d_2ms_v1.csv')

sns.set_theme(style="whitegrid", font_scale=1.2)
fig, ax = plt.subplots(figsize=(12, 6))

# Filter for fixed mode to see spatial/shot convergence
df_fixed = df[df['mode'] == 'fixed'].copy()

sns.violinplot(
    data=df_fixed, 
    x='N_shots', 
    y='rmse_corr', 
    hue='n_pts', 
    palette='viridis', 
    inner='quartile',
    cut=0, 
    linewidth=1.2,
    ax=ax
)

ax.set_title('GP Fit Error (Corrected RMSE) vs. Shots and Training Points', fontsize=16, pad=15)
ax.set_xlabel('Shots per Point ($N$)', fontsize=14)
ax.set_ylabel('Corrected RMSE', fontsize=14)
ax.set_yscale('log')
ax.legend(title='Training Pts ($n_{pts}$)', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
fig.savefig('violin_trends.png', dpi=300)
plt.show()

# ====================================================================
# 2. 1D Slices Plot (GP Approximation)
# ====================================================================
# Load the slices dataset
try:
    df_slices = pd.read_csv('data/slices_output.csv')
    
    # Example: We want to plot the f_cl slice for Fixed mode, N_shots = 1000
    target_shots = 1000
    slice_ax = 'fcl'  # Can also be 'fsb'
    
    # Filter dataset
    df_plot = df_slices[
        (df_slices['mode'] == 'fixed') & 
        (df_slices['N_shots'] == target_shots) & 
        (df_slices['slice_axis'] == slice_ax)
    ]
    
    # Get the unique point counts available
    pts_list = sorted(df_plot['n_pts'].unique())
    colors = sns.color_palette("husl", len(pts_list))

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, pts in enumerate(pts_list):
        subset = df_plot[df_plot['n_pts'] == pts].sort_values('u')
        
        u = subset['u'].values
        mu = subset['mu'].values
        sigma = subset['sigma'].values
        
        # Plot GP Mean
        ax.plot(u, mu, label=f'GP ({pts} pts)', color=colors[idx], linewidth=2)
        # Plot 2-Sigma Uncertainty Ribbon
        ax.fill_between(u, mu - 2*sigma, mu + 2*sigma, color=colors[idx], alpha=0.15)

    ax.set_title(f'1D Slice Prediction: {slice_ax} axis (Fixed shots = {target_shots})', fontsize=16)
    ax.set_xlabel('Normalized Coordinate ($u$)', fontsize=14)
    ax.set_ylabel('Fidelity ($Q_{varMS}$)', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower center', ncol=len(pts_list), frameon=False, bbox_to_anchor=(0.5, -0.2))

    plt.tight_layout()
    fig.savefig(f'data/1d_slice_{slice_ax}.png', dpi=300, bbox_inches='tight')
    plt.show()

except FileNotFoundError:
    print("slices_output.csv not found in the directory. Run the Julia script to generate it!")