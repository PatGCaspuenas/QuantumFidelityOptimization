import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mc
import colorsys

def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# Just UCB
methods = ['ucb']

my_husl_blue = sns.husl_palette(n_colors=1, h=0.72, s=0.9, l=0.6)[0]

# Now you can use it exactly as you did before:
color_main = adjust_lightness(my_husl_blue, 0.9)

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True, subplot_kw={'box_aspect': 1})

for method in methods:
    files = glob.glob(f"data/traces/trace_seed*_{method}.csv")
    if not files:
        print(f"No files found for method: {method}")
        continue
        
    dfs = [pd.read_csv(f) for f in files]
    
    # Find the absolute minimum and maximum iteration across ALL seeds
    min_iter = int(min(df['iter'].min() for df in dfs))
    max_iter = int(max(df['iter'].max() for df in dfs))
    n_range = np.arange(min_iter, max_iter + 1)
    
    pooled_samples = []
    
    for idx, df in enumerate(dfs):
        # ─── CHANGED HERE ────────────────────────────────────────────────────────
        # Align to global index and forward-fill the last known values
        df_aligned = df.set_index('iter').reindex(n_range).ffill()
        # ─────────────────────────────────────────────────────────────────────────
        
        m = 1.0 - df_aligned['m_rec'].values
        s = df_aligned['s_rec'].values
        
        # Mask to identify valid entries (in case any seeds START late)
        valid = ~np.isnan(m) & ~np.isnan(s)
        
        samples = np.full((500, len(n_range)), np.nan)
        
        if np.any(valid):
            # Sample only for valid indices
            np.random.seed(42 + idx) 
            valid_samples = np.random.normal(loc=m[valid], scale=s[valid], size=(500, np.sum(valid)))
            valid_samples = np.clip(valid_samples, 1e-10, 2.0)
            samples[:, valid] = valid_samples
            
        pooled_samples.append(samples)
        
    metric_data = np.vstack(pooled_samples)
    
    # Calculate median (early stopping seeds are now naturally represented by their final values)
    m_median = np.nanmedian(metric_data, axis=0)
    
    n_layers = 15
    quantiles = np.linspace(0.05, 0.45, n_layers)
    base_alpha = 0.5 / n_layers
    
    for q in quantiles:
        m_lower = np.nanquantile(metric_data, q, axis=0)
        m_upper = np.nanquantile(metric_data, 1 - q, axis=0)
        ax.fill_between(n_range, m_lower, m_upper,
                        color=color_main, alpha=base_alpha,
                        linewidth=0, edgecolor='none', zorder=1)

    ax.plot(n_range, m_median, '-', c=color_main, lw=2.5, zorder=2)
    
    ci_lo = np.nanquantile(metric_data, 0.025, axis=0)
    ci_hi = np.nanquantile(metric_data, 0.975, axis=0)

ax.set_yscale('log')
ax.set_ylim(1e-6, 1e0)
ax.set_xlim(0, max_iter)
ax.set_xlabel('Iteration $n$', fontsize=14)
ax.set_ylabel(r'$1 - \hat{\mu}(\mathbf{x}^*_n)$', fontsize=14)
ax.grid(True, which='both', ls=':', alpha=0.5)
ax.tick_params(axis='both', direction='in', length=5, labelsize=12)

fig.savefig('comparison_ucb_only_ffill.png', dpi=300)
plt.show()