import glob
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    trace_glob = os.environ.get('UCB_TRACE_GLOB', f"data/traces/trace_seed*_{method}.csv")
    files = sorted(glob.glob(trace_glob))
    if not files:
        print(f"No files found for method: {method}")
        continue

    dfs = []
    used_files = []
    only_converged = os.environ.get('UCB_ONLY_CONVERGED', '').lower() in ('1', 'true', 'yes', 'on')
    forward_fill = os.environ.get('UCB_FORWARD_FILL', 'true').lower() in ('1', 'true', 'yes', 'on')
    trend_metric = os.environ.get('UCB_TREND_METRIC', 'surrogate').lower()
    q_clip_max = float(os.environ.get('UCB_Q_CLIP_MAX', '1.0'))
    for f in files:
        df = pd.read_csv(f)
        if only_converged and 'stopped_early' in df.columns and not bool(df['stopped_early'].iloc[-1]):
            continue
        dfs.append(df)
        used_files.append(f)

    if not dfs:
        print(f"No trace files left for method {method} after filtering.")
        continue

    print(f"Using {len(dfs)} {method} traces")
    
    # Find the absolute minimum and maximum iteration across ALL seeds
    min_iter = int(min(df['iter'].min() for df in dfs))
    max_iter = int(max(df['iter'].max() for df in dfs))
    n_range = np.arange(min_iter, max_iter + 1)
    
    pooled_samples = []
    
    for idx, df in enumerate(dfs):
        # ─── CHANGED HERE ────────────────────────────────────────────────────────
        # Align to global index. Forward-fill stopped runs only when requested.
        df_aligned = df.set_index('iter').reindex(n_range)
        if forward_fill:
            df_aligned = df_aligned.ffill()
        # ─────────────────────────────────────────────────────────────────────────
        
        if trend_metric in ('qdet', 'q_det', 'deterministic'):
            values = np.maximum(1.0 - np.clip(df_aligned['q_det_rec'].values, 0.0, q_clip_max), 1e-5)
            pooled_samples.append(values.reshape(1, -1))
        elif trend_metric in ('noisy', 'noisy_rec', 'measurement', 'score'):
            y1 = df_aligned['y_rec_check'].values
            y2 = df_aligned['y_rec_second'].values
            y_noisy = np.where(np.isfinite(y2), 0.5 * (y1 + y2), y1)
            values = np.maximum(1.0 - np.clip(y_noisy, 0.0, q_clip_max), 1e-5)
            pooled_samples.append(values.reshape(1, -1))
        elif trend_metric in ('noisy_acq', 'acquisition'):
            values = np.maximum(1.0 - np.clip(df_aligned['y_acq'].values, 0.0, q_clip_max), 1e-5)
            pooled_samples.append(values.reshape(1, -1))
        else:
            m = np.maximum(1.0 - np.clip(df_aligned['m_rec'].values, 0.0, 1.0), 1e-5)
            s = df_aligned['s_rec'].values

            # Mask to identify valid entries (in case any seeds START late)
            valid = ~np.isnan(m) & ~np.isnan(s)

            samples = np.full((500, len(n_range)), np.nan)

            if np.any(valid):
                # Sample only for valid indices
                np.random.seed(42 + idx)
                valid_samples = np.random.normal(loc=m[valid], scale=s[valid], size=(500, np.sum(valid)))
                valid_samples = np.clip(valid_samples, 1e-5, 2.0)
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
ax.set_ylim(1e-5, 1e0)
ax.set_xlim(0, max_iter)
ax.set_xlabel('Iteration $n$', fontsize=14)
metric_for_label = os.environ.get('UCB_TREND_METRIC', 'surrogate').lower()
if metric_for_label in ('qdet', 'q_det', 'deterministic'):
    ax.set_ylabel(r'$1 - Q_{\mathrm{det}}(\mathbf{x}^*_n)$', fontsize=14)
elif metric_for_label in ('noisy', 'noisy_rec', 'measurement', 'score'):
    ax.set_ylabel(r'$1 - Q_{\mathrm{noisy}}(\mathbf{x}^*_n)$', fontsize=14)
elif metric_for_label in ('noisy_acq', 'acquisition'):
    ax.set_ylabel(r'$1 - Q_{\mathrm{noisy}}(\mathbf{x}_{n+1})$', fontsize=14)
else:
    ax.set_ylabel(r'$1 - \hat{\mu}(\mathbf{x}^*_n)$', fontsize=14)
ax.grid(True, which='both', ls=':', alpha=0.5)
ax.tick_params(axis='both', direction='in', length=5, labelsize=12)

output_file = os.environ.get('UCB_TREND_OUT', 'comparison_ucb_only_ffill.png')
fig.savefig(output_file, dpi=300)
print(f"Saved {output_file}")
plt.show()
