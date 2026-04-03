import re
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    'axes.facecolor': '#f5f3ee',
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

# ============================================================
# Helpers
# ============================================================

def parse_results_header(lines):
    """
    Parse the '=== All Results ===' header line from a benchmark file.
    Handles both old format (Sim Seed Iterations Q_det σ=... columns)
    and new format (Sim Seed Iterations TotalShots NTrain Q_det ...).
    Returns a dict with parsing info, or None if the header cannot be found.
    """
    n_shots_hint = None
    n_init_hint = None
    for line in lines:
        if n_shots_hint is None:
            m = re.search(r'N_shots\s*=\s*(\d+)', line)
            if m:
                n_shots_hint = int(m.group(1))
        if n_init_hint is None:
            m = re.search(r'n_init\s*=\s*(\d+)', line)
            if m:
                n_init_hint = int(m.group(1))
        if n_shots_hint is not None and n_init_hint is not None:
            break
    n_init_hint = n_init_hint if n_init_hint is not None else 12

    for i, line in enumerate(lines):
        if ('=== All Results ===' in line or 'INDIVIDUAL RESULTS' in line) and i < len(lines) - 1:
            for j in range(i + 1, min(i + 4, len(lines))):
                hdr = lines[j]
                if not hdr.strip():
                    continue
                parts = hdr.split('\t')

                sigma_cols = []
                sigma_levels = []
                for k, p in enumerate(parts):
                    m = re.search(r'σ=([\d.]+)', p.strip())
                    if m:
                        sigma_cols.append(k)
                        sigma_levels.append(float(m.group(1)))

                def find_col(name):
                    for k, p in enumerate(parts):
                        if p.strip() == name:
                            return k
                    return None

                iter_col        = find_col('Iterations')
                qdet_col        = find_col('Q_det')
                total_shots_col = find_col('TotalShots')

                if qdet_col is None:
                    continue

                iter_col = iter_col if iter_col is not None else 2
                qdet_col = qdet_col if qdet_col is not None else 3

                if n_shots_hint is None and sigma_levels:
                    n_shots_hint = round(1 / sigma_levels[0] ** 2)

                return dict(
                    sigma_levels=sigma_levels,
                    iter_col=iter_col,
                    qdet_col=qdet_col,
                    total_shots_col=total_shots_col,
                    sigma_cols=sigma_cols,
                    n_shots_hint=n_shots_hint,
                    n_init_hint=n_init_hint,
                )
    return None


def benchmark_label(filepath, sigma_levels, user_labels=None):
    key = Path(filepath).name
    if user_labels and key in user_labels:
        return user_labels[key]
    filename = key.lower()
    m = re.search(r'benchmark_results_(.+?)\.txt', filename)
    tag = m.group(1) if m else key
    if len(sigma_levels) == 1:
        N = round(1 / sigma_levels[0] ** 2)
        return f'Static N={N} ({tag})'
    return tag


# ============================================================
# Main plot
# ============================================================

def plot_shots_vs_fidelity(files, output_file=None, max_iterations=120, labels=None):
    """
    Log10(infidelity) vs log10(total shots) scatter.
    Faint individual trial points + mean ± std summary marker per file.
    """
    palette = sns.color_palette("husl", len(files))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), subplot_kw={'box_aspect': 1},
                           layout='constrained')

    all_x, all_y = [], []

    for file_idx, filepath in enumerate(files):
        filepath = Path(filepath)
        if not filepath.is_file():
            warnings.warn(f'File not found: {filepath}')
            continue

        lines = filepath.read_text().splitlines()
        info = parse_results_header(lines)
        if info is None:
            warnings.warn(f'Could not parse header in {filepath} — skipping')
            continue

        sigma_levels     = info['sigma_levels']
        iter_col         = info['iter_col']
        qdet_col         = info['qdet_col']
        sigma_cols       = info['sigma_cols']
        total_shots_col  = info['total_shots_col']
        n_shots_hint     = info['n_shots_hint']
        n_init_hint      = info['n_init_hint']

        label = benchmark_label(filepath, sigma_levels, labels)
        col   = palette[file_idx]

        log_shots_vec = []
        fidelity_vec  = []
        n_unconverged = 0

        in_results = False
        header_seen = False
        for line in lines:
            if '=== All Results ===' in line or 'INDIVIDUAL RESULTS' in line:
                in_results = True
                header_seen = False
                continue
            if not in_results or not line.strip():
                continue
            if not header_seen and 'Sim' in line:
                header_seen = True
                continue
            if not header_seen:
                continue

            parts = line.split('\t')
            if len(parts) <= max(iter_col, qdet_col):
                continue
            try:
                itr   = int(parts[iter_col].strip())
                q_det = float(parts[qdet_col].strip())
            except ValueError:
                continue

            if itr == max_iterations:
                n_unconverged += 1
                continue

            # Total shots: prefer TotalShots column; fall back to old-format estimate
            lx = None
            if total_shots_col is not None and len(parts) > total_shots_col:
                try:
                    ts = int(parts[total_shots_col].strip())
                    if ts > 0:
                        lx = np.log10(float(ts))
                except ValueError:
                    pass
            if lx is None and n_shots_hint is not None:
                lx = np.log10(float((n_init_hint + 2 * itr) * n_shots_hint))
            if lx is None and sigma_cols and len(parts) >= sigma_cols[-1] + 1:
                try:
                    total_shots = sum(
                        int(parts[sigma_cols[j]].strip()) * round(1 / sigma_levels[j] ** 2)
                        for j in range(len(sigma_levels))
                    )
                    if total_shots > 0:
                        lx = np.log10(float(total_shots))
                except ValueError:
                    pass
            if lx is None:
                continue

            log_shots_vec.append(lx)
            fidelity_vec.append(q_det)

        if n_unconverged:
            warnings.warn(f'{filepath.name}: {n_unconverged} run(s) hit max_iterations and were excluded')

        if not log_shots_vec:
            continue

        infidelity_vec = 1.0 - np.array(fidelity_vec)
        log_shots_vec  = np.array(log_shots_vec)

        all_x.extend(log_shots_vec)
        all_y.extend(infidelity_vec)

        # Individual trial scatter (faint)
        ax.scatter(log_shots_vec, infidelity_vec,
                   color=col, alpha=0.2, s=18, linewidths=0,
                   zorder=2, clip_on=False)

        # Mean ± std summary point
        mx = np.mean(log_shots_vec)
        sx = np.std(log_shots_vec) if len(log_shots_vec) > 1 else 0.0
        my = np.mean(infidelity_vec)
        sy = np.std(infidelity_vec) if len(infidelity_vec) > 1 else 0.0

        full_label = f'{label} [{n_unconverged}]' if n_unconverged else label

        ax.errorbar(mx, my,
                    xerr=[[min(sx, mx)], [sx]],
                    yerr=[[min(sy, my * 0.9999)], [min(sy, max(1.0 - my, 0.0))]],
                    fmt='o', color=col, markersize=7, markeredgecolor=col,
                    markeredgewidth=1.5, capsize=3, elinewidth=1.2,
                    label=full_label, zorder=5, clip_on=False)

    if all_x:
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        x_pad = max((all_x.max() - all_x.min()) * 0.08, 0.1)
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y[all_y > 0].min() / 2.0, np.quantile(all_y, 0.95) * 3.0)

    ax.set_yscale('log')
    ax.set_xlabel('$\\log_{10}$(Total shots)', fontsize=14)
    ax.set_ylabel('$\\log_{10}(1 - \\mathrm{Fidelity})$', fontsize=14)
    ax.set_xlim([3.5,5.5])
    ax.set_ylim([1e-4, 1e-2])
    ax.set_title('$N=400$', fontsize=14)
    ax.tick_params(axis='both', direction='in', length=5,
                   top=False, bottom=True, left=True, right=False, labelsize=14)
    ax.grid(True, which='both', ls='-', alpha=0.5)
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=12)
    legend.set_title('[n] = unconverged runs', prop={'size': 10})


    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f'Saved → {output_file}')
    else:
        plt.show()

    return fig, ax


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    data_dir   = script_dir / 'data'
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # ── Files to compare ──────────────────────────────────────────────────────
    candidates = [
        data_dir / 'benchmark_results_onelevel_N400.txt',
        data_dir / 'benchmark_results_N400_newseeds_comparison.txt',       
        data_dir / 'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add.txt',
        data_dir / 'benchmark_results_N400_linear_binomial_every1_LFBGS_Ncheck2_add.txt',   
data_dir / 'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_restarts20.txt',
data_dir / 'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_explore25.txt',
data_dir / 'benchmark_results_N400_log_binomial_LFBGS_Ncheck2_add.txt',
       data_dir /  'benchmark_results_N400_linear_simple_LFBGS_Ncheck2_add_bounds.txt',
        data_dir / 'benchmark_results_N400_linear_simple_m_rec.txt',
        data_dir / 'benchmark_results_N400_linear_binomial_every1_sobol_LFBGS_Ncheck2_add.txt',


    ]
    files = [f for f in candidates if f.is_file()]

    # ── Human-readable labels (keyed by basename) ─────────────────────────────
    labels = {
        'benchmark_results_onelevel_N400_old.txt':                                           'v0 (old)',
        'benchmark_results_onelevel_N400.txt':                                                'v0',
        'benchmark_results_N400_newseeds_comparison.txt':                                     'v1',
        'benchmark_results_N400_linear_simple_LFBGS.txt':                                    'v1 LFBGS, 1 check, no add',
        'benchmark_results_N400_linear_simple_LFBGS_Ncheck2.txt':                            'v1 LFBGS, 2 checks, no add',
        'benchmark_results_N400_linear_simple_LFBGS_Ncheck2_add.txt':                        'v1 LFBGS, 2 checks, add',
        'benchmark_results_N400_linear_simple_LFBGS_add.txt':                                'v1 LFBGS, 1 check, add',
        'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add.txt':                      'v2',
        'benchmark_results_N400_linear_binomial_every1_sobol_LFBGS_Ncheck2_add.txt':                 'v2 sobol',
        'benchmark_results_N400_log_binomial_LFBGS_Ncheck2_add.txt':                         'v2 log',
        'benchmark_results_N400_linear_binomial_every1_LFBGS_Ncheck2_add.txt':            'v2 every 1',
        'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_restarts20.txt':         'v2 20 restarts',
        'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_explore25.txt':            'v2 explore 0.25',
        'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_ninit25.txt':              'v2 $n_{init}=25$',
        'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_ninit6.txt':               'v2 $n_{init}=6$',
        'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_ninit_fixed.txt':         'v2 $n_{init}$ fixed',
        'benchmark_results_N400_linear_simple_LFBGS_Ncheck2_add_bounds.txt':                 'v2 bounds',
        'benchmark_results_N400_linear_simple_m_rec.txt':                                     'v2 m rec',
        'benchmark_results_N400_linear_binomial_every1_LFBGS_Ncheck2_add_nfreeze20_all.txt': 'v2 $n_{freeze}=20$ (all)',
        'benchmark_results_N400_linear_binomial_every1_LFBGS_Ncheck2_add_nfreeze20.txt':     'v2 $n_{freeze}=20$ ($l$)',
        'benchmark_results_N400_linear_binomial_every1_LFBGS_Ncheck2_add_pretrain_all.txt':  'v2 pretrain (all)',
        'benchmark_results_N400_linear_binomial_every1_LFBGS_Ncheck2_add_pretrain.txt':      'v2 pretrain ($l$)',

    }
    # ─────────────────────────────────────────────────────────────────────────

    if not files:
        raise FileNotFoundError(f'No benchmark files found in {data_dir}')

    plot_shots_vs_fidelity(
        files,
        labels=labels,
    )
