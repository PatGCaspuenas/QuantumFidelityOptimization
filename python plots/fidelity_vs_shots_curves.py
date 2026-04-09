import re
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
# Helpers (shared with fidelity_vs_shots_benchmarks.py)
# ============================================================

def parse_results_header(lines):
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


def extract_shots_and_fidelity(filepath, max_iterations=120):
    """Parse a benchmark file and return (log10_shots, infidelity, n_unconverged)."""
    filepath = Path(filepath)
    if not filepath.is_file():
        warnings.warn(f'File not found: {filepath}')
        return None, None, 0

    lines = filepath.read_text().splitlines()
    info = parse_results_header(lines)
    if info is None:
        warnings.warn(f'Could not parse header in {filepath} — skipping')
        return None, None, 0

    iter_col        = info['iter_col']
    qdet_col        = info['qdet_col']
    sigma_cols      = info['sigma_cols']
    sigma_levels    = info['sigma_levels']
    total_shots_col = info['total_shots_col']
    n_shots_hint    = info['n_shots_hint']
    n_init_hint     = info['n_init_hint']

    log_shots_vec, fidelity_vec = [], []
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
        return None, None, n_unconverged

    return np.array(log_shots_vec), 1.0 - np.array(fidelity_vec), n_unconverged


# ============================================================
# Main plot
# ============================================================

def plot_versions(n_pairs, output_file=None, max_iterations=120):
    """
    n_pairs: list of dicts with keys:
        N        - shot count label (int)
        old_file - path to old-version benchmark file (optional)
        new_file - path to new-version benchmark file (optional)
        var      - if True, treat as a variable-N pair: drawn in black with
                   diamond markers (old=hollow 'D', new=filled 'D')
    Color encodes N; marker encodes version (circle=old, triangle=new).
    var pairs are always black with diamond markers and sit on top (zorder=10).
    Faint individual points + mean±std summary markers.
    N legend below the plot.
    """
    fixed_pairs = [p for p in n_pairs if not p.get('var')]
    N_vals    = [p['N'] for p in fixed_pairs]
    palette   = sns.color_palette("husl", len(N_vals))
    color_map = dict(zip(N_vals, palette))

    marker_old = 'o'
    marker_new = '^'
    marker_var_old = 'o'   # hollow diamond  (mfc='none')
    marker_var_new = '^'   # filled diamond

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={'box_aspect': 1},
                           layout='constrained')

    all_x, all_y = [], []
    x_range_per_N = {}  # N -> [xmin, xmax] across both versions

    for pair in n_pairs:
        is_var = pair.get('var', False)
        col    = 'black' if is_var else color_map[pair['N']]

        entries = []
        entries = [(pair.get('old_file'), marker_old, False),
                       (pair.get('new_file'), marker_new, False)]

        for filepath, marker, hollow in entries:
            if filepath is None:
                continue

            log_shots, infidelity, _ = extract_shots_and_fidelity(
                filepath, max_iterations=max_iterations)
            if log_shots is None:
                continue

            all_x.extend(log_shots)
            all_y.extend(infidelity)

            if not is_var:
                N = pair['N']
                xlo, xhi = log_shots.min(), log_shots.max()
                if N not in x_range_per_N:
                    x_range_per_N[N] = [xlo, xhi]
                else:
                    x_range_per_N[N][0] = min(x_range_per_N[N][0], xlo)
                    x_range_per_N[N][1] = max(x_range_per_N[N][1], xhi)

            mfc   = 'none' if hollow else col
            zbase = 2
            msize = 7
            ew    =  1.5

            # Faint individual trial points
            ax.scatter(log_shots, infidelity,
                       color=col, alpha=0.15,
                       s=18, linewidths=0,
                       marker=marker, zorder=zbase, clip_on=True)

            # Mean ± std summary marker
            mx = np.mean(log_shots)
            sx = np.std(log_shots) if len(log_shots) > 1 else 0.0
            my = np.mean(infidelity)
            sy = np.std(infidelity) if len(infidelity) > 1 else 0.0

            ax.errorbar(mx, my,
                        xerr=[[min(sx, mx)], [sx]],
                        yerr=[[min(sy, my * 0.9999)], [min(sy, max(1.0 - my, 0.0))]],
                        fmt=marker, color=col, markersize=msize,
                        markerfacecolor=mfc,
                        markeredgecolor=col, markeredgewidth=ew,
                        capsize=3, elinewidth=ew,
                        zorder=zbase + 1, clip_on=True)

    if all_x:
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        x_pad = max((all_x.max() - all_x.min()) * 0.08, 0.1)
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        pos_y = all_y[all_y > 0]
        if len(pos_y):
            ax.set_ylim(1e-5, np.quantile(all_y, 0.95) * 3.0)

    # Threshold lines: y = 1/N, spanning only the x range of that N's data
    for N, (xlo, xhi) in x_range_per_N.items():
        ax.plot([xlo, xhi], [1 / N, 1 / N],
                color=color_map[N], linestyle='--', linewidth=2.0, alpha=0.9,
                solid_capstyle='round', zorder=1)

    ax.set_yscale('log')
    ax.set_xlabel('$\\log_{10}$(Total shots)', fontsize=14)
    ax.set_ylabel('$1 - \\mathrm{Fidelity}$', fontsize=14)
    ax.tick_params(axis='both', direction='in', length=5,
                   top=False, bottom=True, left=True, right=False, labelsize=14)
    ax.grid(True, which='both', ls='-', alpha=0.5)

    # Version legend (markers, no color) — inside the plot
    version_handles = [
        mlines.Line2D([], [], color='dimgray', marker=marker_old, linestyle='None',
                      markersize=7, label='Old'),
        mlines.Line2D([], [], color='dimgray', marker=marker_new, linestyle='None',
                      markersize=7, label='New'),
        mlines.Line2D([], [], color='dimgray', linestyle='--', linewidth=2.0,
                      label='Threshold'),
    ]
    version_legend = ax.legend(handles=version_handles, loc='upper right', fontsize=12)
    ax.add_artist(version_legend)

    # N legend below the plot (color patches)
    n_handles = [mpatches.Patch(color=color_map[N], label=f'${N}$') for N in N_vals]
    if is_var:
        n_handles.append(mpatches.Patch(color='black', label='Var'))
    fig.legend(handles=n_handles, loc='outside lower center', ncol=len(N_vals) + 1 if is_var else len(N_vals),
               title='Number of shots $N$',
               frameon=True, fontsize=12, title_fontsize=12,
               handlelength=1.0, handleheight=0.8)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'Saved → {output_file}')
    else:
        plt.show()

    return fig, ax


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    script_dir  = Path(__file__).parent
    data_dir    = script_dir / 'data'
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    n_pairs = [
        dict(N=50,
             old_file=data_dir / 'benchmark_results_onelevel_N50.txt',
             new_file=data_dir / 'benchmark_results_N50_linear_binomial_LFBGS_Ncheck2_add.txt'),
        dict(N=100,
             old_file=data_dir / 'benchmark_results_onelevel_N100.txt',
             new_file=data_dir / 'benchmark_results_N100_linear_binomial_LFBGS_Ncheck2_add.txt'),
        dict(N=250,
             old_file=data_dir / 'benchmark_results_onelevel_N250.txt',
             new_file=data_dir / 'benchmark_results_N250_linear_binomial_LFBGS_Ncheck2_add.txt'),
        dict(N=400,
             old_file=data_dir / 'benchmark_results_onelevel_N400.txt',
             new_file=data_dir / 'benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add.txt'),
        dict(N=2500,
             old_file=data_dir / 'benchmark_results_onelevel_N2500.txt',
             new_file=data_dir / 'benchmark_results_N2500_linear_binomial_LFBGS_Ncheck2_add.txt'),
        dict(N=10000,
             old_file=data_dir / 'benchmark_results_onelevel_N10000.txt',
             new_file=data_dir / 'benchmark_results_N10000_linear_binomial_LFBGS_Ncheck2_add.txt'),
        dict(var=True,
             old_file=data_dir / 'benchmark_results_fourlevels.txt',
             new_file=data_dir / 'benchmark_results_varN_verify_floor_binomial.txt'),
    ]

    plot_versions(
        n_pairs,
    )
