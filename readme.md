# QuantumFidelityOptimization

## Overview

This repository implements heteroscedastic Gaussian Process Bayesian Optimization (GP-UCB) for calibrating trapped-ion Mølmer–Sørensen (MS) gates.

The calibration objective is **p_dd**, the population measured in the target |DD⟩ outcome:

```
Q(u) = clamp(p_dd,  0, 1)
```

where p_ss, p_sd, p_ds, p_dd are the four measured two-qubit outcome probabilities (normalized to sum to 1). With N shots the dominant noise source is multinomial: σ² ≈ p_dd(1−p_dd)/N, which varies across parameter space — the motivation for using a heteroscedastic GP.

---

## Repository structure

```
QuantumFidelityOptimization/
├── src/
│   ├── CalibrationCode.jl        # module entry point + exports
│   ├── calibration.jl            # IonSim physics + estimators (ideal, Q_noisy, Q_varMS p_dd objective)
│   ├── ms_sequences.jl           # MS subgate pulse construction + population calculator
│   └── bayes_opt.jl              # heteroscedastic GP-UCB (HeteroGP, fit_heterogp, bayesopt_ucb)
│
├── scripts/
│   ├── run_main.sh                    # entry point: runs the full BO sweep
│   ├── main_opt.jl                    # distributed BO trace generator
│   ├── gp_fit_quality_study.jl        # GP fit quality vs. N_shots and n_pts
│   ├── contour_data_generation.jl     # 2D score heatmaps (Q_infinity landscapes)
│   ├── gp_data_generation.jl          # GP posterior slices for selected-N traces
│   └── plots.ipynb                    # figure generation: one cell per figure,
│                                       # shown inline and saved as PDF to figures/
│
├── figures/                          # PDFs written by scripts/plots.ipynb
│
├── examples/
│   ├── toy_standard_2d.jl    # GP-UCB on a 2D toy function, constant-σ noise
│   ├── toy_standard_3d.jl    # GP-UCB on a 3D toy function, constant-σ noise
│   ├── toy_hetero_2d.jl      # GP-UCB on a 2D toy function, binomial shot noise
│   ├── toy_hetero_3d.jl      # GP-UCB on a 3D toy function, binomial shot noise
│   ├── calib_standard2D.jl   # GP-UCB on Q_noisy (f_sb, A)
│   └── calib_standard3D.jl   # GP-UCB on Q_noisy (f_cl, f_sb, A)
│
├── data/
│   ├── traces_*/                        # per-seed BO trace CSVs (one dir per N)
│   ├── benchmark_trace_N*.txt           # aggregate statistics per N
│   ├── gp_fit_quality_3d_2ms.csv        # GP accuracy metrics vs. N_shots and n_pts
│   ├── slices_output.csv                # 1D GP posterior slices
│   └── train_near_output.csv            # training points near 1D axes
│
├── Project.toml
├── Manifest.toml
├── requirements.txt   # pinned Python deps for scripts/plots.ipynb
├── setup.sh           # automated setup script (Julia + IonSim + Python env)
├── setup_python.sh    # Python venv + Jupyter kernel setup (also called by setup.sh)
└── installation.md    # detailed installation notes
```

---

## Installation

IonSim is an unregistered package that requires a local development checkout. The automated script handles everything:

```bash
bash setup.sh
```

This installs Julia 1.11.6, clones IonSim v0.5.1, applies a required compatibility patch to `iontraps.jl`, and instantiates all other dependencies from the locked `Manifest.toml`. See [installation.md](installation.md) for details and manual steps.

---

### BO traces 

Runs 100 independent BO seeds for each shot count N ∈ {100, 1000, 10000, 100000, ∞}:

```bash
bash scripts/run_main.sh
```

Output is written to `data/traces_<tag>/` (one CSV per seed) and a summary text file per N in `data/`. The pre-generated data is already included in `data/`.

Key parameters (set in `run_main.sh`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `N_ITER` | 100 | BO iterations per run |
| `N_INIT` | 12 | Latin hypercube initial design |
| `KAPPA` | 1.96 | GP-UCB exploration parameter |
| `BOUND_SCALE` | 0.5 | Search box half-width (normalized units) |
| `FREQ_SPAN_KHZ` | 10 | Physical frequency span per axis |

### GP fit quality study

Evaluates GP posterior accuracy across training sizes n ∈ {10, 50, 100, 250, 500, 1000} and shot counts N ∈ {100, 1000, 10000, 100000, ∞}:

```bash
julia --project=. scripts/gp_fit_quality_study.jl
```

Output: `data/gp_fit_quality_3d_2ms.csv`, `data/slices_output.csv`, `data/train_near_output.csv`.

### Paper figure data (contour + GP slices)

Two more generators feed the figure notebook:

```bash
julia --project=. scripts/contour_data_generation.jl   # -> data/varms_2_heatmap_{rabi_sideband,rabi_fcl,sideband_fcl}.csv
julia --project=. scripts/gp_data_generation.jl         # -> data/gp_slices/, data/score_cache/
```

`contour_data_generation.jl` computes deterministic (N=∞) 2D score heatmaps over the 2×MS(π/2) sequence (the contour figure). `gp_data_generation.jl` picks, for each N ∈ {100, 1000, 10000, 100000, ∞}, the BO trace whose final Q_det is closest to the median across all 100 seeds (requires `scripts/run_main.sh` to have been run first), then replays its training set at iterations 10/30/50 and fits a GP snapshot at each (the GP-slices figure).

### Generating the figures

All figures live in one notebook, `scripts/plots.ipynb` — one cell per figure, each displayed inline and saved as a PDF into `figures/`. See [Python environment](installation.md#python-environment-for-figure-generation) for one-time setup, then:

```bash
source .venv/bin/activate
jupyter notebook scripts/plots.ipynb
```

Run all cells (select the "Python 3 (QuantumFidelityOptimization)" kernel if prompted). Each cell reads straight from `data/` — no dependency on the old standalone `python_scripts/*.py` files.

---

## Examples

Quick sanity checks that require no data files:

```bash
# Toy functions — verify BO algorithms run correctly
julia --project=. examples/toy_standard_2d.jl
julia --project=. examples/toy_standard_3d.jl
julia --project=. examples/toy_hetero_2d.jl
julia --project=. examples/toy_hetero_3d.jl

# Calibration — BO directly on the IonSim physics model
julia --project=. examples/calib_standard2D.jl
julia --project=. examples/calib_standard3D.jl
```

---

## Core API

```julia
import Pkg; Pkg.activate(".")
include("src/CalibrationCode.jl")
using .CalibrationCode

# Physics model
base = ideal(100.0)                       # ideal parameters at t = 100 μs

# Calibration objective: p_dd of numMS MS(π/2) gates → (y, σy)
y, σy = Q_varMS(t, f_cl, f_sb, A; N=1000)

# Alternative Bell-parity estimator (used by the calib_standard examples)
q = Q_noisy(t, f_cl, f_sb, A; N=1000, phase_grid=0.0:0.2:π)

# Fit heteroscedastic GP
gp = fit_heterogp(X, y, σy)          # X: d×n, y/σy: n-vectors
μ, s2 = predict_latent(gp, x)        # posterior mean and variance
x_rec, m_rec, s_rec = recommend_mean(gp, bounds)   # GP-mean maximizer

# Run heteroscedastic GP-UCB
res = bayesopt_ucb(f; bounds, n_shots=500, n_init=12, n_iter=100, κ=1.96)
```
