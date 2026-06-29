# QuantumFidelityOptimization

## Overview

This repository implements heteroscedastic Gaussian Process Bayesian Optimization (GP-UCB) for calibrating trapped-ion Mølmer–Sørensen (MS) gates.

The calibration objective is the **full-L1 fidelity score**

```
Q(u) = clamp(1 - 0.5 * (|p_ss| + |p_sd| + |p_ds| + |p_dd - 1|),  0, 1)
```

where p_ss, p_sd, p_ds, p_dd are the four measured two-qubit outcome probabilities. With N shots the dominant noise source is multinomial: σ² ≈ p_dd(1−p_dd)/N, which varies across parameter space — the motivation for using a heteroscedastic GP.

---

## Repository structure

```
QuantumFidelityOptimization/
├── src/
│   ├── CalibrationCode.jl        # module entry point + exports
│   ├── calibration.jl            # IonSim physics + estimators (ideal, Q_noisy, Q_varMS full-L1 objective)
│   ├── ms_sequences.jl           # MS subgate pulse construction + population calculator
│   └── bayes_opt.jl              # heteroscedastic GP-UCB (HeteroGP, fit_heterogp, bayesopt_ucb)
│
├── scripts/
│   ├── run_main.sh               # entry point: runs the full BO sweep
│   ├── main_opt.jl              # distributed BO trace generator
│   └── gp_fit_quality_study.jl  # GP fit quality vs. N_shots and n_pts
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
├── setup.sh          # automated setup script (installs Julia, IonSim, all deps)
└── installation.md   # detailed installation notes
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

# Calibration objective: full-L1 fidelity of numMS MS(π/2) gates → (y, σy)
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
