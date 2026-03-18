# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CalibrationCode** is a Julia-based Bayesian Optimization (BO) toolkit for quantum gate fidelity optimization in ion-trap calibration. The core problem is optimizing carrier frequency (`f_cl`), sideband frequency (`f_sb`), and laser intensity (`A`) to maximize MS-gate fidelity on a Ca-40 trapped-ion system, using noisy (shot-limited) measurements.

## Setup & Running

```julia
# Activate and install dependencies
import Pkg; Pkg.activate("."); Pkg.instantiate()
```

```bash
# Quick sanity checks (toy problems, no quantum physics, fast)
julia --project=. examples/toy_standard_2d.jl
julia --project=. examples/toy_hetero_2d.jl
julia --project=. examples/toy_mf_2d.jl
julia --project=. examples/toy_cokrig_2d.jl

# Full quantum calibration runs (slower, uses IonSim)
julia --project=. scripts/opt_qnoisy_3d.jl
julia --project=. scripts/opt_qnoisy_2d_fcl_fsb.jl
julia --project=. scripts/opt_qnoisy_2d_fsb_A.jl
julia --project=. scripts/opt_qdet_vs_qnoisy.jl
```

Outputs (PNG/GIF) go to the working directory or `figures/`.

## Architecture

### Module entrypoint
`src/CalibrationCode.jl` — loads all submodules. All algorithms are included here.

### BO Algorithms (`src/`)

| File | Algorithm | Acquisition | When to use |
|------|-----------|-------------|-------------|
| `bayes_opt.jl` | Standard homoscedastic GP | EI | Deterministic or fixed-noise objectives |
| `bayes_hetero_opt.jl` | **Heteroscedastic BO** (primary) | GP-UCB + noise threshold | Shot-noisy quantum objectives |
| `bayes_MF_opt.jl` | Multi-fidelity (augmented-input) | EI/cost | Multiple simulation fidelity levels |
| `bayes_cokrig_opt.jl` | Co-kriging multi-fidelity | EI/cost | AR(1) structure across fidelities |
| `coordinate_search.jl` | Coordinate search (baseline) | — | Non-BO comparison |

### Key function signatures

```julia
# Standard BO
bayesopt(f; bounds, n_init=8, n_iter=30, xi=0.01, maximize=true)

# Heteroscedastic BO (main workhorse)
bayesopt_ucb_threshold(f; bounds, σ_levels, n_init, n_iter, κ=2.0, α=0.5, seed)

# Multi-fidelity
bayesopt_mf(f; bounds, z_levels, costs, n_init, n_iter, maximize)
cokrig_bayesopt(...)
```

### Quantum Physics Model (`src/calibration.jl`)

- `ideal(t)` — returns optimal `(f_cl, f_sb, A)` for gate time `t` (baseline reference)
- `Q_det(t, f_cl, f_sb, A; phi_1, phi_2)` — deterministic fidelity via IonSim Hamiltonian evolution
- `Q_noisy(t, f_cl, f_sb, A; N=100, phase_grid=...)` — shot-noise fidelity: samples N outcomes, performs parity scan + cosine fit
- `Q_varMS(t, f_cl, Δ, I; N=1000, numMS=2)` — fidelity for multiple MS gates, contains shot-noise fidelity

### Noise/shot mapping
`N_from_sigma(σ) = clamp(round(Int, 1/σ²), 20, 10000)` — defined in calibration scripts. The BO selects both a location `x` and noise level `σ ∈ σ_levels`, which maps to shot count `N`.

## Key Hyperparameters

**BO settings:**
- `n_init` = 8–12 (initial random evaluations)
- `n_iter` = 30–120 (BO iterations)
- `M` = 2000–4000 (acquisition function candidate points)
- `κ` = 2.0 (GP-UCB exploration weight)
- `α` = 0.5 (noise threshold parameter)
- `σ_levels` = [0.1, 0.05, 0.02, 0.01] (available noise levels)

**GP kernel:** Matérn 3/2 with ARD (per-dimension lengthscales). Hyperparameters fit via bounded L-BFGS with multi-start restarts.

**Quantum:**
- Gate time `t` ≈ 100 µs
- Phase scan grid: `0:0.1:π`
- Ion species: Ca-40, 2-ion chain, S/D levels

## Data & Outputs

- `scripts/data/` — benchmark result `.txt` files (convergence stats, seeds, fidelity metrics)
- `figures/` — generated PNG/GIF plots
- `scripts/plots/` — organized plot subdirectories

## Style & Implementation Guidelines

- Responses should be concise and clear — lead with the answer, skip preamble.
- Keep implementations as simple as possible while maintaining functionality. Prefer straightforward code over clever abstractions. Do not add complexity, extra configurability, or helper utilities beyond what is directly needed.

## Notes

- Scripts use explicit random seeds for reproducibility.
- `*_backup.jl` files are archived versions; prefer the non-backup counterparts.
- The heteroscedastic BO (`bayes_hetero_opt.jl`) is the primary algorithm; it supports a `pretrain_heterogp_deterministic()` warm-start from cheaper deterministic evaluations.
- All examples and scripts are standalone (not unit tests) — run them directly to validate behavior.
