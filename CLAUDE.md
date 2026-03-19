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

# Pretraining pipeline (run first, then use in parallel benchmark)
julia --project=. scripts/pretrain_gp_3d.jl           # generates scripts/data/pretrained_theta_3d.jl
julia --project=. scripts/opt_qnoisy_3d_parallel_benchmark.jl      # parallel hetero-BO over many seeds
julia --project=. scripts/opt_qnoisy_3dmf_parallel_benchmark.jl    # parallel multi-fidelity BO
julia --project=. scripts/opt_qnoisy_4d_parallel_benchmark.jl      # 5-param BO (adds phi_1, phi_2 phases); outputs benchmark_results_withphase.txt

# Analysis & comparison
julia --project=. scripts/compare_benchmarks.jl        # plot multiple benchmark result files
julia --project=. scripts/compare_efficiency.jl

# Electric field noise study
julia --project=. scripts/efield_noise_bo_prototype.jl
```

Outputs (PNG/GIF) go to the working directory or `figures/`. Benchmark `.txt` results go to `scripts/`.

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

### Input convention

All scripts normalize physical parameters to `[-1, 1]^d` via a local `u_to_params(u)` function before calling physics functions. The bounds passed to `bayesopt_ucb_threshold` are always `[(-1.0, 1.0), ...]`. Physical units: `span_kHz = 2.0` (±2 kHz search window), `span_A = 0.2 * A0`.

### Key function signatures

```julia
# Standard BO
bayesopt(f; bounds, n_init=8, n_iter=30, xi=0.01, maximize=true)

# Heteroscedastic BO (main workhorse)
# f(u, σ) → scalar; u is normalized to [-1,1]^d
bayesopt_ucb_threshold(f; bounds, σ_levels, n_init, n_iter, κ=2.0, α=0.5, seed,
    pretrained_θ=nothing,       # warm-start from pretrain_gp_3d.jl
    freeze_mode=:none,          # :none | :lengthscales | :all
    n_freeze_iters=typemax(Int), # iterations to hold ℓ fixed before releasing to full MLE
    fidelity_threshold=nothing, # early stopping
    explore_frac=0.0)           # fraction of steps forced to high-noise

# Multi-fidelity
bayesopt_mf(f; bounds, z_levels, costs, n_init, n_iter, maximize)
cokrig_bayesopt(...)
```

### Pretraining workflow

`pretrain_gp_3d.jl` samples 500 deterministic evaluations (`Q_varMS` with N=1000), fits a GP via `fit_heterogp`, and saves `θ = [logℓ₁, logℓ₂, logℓ₃, logσf, logc]` to `scripts/data/pretrained_theta_3d.jl`. The parallel benchmark then loads this via `include` and uses `freeze_mode=:lengthscales` to fix lengthscales for the first `n_freeze_iters` iterations, allowing σf and noise scale to adapt.

### Quantum Physics Model (`src/calibration.jl`)

- `ideal(t)` — returns optimal `(f_cl, f_sb, A)` for gate time `t` (baseline reference)
- `Q_det(t, f_cl, f_sb, A; phi_1, phi_2)` — deterministic fidelity via IonSim Hamiltonian evolution
- `Q_noisy(t, f_cl, f_sb, A; N=100, phase_grid=...)` — shot-noise fidelity: samples N outcomes, performs parity scan + cosine fit
- `Q_varMS(t, f_cl, Δ, I; N=1000, numMS=2)` — fidelity for multiple MS gates, contains shot-noise fidelity

### Noise/shot mapping
`N_from_sigma(σ) = clamp(round(Int, 1/σ²), 20, 10000)` — defined in calibration scripts. The BO selects both a location `x` and noise level `σ ∈ σ_levels`, which maps to shot count `N`.

## Key Hyperparameters

**BO settings (current benchmark):**
- `n_init` = 12, `n_iter` = 120
- `M_acq` = 5000, `M_rec` = 20000 (acquisition / recommendation candidate points)
- `κ` = 1.9 (GP-UCB exploration weight)
- `α` = 1.5 (noise threshold parameter)
- `σ_levels` = [0.1412, 0.1, 0.06, 0.04472] (maps to N ≈ 50, 100, 278, 500 shots)
- `explore_frac` = 0.15 (15% of steps forced to coarsest σ)
- `freeze_mode` = `:lengthscales`, `n_freeze_iters` = 40

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
- All examples and scripts are standalone (not unit tests) — run them directly to validate behavior.
- `efield_noise_bo_prototype.jl` models 1/f^α electric field noise as a slow motional mode frequency drift injected into `Q_varMS` calls. The BO sees only programmed parameters; the gate runs with a drifted effective f_sb. Uses `PyPlot` + `FFTW` (not loaded by CalibrationCode).
- All parallel benchmark scripts use Julia's `Distributed` stdlib with `pmap`. Constants must be broadcast to workers via `@everywhere` before use.
- `opt_qnoisy_4d_parallel_benchmark.jl` is misnamed — it optimizes 5 parameters (`f_cl`, `f_sb`, `A`, `phi_1`, `phi_2`) with `span_phi = π`.
- `examples/` contains 3D variants of all toy problems (`toy_hetero_3d.jl`, `toy_mf_3d.jl`, etc.) and real calibration examples (`calib_standard2D.jl`, `calib_standard3D.jl`).
