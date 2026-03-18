# CalibrationCode: Real Bayesian Optimisation in Julia

This repository provides a small, self-contained **Bayesian Optimisation (BO)** toolkit in **Julia**, built around **Gaussian Process (GP)** surrogate models. It includes several BO variants (standard, heteroscedastic, multi-fidelity, and co-kriging), plus a calibration use case based on `IonSim`/`QuantumOptics`.

The code is organised as a Julia project (`Project.toml` / `Manifest.toml`) with a package-style `src/` entrypoint and runnable `examples/` and `scripts/`.

---

## What is included

### BO algorithms
- **Standard BO (homoscedastic noise)**  
  GP surrogate + **Expected Improvement (EI)** acquisition.
- **Heteroscedastic BO (variable observation noise)**  
  GP surrogate with **per-observation noise** and an acquisition rule that selects both:
  - the next point `x`, and
  - an evaluation noise level `σ` (mapped to shots `N` in the calibration setting).
- **Multi-fidelity BO (augmented-input)**  
  Treats fidelity as an additional input coordinate `z ∈ [0, 1]` with an acquisition rule that trades off **EI / cost**.
- **Co-kriging multi-fidelity BO**  
  Multi-fidelity surrogate using an autoregressive structure and EI/cost style selection.

### Calibration model
- `src/calibration.jl` provides:
  - `ideal(t)` baseline settings,
  - `Q_det(...)` deterministic fidelity estimator,
  - `Q_noisy(...; N=...)` shot-noise estimator.

### Plotting utilities
All plotting scripts are in `scripts/` and are written as normal Julia files (no modules), intended for reuse in scripts and examples:
- `plots_standard.jl`
- `plots_hetero.jl`
- `plots_mf.jl`
- `plots_cokrig.jl`
- plus calibration-specific exploratory plotting scripts.

---

## Repository structure

```
calibration_code/
├─ Project.toml
├─ Manifest.toml
├─ readme.md
│
├─ src/
│  ├─ CalibrationCode.jl          # package entrypoint
│  ├─ bayes_opt.jl                # standard BO (EI)
│  ├─ bayes_hetero_opt.jl         # heteroscedastic BO
│  ├─ bayes_MF_opt.jl             # augmented-input multi-fidelity BO
│  ├─ bayes_cokrig_opt.jl         # co-kriging BO
│  ├─ coordinate_search.jl        # coordinate search baseline (non-BO)
│  └─ calibration.jl              # IonSim-based calibration model
│
├─ examples/                      # minimal “toy” validations
│  ├─ toy_standard_2d.jl          # 2D + plots
│  ├─ toy_standard_3d.jl          # 3D optimisation only
│  ├─ toy_hetero_2d.jl            # 2D + plots
│  ├─ toy_hetero_3d.jl            # 3D optimisation only
│  ├─ toy_mf_2d.jl                # 2D + plots (3 fidelities)
│  ├─ toy_mf_3d.jl                # 3D optimisation only
│  ├─ toy_cokrig_2d.jl            # 2D samples + (highest fidelity) view
│  └─ toy_cokrig_3d.jl            # 3D optimisation only
│
├─ scripts/                       # “real” runs + plotting utilities
│  ├─ plots_standard.jl
│  ├─ plots_hetero.jl
│  ├─ plots_mf.jl
│  ├─ plots_cokrig.jl
│  ├─ plots_fidelity.jl
│  ├─ fidelity_local_plot.jl
│  ├─ opt_qdet_vs_qnoisy.jl
│  ├─ opt_qnoisy_3d.jl
│  ├─ opt_qnoisy_2d_fcl_fsb.jl
│  └─ opt_qnoisy_2d_fsb_A.jl
│
└─ figures/                       # generated outputs (png/gif)
```

---

## Installation

This repository is a Julia project.

1. Install Julia (recommended: Julia ≥ 1.9).
2. From the repository root, start Julia and instantiate:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

If you need platform-specific notes (especially for `IonSim`/`QuantumOptics` dependencies), follow your local installation notes or the repository instructions if you maintain a separate installation guide.

---

## Quick start

### Run the toy examples
These are fast sanity checks and validate that the algorithms work and (for 2D) that the plotting utilities behave as expected.

```bash
julia --project=. examples/toy_standard_2d.jl
op

julia --project=. examples/toy_hetero_2d.jl
julia --project=. examples/toy_hetero_3d.jl

julia --project=. examples/toy_mf_2d.jl
julia --project=. examples/toy_mf_3d.jl

julia --project=. examples/toy_cokrig_2d.jl
julia --project=. examples/toy_cokrig_3d.jl
```

Outputs (PNGs/GIFs) are written in the working directory or under `figures/` depending on the example.

### Run the calibration optimisation scripts

- **Compare deterministic BO vs heteroscedastic BO on noisy evaluations**
```bash
julia --project=. scripts/opt_qdet_vs_qnoisy.jl
```

- **Heteroscedastic optimisation of Q_noisy**
```bash
julia --project=. scripts/opt_qnoisy_3d.jl
julia --project=. scripts/opt_qnoisy_2d_fcl_fsb.jl
julia --project=. scripts/opt_qnoisy_2d_fsb_A.jl
```

Generated figures are typically saved as `.png` / `.gif` in the current directory or into `figures/`.

---

## Notes on the calibration objective

The calibration model provides two estimators:

- `Q_det(t, f_cl, f_sb, A)`  
  Deterministic fidelity computation from simulated dynamics.
- `Q_noisy(t, f_cl, f_sb, A; N=...)`  
  Sampling-based estimator where `N` is the number of shots/samples.

In the heteroscedastic BO scripts, the BO “noise level” `σ` is used as an algorithmic knob and is mapped to a shot count `N` (see `N_from_sigma(σ)` inside the scripts). This is where the cost/accuracy trade-off is controlled.

---

## Reproducibility

- All examples and scripts use explicit random seeds.
- `Project.toml` and `Manifest.toml` are committed to support reproducible environments.

---

## Citation

If you use this repository in academic work, cite it as software and include the commit hash.

---

## License

Add a license file if you intend to distribute publicly (e.g., MIT/BSD/Apache-2.0).
