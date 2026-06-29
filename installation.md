# Installation

## Requirements

- Linux or macOS
- `curl` and `git` (pre-installed on both)
- Internet access (for juliaup and package downloads)

## Automated setup (recommended)

From the repository root:

```bash
bash setup.sh
```

This single command:

1. Installs [juliaup](https://github.com/JuliaLang/juliaup) if not present
2. Installs Julia 1.11.6 and sets it as the default
3. Clones IonSim v0.5.1 to `~/.julia/dev/IonSim`
4. Applies a required compatibility patch to `IonSim/src/iontraps.jl` (tightens `Optim.Options` tolerances so the IonSim internal optimizer converges reliably)
5. Instantiates all other dependencies from the locked `Manifest.toml`
6. Builds IonSim

Total time is typically 10–20 minutes on a fresh machine (dominated by package downloads and precompilation).

## Verifying the installation

```bash
julia --project=. examples/toy_hetero_2d.jl
```

Expected output: a recommended point and distance-to-optimum printed to stdout.

## Undoing the IonSim patch

```bash
git -C ~/.julia/dev/IonSim checkout -- src/iontraps.jl
```

## Manual setup (reference)

If you prefer not to use `setup.sh`:

```julia
import Pkg
Pkg.develop("IonSim")
# Checkout v0.5.1:
#   git -C ~/.julia/dev/IonSim checkout v0.5.1
# Apply patch to iontraps.jl (see setup_pkg.jl for the exact replacement)
Pkg.instantiate()
Pkg.build("IonSim")
```

The patch replaces the `Optim.Options(...)` call near line 495 of `iontraps.jl` with:

```julia
Optim.Options(g_tol=1e-6, x_abstol=1e-12, x_reltol=1e-6,
              f_abstol=1e-12, f_reltol=1e-6)
```
