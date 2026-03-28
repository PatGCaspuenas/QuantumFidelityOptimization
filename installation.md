# Setup Instructions

## Requirements
- macOS or Linux
- curl (pre-installed on both)
- Git

## Automated Setup (recommended)

From the repo root, run:

    bash setup.sh

This single command will:
1. Install juliaup (if not already installed)
2. Install and set Julia 1.11.6 as default
3. Develop IonSim into ~/.julia/dev/IonSim
4. Check out IonSim v0.5.1
5. Automatically patch iontraps.jl (Optim.Options tolerances)
6. Restore the locked Manifest.toml so all package versions match exactly
7. Instantiate and build all dependencies

## Running the Project

    julia --project=. src/main.jl

## Undoing the IonSim Patch

    git -C ~/.julia/dev/IonSim checkout -- src/iontraps.jl

## Manual Steps (reference only)

The patch applied to ~/.julia/dev/IonSim/src/iontraps.jl replaces the
Optim.Options(...) call (around line 495) with:

    Optim.Options(g_tol=1e-6, x_abstol=1e-12, x_reltol=1e-6,
                  f_abstol=1e-12, f_reltol=1e-6)

If you need to redo the setup from scratch, remove your existing Julia
environment first:

    rm -rf ~/.julia
    bash setup.sh
