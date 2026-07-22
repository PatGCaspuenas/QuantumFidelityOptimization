# Installation

## Requirements

- Linux or macOS
- `curl` and `git` (pre-installed on both)
- Internet access (for juliaup and package downloads)
- Python >= 3.12 (checked by `setup_python.sh`; needed by the pinned `numpy`/`scipy` versions in `requirements.txt`)
- A LaTeX distribution providing `pdflatex` and `dvipng` on `PATH` (checked by `setup_python.sh`, warns but doesn't install automatically — see below). Required because every figure script sets matplotlib's `text.usetex=True`; without it, `scripts/plots.ipynb` fails when rendering figure text.

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
7. Runs `setup_python.sh` (see below) to set up the Python figure-generation environment

Total time is typically 10–20 minutes on a fresh machine (dominated by package downloads and precompilation).

## Python environment (for figure generation)

All paper figures are generated from `scripts/plots.ipynb`, which needs `matplotlib`, `numpy`, `pandas`, `scipy`, `seaborn`, and Jupyter/`ipykernel`. `setup.sh` sets this up automatically as its last step; to (re)run it on its own:

```bash
bash setup_python.sh
```

This checks your Python version (>= 3.12) and warns if `pdflatex`/`dvipng` aren't found (install commands are printed for Debian/Ubuntu and macOS), then creates a `.venv/` virtual environment at the repo root, installs the exact pinned versions from `requirements.txt` (a full `pip freeze` lock, so re-running it always reproduces the same environment), and registers a Jupyter kernel named `qfo-venv` (display name "Python 3 (QuantumFidelityOptimization)") so the notebook runs with the right environment out of the box.

Activate the environment with:

```bash
source .venv/bin/activate
```

or launch the notebook directly without activating:

```bash
.venv/bin/jupyter notebook scripts/plots.ipynb
```

Open `scripts/plots.ipynb`, select the "Python 3 (QuantumFidelityOptimization)" kernel if it isn't already selected, and run all cells — each one displays its figure inline and saves it as a PDF into `figures/`.

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
