#!/usr/bin/env bash
# Budget sensitivity benchmark for Q_varMS_balance(numMS=3).
#
# Fixed settings: N=400, threshold=auto (1-1/N), baseline acquisition (k=1,
#                 no zoom, no L-BFGS), both 3D and 4D inputs.
#
# Budget variants:
#   iter500      — n_init=12 (default),  n_iter=500
#   init50       — n_init=50,            n_iter=120 (default)
#   init50_iter500 — n_init=50,          n_iter=500
#
# Total: 2 dims × 3 budget variants = 6 runs.
#
# ENV overrides (set before calling this script):
#   BO_NUM_SIMS   — simulations per run    (default 40)
#   BO_N_WORKERS  — parallel Julia workers (default 20)
#   BO_Q_NOISY_N  — N for Q_noisy eval     (default 5000)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"
mkdir -p "$OUTPUT_DIR"

# ---------- Shared settings ----------
export BO_N_SHOTS=1000
export BO_THRESH_Q=auto
export BO_VAR_N_MODE=false
export BO_OBJECTIVE_MODE=3ms_balance
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-18}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-5000}"
export BO_OUTPUT_DIR="$OUTPUT_DIR"

# Baseline acquisition
export BO_K_ACQ=1
export BO_USE_ZOOM=false
export BO_USE_LBFGS_ACQ=false
export BO_USE_GRAD_ACQ=false

TOTAL_RUNS=4
RUN=0

echo "=========================================="
echo "  Balance 3ms budget benchmark"
echo "  N=400, threshold=auto, acq=baseline, iter=200"
echo "  sims=$BO_NUM_SIMS, workers=$BO_N_WORKERS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

# run_one <dims: 2d|3d> <n_init> <n_iter> <n_shots> <label>
run_one() {
    local dims="$1" n_init="$2" n_iter="$3" n_shots="$4" label="$5"
    RUN=$((RUN + 1))
    local outfile="benchmark_N${n_shots}_auto_3ms_balance_${dims}_${label}.txt"
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS]  $dims  n_init=$n_init  n_iter=$n_iter  N=$n_shots  label=$label"
    export BO_USE_4D="false"
    export BO_USE_2D="false"
    [ "$dims" = "4d" ] && export BO_USE_4D="true"
    [ "$dims" = "2d" ] && export BO_USE_2D="true"
    export BO_N_SHOTS="$n_shots"
    export BO_N_INIT="$n_init"
    export BO_N_ITER="$n_iter"
    export BO_OUTPUT_FILE="$outfile"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] done → $outfile"
}

# N=400, 200 iterations, 2D (f_cl + f_sb only, A and phi fixed)
run_one "2d"  12 120 400 "iter120"

# N=400, 200 iterations, 3D (f_cl + f_sb + A, phi fixed)
run_one "3d"  12 120 400 "iter120"

echo ""
echo "=========================================="
echo "  All $TOTAL_RUNS runs finished."
echo "  Results in: $OUTPUT_DIR"
echo "  File pattern: benchmark_N{N}_auto_3ms_balance_{2d|3d|4d}_{variant}.txt"
echo "=========================================="
