#!/usr/bin/env bash
# Acquisition function comparison for Q_varMS_balance(numMS=3), 3D, N=400, iter120.
#
# Compares: UCB | EI | TS | MES × random init | Sobol init
# Fixed: N=400, threshold=auto, baseline (no zoom/lbfgs), A_bound=0.3, 3D
#
# ENV overrides:
#   BO_NUM_SIMS   — simulations per run    (default 40)
#   BO_N_WORKERS  — parallel Julia workers (default 18)
#   BO_Q_NOISY_N  — N for Q_noisy eval     (default 5000)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"
mkdir -p "$OUTPUT_DIR"

# ── Shared settings ───────────────────────────────────────────────────────────
export BO_N_SHOTS=400
export BO_THRESH_Q=auto
export BO_VAR_N_MODE=false
export BO_OBJECTIVE_MODE=3ms_balance
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-18}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-5000}"
export BO_OUTPUT_DIR="$OUTPUT_DIR"

export BO_N_INIT=20
export BO_N_ITER=120
export BO_K_ACQ=1
export BO_USE_ZOOM=false
export BO_USE_LBFGS_ACQ=false
export BO_USE_GRAD_ACQ=false
export BO_USE_4D=false
export BO_USE_2D=false

export BO_A_BOUND=1.0
export BO_FCL_BOUND=1.0
export BO_FSB_BOUND=1.0

export BO_MES_N=50     # number of f* samples for MES

TOTAL_RUNS=8
RUN=0

echo "=========================================="
echo "  3ms_balance acquisition comparison"
echo "  N=400, A_bound=1.0, 3D, iter120"
echo "  acq: UCB | EI | TS | MES  ×  init: random | sobol"
echo "  sims=$BO_NUM_SIMS, workers=$BO_N_WORKERS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

run_one() {
    local acq="$1" init="$2"
    RUN=$((RUN + 1))
    local label="${acq}_${init}_Abound1"
    local outfile="benchmark_N400_auto_3ms_balance_3d_${label}.txt"
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS]  acq=$acq  init=$init"
    export BO_ACQ_TYPE="$acq"
    export BO_INIT_METHOD="$init"
    export BO_OUTPUT_FILE="$outfile"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] done → $outfile"
}

# Sobol init (better space coverage)
# run_one ucb    sobol #bad
run_one ei     sobol
run_one ts     sobol
run_one mes    sobol

echo ""
echo "=========================================="
echo "  All $TOTAL_RUNS runs finished."
echo "  Results in: $OUTPUT_DIR"
echo "  File pattern: benchmark_N400_auto_3ms_balance_3d_{ucb|ei|ts|mes}_{random|sobol}_Abound1.txt"
echo "=========================================="
