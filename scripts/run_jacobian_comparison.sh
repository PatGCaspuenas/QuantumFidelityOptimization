#!/usr/bin/env bash
# Comparison: Q_varMS(M=2) vs Q_jacobian-biased vs Q_jacobian-debiased
# Each run at bounds=1.0 and bounds=0.1 (all three bounds identical per run).
# Settings: N=400, Q_thresh=0.999, 3D (no 4D/2D), stop_mode=two_checks, n_iter=120
#
# ENV overrides:
#   BO_NUM_SIMS   — simulations per run    (default 40)
#   BO_N_WORKERS  — parallel Julia workers (default 20)
#   BO_Q_NOISY_N  — N for final Q_noisy    (default 100000)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"
mkdir -p "$OUTPUT_DIR"

# ── Shared settings ───────────────────────────────────────────────────────────
export BO_N_SHOTS=400
export BO_THRESH_Q=0.999
export BO_VAR_N_MODE=false
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-20}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-100000}"
export BO_OUTPUT_DIR="$OUTPUT_DIR"

export BO_N_INIT=12
export BO_N_ITER=120
export BO_K_ACQ=1
export BO_USE_ZOOM=false
export BO_USE_LBFGS_ACQ=false
export BO_USE_GRAD_ACQ=false
export BO_USE_4D=false
export BO_USE_2D=false
export BO_INIT_METHOD=random
export BO_STOP_MODE=two_checks
export BO_ACQ_TYPE=ucb
export BO_LOG_TRACE=false

TOTAL_RUNS=6
RUN=0

echo "=========================================================="
echo "  Bias comparison: Q_varMS vs Q_jac biased vs debiased"
echo "  N=400, Q_thresh=0.999, 3D, stop=two_checks, n_iter=120"
echo "  sims=$BO_NUM_SIMS, workers=$BO_N_WORKERS"
echo "=========================================================="

# ── Run 1: Q_varMS(M=2), bounds=1.0 ──────────────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  Q_varMS(M=2), bounds=1.0"
export BO_OBJECTIVE_MODE=2ms
export BO_DEBIAS_JAC=false
export BO_A_BOUND=1.0
export BO_FCL_BOUND=1.0
export BO_FSB_BOUND=1.0
export BO_OUTPUT_FILE="benchmark_N400_q999_2ms_3d_bound1.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

# ── Run 2: Q_varMS(M=2), bounds=0.1 ──────────────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  Q_varMS(M=2), bounds=0.1"
export BO_OBJECTIVE_MODE=2ms
export BO_DEBIAS_JAC=false
export BO_A_BOUND=0.1
export BO_FCL_BOUND=0.1
export BO_FSB_BOUND=0.1
export BO_OUTPUT_FILE="benchmark_N400_q999_2ms_3d_bound01.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

# ── Run 3: Q_jacobian biased, bounds=1.0 ─────────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  Q_jacobian biased, bounds=1.0"
export BO_OBJECTIVE_MODE=jacobian
export BO_DEBIAS_JAC=false
export BO_A_BOUND=1.0
export BO_FCL_BOUND=1.0
export BO_FSB_BOUND=1.0
export BO_OUTPUT_FILE="benchmark_N400_q999_jacobian_biased_3d_bound1.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

# ── Run 4: Q_jacobian biased, bounds=0.1 ─────────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  Q_jacobian biased, bounds=0.1"
export BO_OBJECTIVE_MODE=jacobian
export BO_DEBIAS_JAC=false
export BO_A_BOUND=0.1
export BO_FCL_BOUND=0.1
export BO_FSB_BOUND=0.1
export BO_OUTPUT_FILE="benchmark_N400_q999_jacobian_biased_3d_bound01.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

# ── Run 5: Q_jacobian debiased, bounds=1.0 ───────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  Q_jacobian debiased (folded-var noise), bounds=1.0"
export BO_OBJECTIVE_MODE=jacobian
export BO_DEBIAS_JAC=true
export BO_A_BOUND=1.0
export BO_FCL_BOUND=1.0
export BO_FSB_BOUND=1.0
export BO_OUTPUT_FILE="benchmark_N400_q999_jacobian_debiased_3d_bound1.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

# ── Run 6: Q_jacobian debiased, bounds=0.1 ───────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  Q_jacobian debiased (folded-var noise), bounds=0.1"
export BO_OBJECTIVE_MODE=jacobian
export BO_DEBIAS_JAC=true
export BO_A_BOUND=0.1
export BO_FCL_BOUND=0.1
export BO_FSB_BOUND=0.1
export BO_OUTPUT_FILE="benchmark_N400_q999_jacobian_debiased_3d_bound01.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

echo ""
echo "=========================================================="
echo "  All $TOTAL_RUNS runs finished."
echo "  Output files in $OUTPUT_DIR/:"
echo "    benchmark_N400_q999_2ms_3d_bound1.txt"
echo "    benchmark_N400_q999_2ms_3d_bound01.txt"
echo "    benchmark_N400_q999_jacobian_biased_3d_bound1.txt"
echo "    benchmark_N400_q999_jacobian_biased_3d_bound01.txt"
echo "    benchmark_N400_q999_jacobian_debiased_3d_bound1.txt"
echo "    benchmark_N400_q999_jacobian_debiased_3d_bound01.txt"
echo "=========================================================="
