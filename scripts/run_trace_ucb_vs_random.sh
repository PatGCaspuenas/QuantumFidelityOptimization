#!/usr/bin/env bash
# N=400, Q_varMS (M=2) 3D, Q_thresh=0.999, stop_mode=mu_one_check
# Two runs: UCB acquisition vs random acquisition
# For each of 40 seeds: saves x_acq, x_rec, m_rec, s_rec, y_acq, y_check per iteration
# Traces written to: scripts/data/traces/trace_seed<SEED>_<acq>.csv
#
# ENV overrides:
#   BO_NUM_SIMS   — simulations per run    (default 40)
#   BO_N_WORKERS  — parallel Julia workers (default 18)
#   BO_Q_NOISY_N  — N for final Q_noisy    (default 5000)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"
mkdir -p "$OUTPUT_DIR"

# ── Shared settings ───────────────────────────────────────────────────────────
export BO_OBJECTIVE_MODE=q_varMS
export BO_N_SHOTS=400
export BO_THRESH_Q=0.999
export BO_VAR_N_MODE=false
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-18}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-5000}"
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
export BO_STOP_MODE=mu_one_check

export BO_A_BOUND=1.0
export BO_FCL_BOUND=1.0
export BO_FSB_BOUND=1.0

export BO_LOG_TRACE=true

TOTAL_RUNS=2
RUN=0

echo "=========================================="
echo "  Trace study: UCB vs random acquisition"
echo "  Q_varMS (M=2) 3D, N=400, Q_thresh=0.999"
echo "  stop_mode=mu_one_check, n_iter=120"
echo "  sims=$BO_NUM_SIMS, workers=$BO_N_WORKERS"
echo "  Traces → $OUTPUT_DIR/traces/"
echo "=========================================="

# ── Run 1: UCB acquisition ────────────────────────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  acq_type=ucb"
export BO_ACQ_TYPE=ucb
export BO_OUTPUT_FILE="benchmark_N400_q999_2ms_3d_trace_ucb.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

# ── Run 2: random acquisition ─────────────────────────────────────────────────
RUN=$((RUN + 1))
echo ""
echo ">>> [$RUN/$TOTAL_RUNS]  acq_type=random"
export BO_ACQ_TYPE=random
export BO_OUTPUT_FILE="benchmark_N400_q999_2ms_3d_trace_random.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> [$RUN/$TOTAL_RUNS] done → ${BO_OUTPUT_FILE}"

echo ""
echo "=========================================="
echo "  Both runs finished."
echo "  Summary files:"
echo "    benchmark_N400_q999_2ms_3d_trace_ucb.txt"
echo "    benchmark_N400_q999_2ms_3d_trace_random.txt"
echo "  Per-seed traces:"
echo "    $OUTPUT_DIR/traces/trace_seed*_ucb.csv"
echo "    $OUTPUT_DIR/traces/trace_seed*_random.csv"
echo "=========================================="
