#!/usr/bin/env bash
# Sweep N = 50 100 250 400 1000 2500 for Q_varMS (M=2) 3D
# with stop modes: two_checks, lcb, mu_one_check
# Q_threshold = 0.999 (fixed), 40 seeds each → 18 total runs
#
# ENV overrides:
#   BO_NUM_SIMS   — simulations per run    (default 40)
#   BO_N_WORKERS  — parallel Julia workers (default 18)
#   BO_Q_NOISY_N  — N for final Q_noisy    (default 5000)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data/sweep"
mkdir -p "$OUTPUT_DIR"

# ── Shared settings ───────────────────────────────────────────────────────────
export BO_OBJECTIVE_MODE=q_varMS
export BO_THRESH_Q=0.999
export BO_VAR_N_MODE=false
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-20}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-100000}"
export BO_OUTPUT_DIR="$OUTPUT_DIR"

export BO_N_INIT=12
export BO_N_ITER=120
export BO_K_ACQ=1
export BO_ACQ_TYPE=ucb
export BO_USE_ZOOM=false
export BO_USE_LBFGS_ACQ=false
export BO_USE_GRAD_ACQ=false
export BO_USE_4D=false
export BO_USE_2D=false
export BO_INIT_METHOD=random

export BO_A_BOUND=1.0
export BO_FCL_BOUND=1.0
export BO_FSB_BOUND=1.0

N_VALUES=(50 100 250 400 1000 2500)
STOP_MODES=(two_checks lcb mu_one_check)
Q_THRESH=(auto 0.999)

TOTAL_RUNS=$(( ${#N_VALUES[@]} * ${#STOP_MODES[@]} * ${#Q_THRESH[@]} ))
RUN=0

echo "=========================================="
echo "  N-sweep + stop-mode comparison"
echo "  Q_varMS (M=2) 3D, Q_thresh=0.999"
echo "  N in: ${N_VALUES[*]}"
echo "  stop modes: ${STOP_MODES[*]}"
echo "  sims=$BO_NUM_SIMS, workers=$BO_N_WORKERS"
echo "  Total runs: $TOTAL_RUNS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Thresholds: ${Q_THRESH[*]}"
echo "=========================================="

for Q in "${Q_THRESH[@]}"; do
    export BO_THRESH_Q=$Q
    for N in "${N_VALUES[@]}"; do
        export BO_N_SHOTS=$N
        for MODE in "${STOP_MODES[@]}"; do
            RUN=$((RUN + 1))
            export BO_STOP_MODE=$MODE
            export BO_OUTPUT_FILE="benchmark_N${N}_q999_2ms_3d_stop_${MODE}.txt"
            echo ""
            echo ">>> [$RUN/$TOTAL_RUNS]  N=$N  stop_mode=$MODE → ${BO_OUTPUT_FILE}"
            julia --project="$REPO_DIR" "$SCRIPT"
            echo ">>> [$RUN/$TOTAL_RUNS] done"
        done
    done
done

echo ""
echo "=========================================="
echo "  All $TOTAL_RUNS runs finished."
echo "  Results in: $OUTPUT_DIR"
for Q in "${Q_THRESH[@]}"; do
    echo "  Threshold: $Q"
    for N in "${N_VALUES[@]}"; do
        for MODE in "${STOP_MODES[@]}"; do
            echo "    benchmark_N${N}_q999_2ms_3d_stop_${MODE}.txt"
        done
    done
done
echo "=========================================="
