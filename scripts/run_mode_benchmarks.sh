#!/usr/bin/env bash
# Run the new benchmark modes across all N levels.
# Modes 1-4: 3D (no phi), all N levels, 40 sims each
# Mode 5: 4D (phi control), all N levels, 20 sims each
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_qnoisy_3d_parallel_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"

mkdir -p "$OUTPUT_DIR"

export BO_N_WORKERS=4
export BO_OUTPUT_DIR="$OUTPUT_DIR"
export BO_VAR_N_MODE=false
export BO_THRESH_Q=auto

N_LEVELS="50 100 250 500 1000 2500"

# Count total runs: 4 modes × 6 N levels + 1 mode × 6 N levels = 30
TOTAL_RUNS=30
RUN=0

echo "=========================================="
echo "  Running mode benchmarks across N levels"
echo "  N levels: $N_LEVELS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

# --- 1) 2MS log-infidelity, all N levels ---
export BO_OBJECTIVE_MODE=2ms_log
export BO_USE_4D=false
export BO_NUM_SIMS=40
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] 2MS log10(1-F), N=$N"
    export BO_N_SHOTS=$N
    export BO_OUTPUT_FILE="benchmark_2ms_log_N${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] complete."
done

# --- 2) 3MS balance, all N levels ---
export BO_OBJECTIVE_MODE=3ms_balance
export BO_USE_4D=false
export BO_NUM_SIMS=40
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] 3MS balance, N=$N"
    export BO_N_SHOTS=$N
    export BO_OUTPUT_FILE="benchmark_3ms_balance_N${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] complete."
done

# --- 3) 3MS balance log, all N levels ---
export BO_OBJECTIVE_MODE=3ms_balance_log
export BO_USE_4D=false
export BO_NUM_SIMS=40
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] 3MS balance log10(1-F), N=$N"
    export BO_N_SHOTS=$N
    export BO_OUTPUT_FILE="benchmark_3ms_balance_log_N${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] complete."
done

# --- 4) Jacobian-searched sequence, all N levels ---
export BO_OBJECTIVE_MODE=jacobian
export BO_USE_4D=false
export BO_NUM_SIMS=40
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] Jacobian sequence, N=$N"
    export BO_N_SHOTS=$N
    export BO_OUTPUT_FILE="benchmark_jacobian_N${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] complete."
done

# --- 5) 2MS 4D with inter-gate phase, all N levels, 20 sims ---
export BO_OBJECTIVE_MODE=2ms
export BO_USE_4D=true
export BO_NUM_SIMS=20
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] 2MS 4D (phi), N=$N, 20 sims"
    export BO_N_SHOTS=$N
    export BO_OUTPUT_FILE="benchmark_2ms_4d_N${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] complete."
done

echo ""
echo "=========================================="
echo "  All $TOTAL_RUNS mode benchmarks finished."
echo "  Results in: $OUTPUT_DIR"
echo "=========================================="
