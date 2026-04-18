#!/usr/bin/env bash
# Run benchmark modes 2, 4, 5 across all N levels.
# Mode 2: 3MS balance  Mode 4: Jacobian sequence  Mode 5: 2MS 4D with phi
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"

mkdir -p "$OUTPUT_DIR"

export BO_N_WORKERS=4
export BO_OUTPUT_DIR="$OUTPUT_DIR"
export BO_VAR_N_MODE=false
export BO_THRESH_Q=auto
export BO_NUM_SIMS=5

N_LEVELS="50 100 250 500 1000 2500"

TOTAL_RUNS=18
RUN=0

echo "=========================================="
echo "  Running mode benchmarks (2, 4, 5)"
echo "  N levels: $N_LEVELS"
echo "  Sims per run: $BO_NUM_SIMS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

# --- 2) 3MS balance, all N levels ---
export BO_OBJECTIVE_MODE=3ms_balance
export BO_USE_4D=false
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] 3MS balance F=1-|½-Pss|-|½-Pdd|, N=$N"
    export BO_N_SHOTS=$N
    export BO_OUTPUT_FILE="benchmark_3ms_balance_N${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] complete."
done

# --- 4) Jacobian-searched sequence, all N levels ---
export BO_OBJECTIVE_MODE=jacobian
export BO_USE_4D=false
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] Jacobian sequence, N=$N"
    export BO_N_SHOTS=$N
    export BO_OUTPUT_FILE="benchmark_jacobian_N${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] complete."
done

# --- 5) 2MS 4D with inter-gate phase, all N levels ---
export BO_OBJECTIVE_MODE=2ms
export BO_USE_4D=true
for N in $N_LEVELS; do
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] 2MS 4D (phi), N=$N"
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
