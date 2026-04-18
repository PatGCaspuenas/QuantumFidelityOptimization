#!/usr/bin/env bash
# Run all 7 benchmark configurations sequentially.
# Output goes to scripts/data/
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"

mkdir -p "$OUTPUT_DIR"

export BO_NUM_SIMS=40
export BO_N_WORKERS=5
export BO_OUTPUT_DIR="$OUTPUT_DIR"

echo "=========================================="
echo "  Running 7 benchmark configurations"
echo "  Output dir: $OUTPUT_DIR"
echo "=========================================="

# --- Fixed N runs ---
for N in 50 100 250 500 1000 2500; do
    echo ""
    echo ">>> Fixed N=$N  (threshold = auto = 1-1/$N)"
    export BO_N_SHOTS=$N
    export BO_THRESH_Q=auto
    export BO_VAR_N_MODE=false
    export BO_OUTPUT_FILE="benchmark_fixedN${N}.txt"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> Fixed N=$N complete."
done

# --- Variable N run ---
echo ""
echo ">>> Variable N  (n_floor=50, n_max=2500, threshold=0.9996)"
export BO_N_SHOTS=50
export BO_THRESH_Q=0.9996
export BO_VAR_N_MODE=true
export BO_N_FLOOR=50
export BO_N_MAX=2500
export BO_OUTPUT_FILE="benchmark_varN_max2500.txt"
julia --project="$REPO_DIR" "$SCRIPT"
echo ">>> Variable N complete."

echo ""
echo "=========================================="
echo "  All 7 benchmarks finished."
echo "  Results in: $OUTPUT_DIR"
echo "=========================================="
