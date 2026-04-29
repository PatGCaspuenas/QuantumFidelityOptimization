#!/usr/bin/env bash
# Acquisition function comparison for Q_varMS_balance(numMS=3), N=400.
#
# Fixed settings: N=400, threshold=auto (1-1/N), baseline multi-start (k=1,
#                 no zoom, no L-BFGS), both 3D and 4D inputs.
#
# Variants:
#   A  ei_full       — EI,  full u_A ∈ [-1, 1]
#   B  ucb_Abound    — UCB, restricted u_A ∈ [-0.3, 0.3]
#   C  ei_Abound     — EI,  restricted u_A ∈ [-0.3, 0.3]
#
# Total: 2 dims × 3 variants = 6 runs.
#
# ENV overrides (set before calling this script):
#   BO_NUM_SIMS   — simulations per run    (default 40)
#   BO_N_WORKERS  — parallel Julia workers (default 20)
#   BO_Q_NOISY_N  — N for Q_noisy eval     (default 5000)
#   BO_A_BOUND    — u_A restriction half-width (default 0.3)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"
mkdir -p "$OUTPUT_DIR"

# ---------- Shared settings ----------
export BO_N_SHOTS=400
export BO_THRESH_Q=auto
export BO_VAR_N_MODE=false
export BO_OBJECTIVE_MODE=3ms_balance
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-20}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-5000}"
export BO_OUTPUT_DIR="$OUTPUT_DIR"

# Baseline acquisition (random scan, k=1)
export BO_K_ACQ=1
export BO_USE_ZOOM=false
export BO_USE_LBFGS_ACQ=false
export BO_USE_GRAD_ACQ=false

# Restricted u_A half-width (override with BO_A_BOUND before calling)
A_BOUND="${BO_A_BOUND:-0.3}"

TOTAL_RUNS=6
RUN=0

echo "=========================================="
echo "  EI vs UCB acquisition comparison"
echo "  3ms_balance, N=400, threshold=auto"
echo "  sims=$BO_NUM_SIMS, workers=$BO_N_WORKERS"
echo "  Q_noisy_N=$BO_Q_NOISY_N"
echo "  A_BOUND (restricted runs) = $A_BOUND"
echo "  Output dir: $OUTPUT_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

# run_one <use_4d> <acq_type> <a_bound> <label>
run_one() {
    local use_4d="$1" acq_type="$2" a_bound="$3" label="$4"
    RUN=$((RUN + 1))
    local nd; [ "$use_4d" = "true" ] && nd="4d" || nd="3d"
    local outfile="benchmark_N400_auto_3ms_balance_${nd}_${label}.txt"
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS]  $nd  acq=$acq_type  a_bound=$a_bound  label=$label"
    export BO_USE_4D="$use_4d"
    export BO_ACQ_TYPE="$acq_type"
    export BO_A_BOUND="$a_bound"
    export BO_OUTPUT_FILE="$outfile"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] done → $outfile"
}

for use_4d in false true; do
    # A: EI, full search space
    run_one "$use_4d"  "ei"  "1.0"    "ei_full"

    # B: UCB, restricted u_A
    run_one "$use_4d"  "ucb" "$A_BOUND" "ucb_Abound${A_BOUND}"

    # C: EI, restricted u_A
    run_one "$use_4d"  "ei"  "$A_BOUND" "ei_Abound${A_BOUND}"
done

echo ""
echo "=========================================="
echo "  All $TOTAL_RUNS runs finished."
echo "  Results in: $OUTPUT_DIR"
echo "  File pattern: benchmark_N400_auto_3ms_balance_{3d|4d}_{variant}.txt"
echo "=========================================="
