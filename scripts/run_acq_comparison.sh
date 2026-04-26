#!/usr/bin/env bash
# Acquisition strategy comparison benchmark.
#
# Fixed settings: N=400, threshold=auto (1-1/N), 3D and 4D inputs.
#
# Objectives (no jacobian):
#   2ms          → Q_varMS(numMS=2)
#   3ms          → Q_varMS(numMS=3)
#   3ms_balance  → Q_varMS_balance(numMS=3)
#
# Acquisition variants (A–D):
#   A  baseline   — k=1,  no zoom, no L-BFGS  (pure random scan, original)
#   B  zoom_k10   — k=10, zoom,    no L-BFGS
#   C  lbfgs_k10  — k=10, no zoom, L-BFGS (finite-diff gradients)
#   D  lbfgs_grad_k10 — k=10, no zoom, L-BFGS + analytical gradients
#
# Total: 4 acq variants × 6 mode/dim combos = 24 runs.
#
# ENV overrides (set before calling this script):
#   BO_NUM_SIMS    — simulations per run      (default 20)
#   BO_N_WORKERS   — parallel Julia workers   (default 4)
#   BO_Q_NOISY_N   — N for Q_noisy benchmark  (default 5000)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"
mkdir -p "$OUTPUT_DIR"

# ---------- Shared settings ----------
export BO_N_SHOTS=400
export BO_THRESH_Q=auto
export BO_VAR_N_MODE=false
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-20}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-5000}"
export BO_OUTPUT_DIR="$OUTPUT_DIR"

TOTAL_RUNS=24
RUN=0

echo "=========================================="
echo "  Acquisition comparison benchmark"
echo "  N=400, threshold=auto, sims=$BO_NUM_SIMS"
echo "  Workers=$BO_N_WORKERS, Q_noisy_N=$BO_Q_NOISY_N"
echo "  Output dir: $OUTPUT_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

# run_one <mode> <use_4d> <k_acq> <use_zoom> <use_lbfgs> <use_grad> <label>
run_one() {
    local mode="$1" use_4d="$2" k_acq="$3" use_zoom="$4" use_lbfgs="$5" use_grad="$6" label="$7"
    RUN=$((RUN + 1))
    local nd; [ "$use_4d" = "true" ] && nd="4d" || nd="3d"
    local outfile="benchmark_N400_auto_${mode}_${nd}_${label}.txt"
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS]  mode=$mode  $nd  acq=$label"
    export BO_OBJECTIVE_MODE="$mode"
    export BO_USE_4D="$use_4d"
    export BO_K_ACQ="$k_acq"
    export BO_USE_ZOOM="$use_zoom"
    export BO_USE_LBFGS_ACQ="$use_lbfgs"
    export BO_USE_GRAD_ACQ="$use_grad"
    export BO_OUTPUT_FILE="$outfile"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] done → $outfile"
}

# ---------- Loop over objectives and dimensions ----------
for mode in 2ms 3ms 3ms_balance; do
    for use_4d in false true; do

        # A: baseline — pure random scan (original behavior)
        run_one "$mode" "$use_4d"  1 false false false "baseline"

        # B: zoom k=10 — separated multi-start + zoom random search, no L-BFGS
        run_one "$mode" "$use_4d" 10 true  false false "zoom_k10"

        # C: L-BFGS k=10 — separated multi-start + L-BFGS, finite-diff gradients
        run_one "$mode" "$use_4d" 10 false true  false "lbfgs_k10"

        # D: L-BFGS + analytical gradients k=10
        run_one "$mode" "$use_4d" 10 false true  true  "lbfgs_grad_k10"

    done
done

echo ""
echo "=========================================="
echo "  All $TOTAL_RUNS runs finished."
echo "  Results in: $OUTPUT_DIR"
echo "  File pattern: benchmark_N400_auto_{mode}_{3d|4d}_{variant}.txt"
echo "=========================================="
