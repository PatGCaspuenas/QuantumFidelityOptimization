#!/usr/bin/env bash
# Master benchmark script — runs all configured benchmark groups sequentially.
#
# Groups (in order):
#   B) 2ms 4D baseline        — full bounds ±1,     iter120
#   C) 2ms 3D+4D bounded      — A_bound=0.3,        iter120 + iter500
#   D) jacobian 3D full       — full bounds ±1,     iter120 + iter500
#   E) jacobian 3D+4D bounded — all bounds ±0.3,    iter120 + iter500
#
# Group A (2ms 3D, full bounds, iter120) was already run:
#   → scripts/data/benchmark_N400_auto_2ms_3d_baseline.txt
#
# To skip a group, comment out its call at the bottom of this file.
#
# ENV overrides (apply to all groups):
#   BO_NUM_SIMS   — simulations per run    (default 40)
#   BO_N_WORKERS  — parallel Julia workers (default 18)
#   BO_Q_NOISY_N  — N for Q_noisy eval     (default 5000)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/opt_benchmark.jl"
OUTPUT_DIR="$REPO_DIR/scripts/data"
mkdir -p "$OUTPUT_DIR"

# ── Shared ENV ────────────────────────────────────────────────────────────────
export BO_N_SHOTS=400
export BO_THRESH_Q=auto
export BO_VAR_N_MODE=false
export BO_NUM_SIMS="${BO_NUM_SIMS:-40}"
export BO_N_WORKERS="${BO_N_WORKERS:-18}"
export BO_Q_NOISY_N="${BO_Q_NOISY_N:-5000}"
export BO_OUTPUT_DIR="$OUTPUT_DIR"
export BO_K_ACQ=1
export BO_USE_ZOOM=false
export BO_USE_LBFGS_ACQ=false
export BO_USE_GRAD_ACQ=false
export BO_USE_2D=false

RUN=0
TOTAL_RUNS=11   # B:1  C:4  D:2  E:4

# ── Generic runner ────────────────────────────────────────────────────────────
# run_one <objective> <dims> <n_init> <n_iter> <fcl_bound> <fsb_bound> <a_bound> <label>
run_one() {
    local obj="$1" dims="$2" n_init="$3" n_iter="$4" \
          fcl_b="$5" fsb_b="$6" a_b="$7" label="$8"
    RUN=$((RUN + 1))
    local outfile="benchmark_N${BO_N_SHOTS}_auto_${obj}_${dims}_${label}.txt"

    export BO_OBJECTIVE_MODE="$obj"
    export BO_USE_4D="false"
    export BO_DEBIAS_JAC="true"
    [ "$dims" = "4d" ] && export BO_USE_4D="true"
    export BO_N_INIT="$n_init"
    export BO_N_ITER="$n_iter"
    export BO_FCL_BOUND="$fcl_b"
    export BO_FSB_BOUND="$fsb_b"
    export BO_A_BOUND="$a_b"
    export BO_OUTPUT_FILE="$outfile"

    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS]  obj=$obj  dims=$dims  n_init=$n_init  n_iter=$n_iter" \
         " bounds=(fcl=$fcl_b,fsb=$fsb_b,A=$a_b)"
    julia --project="$REPO_DIR" "$SCRIPT"
    echo ">>> [$RUN/$TOTAL_RUNS] done → $outfile"
}

# ── Groups ────────────────────────────────────────────────────────────────────

group_E() {
    echo ""; echo "=== GROUP E: jacobian 3D+4D bounded (all ±0.1) ==="
    run_one jacobian 3d 12 120 0.1 0.1 0.1 "iter120_bound01"
    run_one jacobian 4d 12 120 0.1 0.1 0.1 "iter120_bound01"
}
group_C() {
    echo ""; echo "=== GROUP C: jacobian 3D+4D bounded (A_bound=1, fcl/fsb ±1) ==="
    run_one jacobian 3d 12 120 1.0 1.0 1.0 "iter120_bound1"
    run_one jacobian 4d 12 120 1.0 1.0 1.0 "iter120_bound1"
}

# ── Entry point ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "  Master benchmark — groups B C D E"
echo "  N=400, threshold=auto, acq=baseline"
echo "  sims=$BO_NUM_SIMS, workers=$BO_N_WORKERS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

group_E
group_C


echo ""
echo "=========================================="
echo "  All $TOTAL_RUNS runs finished."
echo "  Results in: $OUTPUT_DIR"
echo "=========================================="
