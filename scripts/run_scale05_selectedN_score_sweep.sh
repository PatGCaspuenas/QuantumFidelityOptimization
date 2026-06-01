#!/usr/bin/env bash
#
# Driver for optimization_data_generation.jl. Produces the selected-N full_l1 BO
# trace dirs (data/traces_freqspan10_bound050_full_l1_N*_nostop100_stream_40seeds)
# feeding paper figures figure_ab_vertical.png and figure_fcl_amp_n100_gpzoom.png.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  tasks=("$@")
else
  tasks=(
    "odd_penalty:10000"
    "odd_penalty:100000"
    "full_l1:100"
    "full_l1:1000"
    "full_l1:10000"
    "full_l1:100000"
    "full_l1:Inf"
  )
fi

trace_count() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f -name 'trace_seed*_ucb.csv' | wc -l | tr -d ' '
}

tag_for() {
  local score_mode="$1"
  local n="$2"
  if [ "$score_mode" = "odd_penalty" ]; then
    echo "freqspan10_bound050_N${n}_nostop100_stream_40seeds"
  else
    echo "freqspan10_bound050_${score_mode}_N${n}_nostop100_stream_40seeds"
  fi
}

for task in "${tasks[@]}"; do
  score_mode="${task%%:*}"
  n="${task##*:}"
  tag="$(tag_for "$score_mode" "$n")"
  trace_dir="$ROOT/data/traces_${tag}"
  gp_dir="$ROOT/data/gp_diagnostics_${tag}"
  script_data_dir="$ROOT/scripts/data"
  done_count="$(trace_count "$trace_dir")"

  if [ "$done_count" -ge 40 ]; then
    echo "=== Skipping ${score_mode} N=${n}: found ${done_count} trace files in ${trace_dir} ==="
    continue
  fi

  echo "=== Starting ${score_mode} N=${n}: ${tag} ==="
  mkdir -p "$trace_dir" "$script_data_dir"

  TRACE_N_WORKERS=5 \
  TRACE_N_SHOTS="$n" \
  TRACE_NUM_SIMS=40 \
  TRACE_N_INIT=12 \
  TRACE_N_ITER=100 \
  TRACE_N_RESTARTS=6 \
  TRACE_HYPER_EVERY=10 \
  TRACE_M_ACQ=5000 \
  TRACE_M_REC=20000 \
  TRACE_KAPPA=1.9 \
  TRACE_SCORE_MODE="$score_mode" \
  TRACE_INIT_DESIGN=latin_hypercube \
  TRACE_BOUND_SCALE=0.5 \
  TRACE_FREQ_SPAN_KHZ=10 \
  TRACE_CENTER_JITTER_U_MAX=0.0 \
  TRACE_DISABLE_STOPPING=true \
  TRACE_REQUIRE_SURROGATE_CONFIDENCE=true \
  TRACE_STOPPING_RULE=surrogate_then_noisy \
  TRACE_THRESH_Q=1.0 \
  TRACE_CONFIDENCE_Z=0.98 \
  TRACE_RESTART_ON_STAGNATION=true \
  TRACE_RESTART_FAILURE_WINDOW=8 \
  TRACE_RESTART_MIN_ACTIVE_ITER=16 \
  TRACE_RESTART_MIN_DELTA=0.01 \
  TRACE_RESTART_DISABLE_LCB=0.8 \
  TRACE_RESTART_MAX_RESTARTS=5 \
  TRACE_COUNT_RESTART_INIT_IN_ITER_BUDGET=false \
  TRACE_RUN_SLICES=false \
  TRACE_GP_DIAG=false \
  TRACE_STREAM_SEED_OUTPUTS=true \
  TRACE_OUTPUT_TAG="$tag" \
  TRACE_OUTPUT_DIR="$trace_dir" \
  TRACE_GP_DIAG_DIR="$gp_dir" \
  TRACE_SCRIPT_DATA_DIR="$script_data_dir" \
  julia --project=. scripts/optimization_data_generation.jl

  echo "=== Finished ${score_mode} N=${n}; traces: ${trace_dir} ==="
done
