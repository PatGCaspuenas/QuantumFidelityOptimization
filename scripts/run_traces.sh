#!/usr/bin/env bash
#
# Run the selected-N full_l1 BO sweep that produces the trace data for paper figures.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  tasks=("$@")
else
  tasks=(
    "full_l1:100"
    "full_l1:1000"
    "full_l1:10000"
    "full_l1:100000"
    "full_l1:Inf"
  )
fi

trace_count() {
  local dir="$1"
  [ -d "$dir" ] || { echo 0; return; }
  find "$dir" -maxdepth 1 -type f -name 'trace_seed*_ucb.csv' | wc -l | tr -d ' '
}

for task in "${tasks[@]}"; do
  score_mode="${task%%:*}"
  n="${task##*:}"
  tag="freqspan10_bound050_${score_mode}_N${n}_nostop100_stream_100seeds"
  trace_dir="$ROOT/data/traces_${tag}"

  done_count="$(trace_count "$trace_dir")"
  if [ "$done_count" -ge 100 ]; then
    echo "=== Skipping ${score_mode} N=${n}: found ${done_count} traces in ${trace_dir} ==="
    continue
  fi

  echo "=== Starting ${score_mode} N=${n}: ${tag} ==="
  mkdir -p "$trace_dir" "$ROOT/data"

  TRACE_N_WORKERS=19 \
  TRACE_N_SHOTS="$n" \
  TRACE_NUM_SIMS=100 \
  TRACE_N_INIT=12 \
  TRACE_N_ITER=100 \
  TRACE_N_RESTARTS=6 \
  TRACE_HYPER_EVERY=10 \
  TRACE_M_ACQ=5000 \
  TRACE_M_REC=20000 \
  TRACE_KAPPA=1.96 \
  TRACE_SCORE_MODE="$score_mode" \
  TRACE_INIT_DESIGN=latin_hypercube \
  TRACE_BOUND_SCALE=0.5 \
  TRACE_FREQ_SPAN_KHZ=10 \
  TRACE_CENTER_JITTER_U_MAX=0.0 \
  TRACE_STREAM_SEED_OUTPUTS=true \
  TRACE_OUTPUT_TAG="$tag" \
  TRACE_OUTPUT_DIR="$trace_dir" \
  TRACE_SCRIPT_DATA_DIR="$ROOT/data" \
  julia --project=. scripts/optimization_data_generation.jl

  echo "=== Finished ${score_mode} N=${n}; traces: ${trace_dir} ==="
done
