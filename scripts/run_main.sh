#!/usr/bin/env bash
#
# Run the selected-N BO sweep that produces the trace data for paper figures.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  tasks=("$@")
else
  tasks=(100 1000 10000 100000 Inf)
fi

trace_count() {
  local dir="$1"
  [ -d "$dir" ] || { echo 0; return; }
  find "$dir" -maxdepth 1 -type f -name 'trace_seed*_ucb.csv' | wc -l | tr -d ' '
}

for task in "${tasks[@]}"; do
  n="$task"
  tag="freqspan10_bound050_N${n}_nostop100_stream_100seeds"
  trace_dir="$ROOT/data/traces_${tag}"

  done_count="$(trace_count "$trace_dir")"
  if [ "$done_count" -ge 100 ]; then
    echo "=== Skipping N=${n}: found ${done_count} traces in ${trace_dir} ==="
    continue
  fi

  echo "=== Starting N=${n}: ${tag} ==="
  mkdir -p "$trace_dir" "$ROOT/data"

  N_WORKERS=19 \
  N_SHOTS="$n" \
  NUM_SIMS=100 \
  N_INIT=12 \
  N_ITER=100 \
  N_RESTARTS=6 \
  HYPER_EVERY=10 \
  M_ACQ=5000 \
  M_REC=20000 \
  KAPPA=1.96 \
  BOUND_SCALE=0.5 \
  FREQ_SPAN_KHZ=10 \
  CENTER_JITTER_U_MAX=0.0 \
  OUTPUT_TAG="$tag" \
  OUTPUT_DIR="$trace_dir" \
  SCRIPT_DATA_DIR="$ROOT/data" \
  julia --project=. scripts/main_opt.jl

  echo "=== Finished N=${n}; traces: ${trace_dir} ==="
done
