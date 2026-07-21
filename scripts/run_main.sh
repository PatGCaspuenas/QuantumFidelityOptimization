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
  
  # Determine which bound scales to run based on N
  if [ "$n" = "Inf" ]; then
    scales=("0.1" "0.5" "1.0")
  else
    scales=("0.5")
  fi

  for scale in "${scales[@]}"; do
    # Format the scale for the output tag string (e.g., 0.5 -> 050)
    case "$scale" in
      0.1) scale_tag="010" ;;
      0.5) scale_tag="050" ;;
      1.0) scale_tag="100" ;;
      *)   echo "Error: Unknown scale $scale"; exit 1 ;;
    esac

    tag="freqspan10_bound${scale_tag}_N${n}_nostop100_stream_100seeds"
    trace_dir="$ROOT/data/traces_${tag}"

    done_count="$(trace_count "$trace_dir")"
    if [ "$done_count" -ge 100 ]; then
      echo "=== Skipping N=${n} (BOUND_SCALE=${scale}): found ${done_count} traces in ${trace_dir} ==="
      continue
    fi

    echo "=== Starting N=${n} (BOUND_SCALE=${scale}): ${tag} ==="
    mkdir -p "$trace_dir" "$ROOT/data"

    N_WORKERS=20 \
    N_SHOTS="$n" \
    NUM_SIMS=100 \
    N_INIT=12 \
    N_ITER=100 \
    N_RESTARTS=6 \
    HYPER_EVERY=10 \
    M_ACQ=5000 \
    M_REC=20000 \
    KAPPA=1.96 \
    BOUND_SCALE="$scale" \
    FREQ_SPAN_KHZ=10 \
    CENTER_JITTER_U_MAX=0.0 \
    OUTPUT_TAG="$tag" \
    OUTPUT_DIR="$trace_dir" \
    SCRIPT_DATA_DIR="$ROOT/data" \
    julia --project=. scripts/main_opt.jl

    echo "=== Finished N=${n} (BOUND_SCALE=${scale}); traces: ${trace_dir} ==="
  done
done