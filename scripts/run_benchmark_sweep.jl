# scripts/run_benchmark_sweep.jl
#
# Runs opt_qnoisy_3d_parallel_benchmark.jl for multiple config combinations.
# Each combination is launched as a separate Julia subprocess so that
# @everywhere const declarations don't clash across runs.
#
# Usage:
#   julia scripts/run_benchmark_sweep.jl
#
# Adjust the COMBINATIONS table below to define what to sweep over.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

# ============================================================
# HOW MANY WORKERS PER RUN
# Each subprocess will use this many parallel workers.
# Total CPU usage = n_workers_per_run (runs are sequential).
# ============================================================
n_workers_per_run = max(1, Sys.CPU_THREADS - 1)

# ============================================================
# SWEEP COMBINATIONS
# Each entry is a NamedTuple with the fields you want to vary.
# Fields not listed here fall back to benchmark defaults.
#
# Available ENV keys and their types:
#   BO_LOG_FID       = "true"/"false"
#   BO_SIGMA_MODE    = "simple"/"binomial"
#   BO_THRESH_Q      = "0.998" (fixed) / "auto" (= 1−1/N) / "" (no threshold, fixed iterations)
#   BO_USE_PRETRAIN  = "true"/"false"
#   BO_PRETRAIN_FILE = filename inside data/  (e.g. "pretrained_theta_3d.jl")
#   BO_FREEZE_MODE   = "none"/"lengthscales"/"lengthscales_from"/"all"/"all_from"
#                      none             → full MLE every hyper_every steps
#                      lengthscales     → hold pretrained ℓ fixed for n_freeze_iters, then release
#                      lengthscales_from→ learn ℓ freely for n_freeze_iters, fix at transition
#                      all              → freeze all hypers (use pretrained_θ throughout)
#                      all_from         → learn all hypers freely for n_freeze_iters, freeze everything at transition
#   BO_N_FREEZE      = "20","40", ...  (iterations before freeze kicks in / is released)
#   BO_HYPER_EVERY   = "1","5","10", ...
#   BO_N_SHOTS       = "400", ...
#   BO_N_INIT        = "12", ...  (number of initial design points)
#   BO_NUM_SIMS      = "40", ...
#   BO_OUTPUT_FILE   = filename inside data/
#   BO_N_WORKERS     = number of parallel workers (overrides n_workers_per_run if set here)
#   BO_INIT_SAMPLING   = "random" / "sobol"  (design points)
#   BO_ACQ_SAMPLING    = "random" / "sobol"  (acquisition candidates per BO step)
#   BO_FIXED_INIT_SEED = "1" or ""           ("" = init varies with run seed; integer = shared init for all runs)
#   BO_FIXED_ACQ_SEED  = "1" or ""           ("" = acq candidates vary with run seed; integer = shared acq candidates across all runs)
#   BO_FIXED_REC_SEED  = "1" or ""           ("" = rec candidates vary with run seed; integer = shared GP mean candidates across all runs)
#   BO_N_RESTARTS      = "6", ...            (MLE multi-start restarts per hyper fit; final fit uses n_restarts+2)
#   BO_N_ITER          = "100", ...          (max BO iterations)
#   BO_VAR_N_MODE      = "none"/"s2"/"mean"/"ci"/"verify"  (variable-N mode; none = fixed N_shots)
#   BO_N_FLOOR         = "50"               (minimum shots for variable-N modes)
#   BO_N_MAX           = "2000"             (maximum shots for variable-N modes)
#   BO_N_PRECISION     = "0.05"             (c in N=1/(c²(1−Q)) for :mean; also used for acq in :ci/:verify with acq_n_mode=mean)
#   BO_ACQ_N_MODE      = "floor"/"mean"     (shot selection for acquisition point in :ci/:verify modes)
#   BO_SIGMA_LEVELS    = "0.1412,0.1,0.06,0.04472"  (σ levels for :s2 mode)
# ============================================================
COMBINATIONS = [

    (
        label          = "noisy_N400_verify_floor",
        BO_OPT_DET     = "false",
        BO_SIGMA_MODE  = "binomial",
        BO_THRESH_Q    = "0.998",
        BO_HYPER_EVERY = "10",
        BO_VAR_N_MODE  = "ci",
        BO_ACQ_N_MODE  = "floor",
        BO_N_FLOOR     = "50",
        BO_N_MAX       = "2000",
        BO_MIN_ITER    = "20",
        BO_OUTPUT_FILE = "benchmark_results_Nvar_v2_ci_floor.txt",
    ),
]

# ============================================================
# RUN
# ============================================================

benchmark_script = joinpath(@__DIR__, "opt_qnoisy_3d_parallel_benchmark.jl")
julia_exe = Base.julia_cmd()[1]   # path to current julia executable

n_total = length(COMBINATIONS)
println("=== Benchmark Sweep: $n_total combinations, $n_workers_per_run workers each ===\n")

for (i, cfg) in enumerate(COMBINATIONS)
    label = cfg.label
    println("[$i/$n_total] Running: $label")
    flush(stdout)

    # Build ENV dict from the NamedTuple (skip :label key)
    env_dict = copy(ENV)
    env_dict["BO_N_WORKERS"] = string(n_workers_per_run)
    for (k, v) in pairs(cfg)
        k === :label && continue
        env_dict[string(k)] = string(v)
    end

    t0 = time()
    try
        run(setenv(`$julia_exe --project=$(joinpath(@__DIR__, "..")) $benchmark_script`, env_dict))
        elapsed = round(time() - t0, digits=1)
        println("[$i/$n_total] ✓ Done in $(elapsed)s: $label\n")
    catch e
        elapsed = round(time() - t0, digits=1)
        println("[$i/$n_total] ✗ FAILED after $(elapsed)s: $label")
        println("  Error: $e\n")
    end
    flush(stdout)
end

println("=== Sweep complete ===")
