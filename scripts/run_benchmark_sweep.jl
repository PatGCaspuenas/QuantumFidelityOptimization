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
#   BO_THRESH_Q      = "0.998" or "" (empty = derive from N)
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
#   BO_NUM_SIMS      = "40", ...
#   BO_OUTPUT_FILE   = filename inside data/
#   BO_N_WORKERS     = number of parallel workers (overrides n_workers_per_run if set here)
#   BO_INIT_SAMPLING   = "random" / "sobol"  (design points)
#   BO_ACQ_SAMPLING    = "random" / "sobol"  (acquisition candidates per BO step)
#   BO_FIXED_INIT_SEED = "1" or ""           ("" = init varies with run seed; integer = shared init for all runs)
#   BO_N_RESTARTS      = "6", ...            (MLE multi-start restarts per hyper fit; final fit uses n_restarts+2)
# ============================================================
COMBINATIONS = [

    (
        label           = "linear_simple_LFBGS_Ncheck2_add_N50",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "simple",
        BO_THRESH_Q     = "",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "50",
        BO_OUTPUT_FILE  = "benchmark_results_N50_linear_simple_LFBGS_Ncheck2_add.txt",
    ),

        (
        label           = "linear_simple_LFBGS_Ncheck2_add_N100",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "simple",
        BO_THRESH_Q     = "",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "100",
        BO_OUTPUT_FILE  = "benchmark_results_N100_linear_simple_LFBGS_Ncheck2_add.txt",
    ),

    (
        label           = "linear_simple_LFBGS_Ncheck2_add_N250",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "simple",
        BO_THRESH_Q     = "",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "250",
        BO_OUTPUT_FILE  = "benchmark_results_N250_linear_simple_LFBGS_Ncheck2_add.txt",
    ),

    (
        label           = "linear_simple_LFBGS_Ncheck2_add_N2500",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "simple",
        BO_THRESH_Q     = "",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "2500",
        BO_OUTPUT_FILE  = "benchmark_results_N2500_linear_simple_LFBGS_Ncheck2_add.txt",
    ),

    (
        label           = "linear_simple_LFBGS_Ncheck2_add_N10000",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "simple",
        BO_THRESH_Q     = "",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "10000",
        BO_OUTPUT_FILE  = "benchmark_results_N10000_linear_simple_LFBGS_Ncheck2_add.txt",
    ),

        (
        label           = "linear_binomial_LFBGS_Ncheck2_add_N50",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "binomial",
        BO_THRESH_Q     = "0.998",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "50",
        BO_OUTPUT_FILE  = "benchmark_results_N50_linear_binomial_LFBGS_Ncheck2_add_0998.txt",
    ),

        (
        label           = "linear_binomial_LFBGS_Ncheck2_add_N100",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "binomial",
        BO_THRESH_Q     = "0.998",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "100",
        BO_OUTPUT_FILE  = "benchmark_results_N100_linear_binomial_LFBGS_Ncheck2_add_0998.txt",
    ),

    (
        label           = "linear_binomial_LFBGS_Ncheck2_add_N250",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "binomial",
        BO_THRESH_Q     = "0.998",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "250",
        BO_OUTPUT_FILE  = "benchmark_results_N250_linear_binomial_LFBGS_Ncheck2_add_0998.txt",
    ),
    (
        label           = "linear_binomial_LFBGS_Ncheck2_add_N400",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "binomial",
        BO_THRESH_Q     = "0.998",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "400",
        BO_OUTPUT_FILE  = "benchmark_results_N400_linear_binomial_LFBGS_Ncheck2_add_0998.txt",
    ),

    (
        label           = "linear_binomial_LFBGS_Ncheck2_add_N2500",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "binomial",
        BO_THRESH_Q     = "0.998",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "2500",
        BO_OUTPUT_FILE  = "benchmark_results_N2500_linear_binomial_LFBGS_Ncheck2_add_0998.txt",
    ),

    (
        label           = "linear_binomial_LFBGS_Ncheck2_add_N10000",
        BO_LOG_FID      = "false",
        BO_SIGMA_MODE   = "binomial",
        BO_THRESH_Q     = "0.998",
        BO_HYPER_EVERY  = "10",
        BO_ADD_CHECK_POINTS = "true",
        BO_N_CHECKS      = "2",
        BO_N_SHOTS       = "10000",
        BO_OUTPUT_FILE  = "benchmark_results_N10000_linear_binomial_LFBGS_Ncheck2_add_0998.txt",
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
