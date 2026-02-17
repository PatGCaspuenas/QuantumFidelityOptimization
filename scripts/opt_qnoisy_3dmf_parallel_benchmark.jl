using Random
using Distributed
import Pkg

# Add worker processes (use all available cores)
if nprocs() == 1
    addprocs()
end

Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

@everywhere include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
@everywhere using .CalibrationCode
@everywhere using Statistics

# Keep stderr visible for debugging
# redirect_stderr(devnull)

# -------------------------
# Baseline
# -------------------------
const t = 100.0
base = CalibrationCode.ideal(t)
const f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

# spans around baseline
const span_fcl = 3e4
const span_fsb = 3e4
const span_A   = 6e4

# Broadcast constants to all workers
@everywhere const t = $t
@everywhere const f_cl0 = $f_cl0
@everywhere const f_sb0 = $f_sb0
@everywhere const A0 = $A0
@everywhere const span_fcl = $span_fcl
@everywhere const span_fsb = $span_fsb
@everywhere const span_A = $span_A

# u ∈ [-1,1]^3 -> physical params
@everywhere u_to_params(u) = (f_cl0 + span_fcl*u[1],
                               f_sb0 + span_fsb*u[2],
                               A0    + span_A  *u[3])

# noise knob -> shots (tunable)
@everywhere function N_from_sigma(σ::Float64)
    # keep bounded for runtime sanity
    N = round(Int, 1 / (σ^2))
    return clamp(N, 20, 10000)
end

# Multi-fidelity version: u is 3D, ℓ is fidelity index (1=numMS 2, 2=numMS 6, 3=numMS 10)
@everywhere function Q_fun_mf(u, ℓ)
    fcl, fsb, A = u_to_params(u)
    numMS_levels = [2, 6, 10]
    numMS = numMS_levels[ℓ]
    σ = 0.01  # Use fixed noise level for simplicity
    return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=100, numMS=numMS)
end

@everywhere function Q_true(u)
    fcl, fsb, A = u_to_params(u)
    return CalibrationCode.Q_det(t, fcl, fsb, A)
end

bounds   = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
# Multi-fidelity levels: z_levels normalized to [0,1], costs reflect relative expense
z_levels = [0.0, 0.5, 1.0]  # Fidelity levels (low, medium, high)
costs = [1.0, 2.5, 4.0]  # Relative costs for numMS = 2, 6, 10

# -------------------------
# Hyperparameter Sweep (Multi-Fidelity)
# -------------------------
num_sims = 40
fidelity_threshold = 0.998
n_init = 12
n_iter = 100

println("=== Multi-Fidelity Benchmark ===")
println("Threshold = $fidelity_threshold")
println("n_init = $n_init, n_iter = $n_iter")
println("Sims per trial = $num_sims")
println("Fidelity levels: numMS = 2, 6, 10")
flush(stdout)

# Broadcast MF constants to workers
@everywhere bounds = $bounds
@everywhere z_levels = $z_levels
@everywhere costs = $costs
@everywhere n_init = $n_init
@everywhere n_iter = $n_iter
@everywhere fidelity_threshold = $fidelity_threshold

# Run one multi-fidelity optimization per simulation
@everywhere function run_one_mf_sim(seed::Int)
    res = CalibrationCode.bayesopt_mf(Q_fun_mf;
        bounds=bounds,
        z_levels=z_levels,
        costs=costs,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
        fidelity_threshold=fidelity_threshold
    )
    println("Completed MF simulation in ", res.n_iter_actual, " iterations.")
    flush(stdout)
    # Extract fidelity level usage from Xa (last row contains z values)
    z_observed = res.Xa[end, :]  # Last row of Xa contains z values
    z_levels_vec = collect(z_levels)
    
    # Count which fidelity level each z corresponds to
    level_counts = [0, 0, 0]  # counts for numMS = 2, 6, 10
    for z in z_observed
        # Find closest z_level
        dists = abs.(z .- z_levels_vec)
        closest_idx = argmin(dists)
        level_counts[closest_idx] += 1
    end
    
    Q_det_val = Q_true(res.x_rec)
    return (
        seed=seed,
        x_rec=res.x_rec,
        Q_det=Q_det_val,
        Q_mf=res.y_rec,
        n_iter_actual=res.n_iter_actual,
        level_2_count=level_counts[1],
        level_6_count=level_counts[2],
        level_10_count=level_counts[3]
    )
end

seeds = rand(1:1_000_000, num_sims)

mf_results = pmap(seeds; batch_size=1) do seed
    println("Running MF optimization...")
    flush(stdout)
    run_one_mf_sim(seed)
end

println("\nCompleted all multi-fidelity optimizations.\n")
flush(stdout)

# Extract statistics
Q_det_values = [r.Q_det for r in mf_results]
avg_Q_det = mean(Q_det_values)
std_Q_det = std(Q_det_values)
iter_values = [r.n_iter_actual for r in mf_results]
avg_iters = mean(iter_values)
std_iters = std(iter_values)

# Count fidelity level usage across all runs
total_level_2 = sum(r.level_2_count for r in mf_results)
total_level_6 = sum(r.level_6_count for r in mf_results)
total_level_10 = sum(r.level_10_count for r in mf_results)
total_evals = total_level_2 + total_level_6 + total_level_10

println("="^50)
println("=== MULTI-FIDELITY OPTIMIZATION RESULTS ===")
println("="^50)
println("Average Q_det = ", avg_Q_det, " ± ", std_Q_det)
println("Min Q_det = ", minimum(Q_det_values))
println("Max Q_det = ", maximum(Q_det_values))
println()
println("=== ITERATIONS ===")
println("Average iterations = ", avg_iters, " ± ", std_iters)
println("Min iterations = ", minimum(iter_values))
println("Max iterations = ", maximum(iter_values))
println()
println("=== FIDELITY LEVEL USAGE ===")
println("numMS=2 (low):  $total_level_2 calls ($(round(100*total_level_2/total_evals; digits=1))%)")
println("numMS=6 (med):  $total_level_6 calls ($(round(100*total_level_6/total_evals; digits=1))%)")
println("numMS=10 (hi):  $total_level_10 calls ($(round(100*total_level_10/total_evals; digits=1))%)")
println("Total:          $total_evals calls")
println("="^50)
flush(stdout)

# -------------------------
# Write Results to File
# -------------------------
output_file = joinpath(@__DIR__, "mf_benchmark_results.txt")
open(output_file, "w") do io
    println(io, "=== MULTI-FIDELITY OPTIMIZATION RESULTS ===")
    println(io, "Fidelity levels: numMS = 2, 6, 10")
    println(io, "Number of simulations = $num_sims")
    println(io, "n_init = $n_init, n_iter = $n_iter")
    println(io, "")
    println(io, "=== SUMMARY STATISTICS ===")
    println(io, "Average Q_det = $avg_Q_det ± $std_Q_det")
    println(io, "Min Q_det = $(minimum(Q_det_values))")
    println(io, "Max Q_det = $(maximum(Q_det_values))")
    println(io, "")
    println(io, "=== ITERATIONS ===")
    println(io, "Average iterations = $avg_iters ± $std_iters")
    println(io, "Min iterations = $(minimum(iter_values))")
    println(io, "Max iterations = $(maximum(iter_values))")
    println(io, "")
    println(io, "=== FIDELITY LEVEL USAGE (AGGREGATE) ===")
    println(io, "numMS=2 (low):  $total_level_2 calls ($(round(100*total_level_2/total_evals; digits=1))%)")
    println(io, "numMS=6 (med):  $total_level_6 calls ($(round(100*total_level_6/total_evals; digits=1))%)")
    println(io, "numMS=10 (hi):  $total_level_10 calls ($(round(100*total_level_10/total_evals; digits=1))%)")
    println(io, "Total:          $total_evals calls")
    println(io, "")
    println(io, "=== INDIVIDUAL RESULTS ===")
    println(io, "Sim\tSeed\tIterations\tQ_det\tnumMS=2\tnumMS=6\tnumMS=10")
    for (i, r) in enumerate(mf_results)
        println(io, "$i\t$(r.seed)\t$(r.n_iter_actual)\t$(r.Q_det)\t$(r.level_2_count)\t$(r.level_6_count)\t$(r.level_10_count)")
    end
end

println("\nResults written to: $output_file")
flush(stdout)
