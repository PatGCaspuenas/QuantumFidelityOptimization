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

@everywhere function Q_fun(u, σ)
    fcl, fsb, A = u_to_params(u)
    return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N_from_sigma(σ), numMS = 2)
end

@everywhere function Q_true(u)
    fcl, fsb, A = u_to_params(u)
    return CalibrationCode.Q_det(t, fcl, fsb, A)
end

σ_levels = [0.1, 0.05, 0.02, 0.01]
bounds   = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

# -------------------------
# Hyperparameter Sweep
# -------------------------
α_values = [0.6, 0.8]
κ_values = [1.5, 2.0]
num_sims = 2
fidelity_threshold = 0.997
n_init = 12
n_iter = 100

println("=== Hyperparameter Sweep ===")
println("Threshold = $fidelity_threshold")
println("n_init = $n_init, n_iter = $n_iter")
println("Sims per (α, κ) = $num_sims")
println("Using $(nprocs()) processes\n")

param_pairs = [(α, κ) for α in α_values for κ in κ_values]
total_pairs = length(param_pairs)

println("Total hyperparameter pairs to evaluate: $total_pairs\n")
flush(stdout)

# Run in parallel across hyperparameter pairs
results_grid = pmap(param_pairs; batch_size=1) do (α, κ)
    try
        seeds = rand(1:1_000_000, num_sims)

        sim_results = map(1:num_sims) do sim_idx
            seed = seeds[sim_idx]
            res = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
                bounds=bounds,
                σ_levels=σ_levels,
                n_init=n_init,
                n_iter=n_iter,
                κ=κ,
                α=α,
                seed=seed,
                fidelity_threshold=fidelity_threshold
            )

            Q_det_val = Q_true(res.x_rec)
            n_iters = res.n_iter_actual > 0 ? res.n_iter_actual : n_iter
            reached = res.n_iter_actual > 0

            (
                seed=seed,
                n_iterations=n_iters,
                reached=reached,
                Q_det=Q_det_val,
                Q_noisy=res.y_last
            )
        end

        iter_values = [r.n_iterations for r in sim_results]
        reach_rate = mean([r.reached for r in sim_results])

        (
            α=α,
            κ=κ,
            avg_iterations=mean(iter_values),
            std_iterations=std(iter_values),
            med_iterations=median(iter_values),
            min_iterations=minimum(iter_values),
            max_iterations=maximum(iter_values),
            reach_rate=reach_rate,
            sims=sim_results
        )
    catch e
        println("ERROR evaluating (α=$α, κ=$κ): $e")
        rethrow()
    end
end

println("\nCompleted evaluation of all hyperparameter pairs.")
flush(stdout)

# Sort by avg iterations, then std (lower is better)
sorted_results = sort(results_grid, by=r -> (r.avg_iterations, r.std_iterations))
best = first(sorted_results)

# Calculate Q_det statistics for best result
Q_det_values = [r.Q_det for r in best.sims]
avg_Q_det = mean(Q_det_values)
std_Q_det = std(Q_det_values)

println("\n" * "="^50)
println("=== BEST (α, κ) BY ITERATIONS & VARIANCE ===")
println("="^50)
println("Best α = ", best.α)
println("Best κ = ", best.κ)
println("Average iterations = ", best.avg_iterations, " ± ", best.std_iterations)
println("Median iterations = ", best.med_iterations)
println("Reach rate = ", best.reach_rate)
println("Average Q_det = ", avg_Q_det, " ± ", std_Q_det)
println("="^50)
flush(stdout)

# -------------------------
# Write Results to File
# -------------------------
output_file = joinpath(@__DIR__, "hyperparam_benchmark_results.txt")
open(output_file, "w") do io
    println(io, "=== HYPERPARAMETER BENCHMARK RESULTS ===")
    println(io, "Threshold = $fidelity_threshold")
    println(io, "n_init = $n_init, n_iter = $n_iter")
    println(io, "Sims per (α, κ) = $num_sims")
    println(io, "")
    println(io, "=== BEST (α, κ) BY ITERATIONS & VARIANCE ===")
    println(io, "Best α = $(best.α)")
    println(io, "Best κ = $(best.κ)")
    println(io, "Average iterations = $(best.avg_iterations) ± $(best.std_iterations)")
    println(io, "Median iterations = $(best.med_iterations)")
    println(io, "Reach rate = $(best.reach_rate)")
    println(io, "Average Q_det = $avg_Q_det ± $std_Q_det")
    println(io, "")
    println(io, "=== SUMMARY TABLE ===")
    println(io, "α\tκ\tavg_iters\tstd_iters\tmedian_iters\tmin_iters\tmax_iters\treach_rate\tavg_Q_det\tstd_Q_det")
    for r in sorted_results
        Q_dets = [s.Q_det for s in r.sims]
        println(io, "$(r.α)\t$(r.κ)\t$(r.avg_iterations)\t$(r.std_iterations)\t$(r.med_iterations)\t$(r.min_iterations)\t$(r.max_iterations)\t$(r.reach_rate)\t$(mean(Q_dets))\t$(std(Q_dets))")
    end
end

println("\nResults written to: $output_file")
flush(stdout)
