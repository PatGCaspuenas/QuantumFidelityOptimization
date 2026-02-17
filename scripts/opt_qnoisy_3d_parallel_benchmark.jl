using Random
using Distributed
using Statistics
import Pkg

# Add worker processes (use all available cores)
if nprocs() == 1
    addprocs()
end

Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

@everywhere include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
@everywhere using .CalibrationCode

# Uncomment the line below to suppress warning messages (but also hides errors!)
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
# Multiple Simulations with Fixed Hyperparameters
# -------------------------
α = 2.0
κ = 1.9
num_sims = 40
fidelity_threshold = 0.998  # Set to a value (e.g., 0.95) to stop early when reached

println("=== Starting Parallel Simulations (Random Seeds) ===")
println("Fixed α = $α, κ = $κ")
println("Number of simulations: $num_sims")
if fidelity_threshold !== nothing
    println("Fidelity threshold: $fidelity_threshold")
end

random_seeds = rand(1:1000000, num_sims)

# Run multiple simulations in parallel with random seeds
results_grid = pmap(1:num_sims; batch_size=1) do sim_idx
    try
        seed = random_seeds[sim_idx]
        println("Starting simulation...")
        flush(stdout)
        res = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
            bounds=bounds,
            σ_levels=σ_levels,
            n_init=12,
            n_iter=120,
            κ=κ,
            α=α,
            seed=seed,
            fidelity_threshold=fidelity_threshold
        )
        
        Q_det_val = Q_true(res.x_rec)
        n_iters = res.n_iter_actual > 0 ? res.n_iter_actual : 120
        
        # Count noise level usage
        σy_used = res.σy[1:n_iters+12]  # n_init=12 + actual iterations
        noise_counts = Dict(σ => count(==(σ), σy_used) for σ in σ_levels)
        
        println("Simulation $sim_idx → Q_det = $(Q_det_val), noisy = $(res.y_last), iterations: $n_iters")
        flush(stdout)
        
        (
            sim_idx=sim_idx,
            seed=seed,
            x_rec=res.x_rec,
            y_rec=res.y_rec,
            Q_det=Q_det_val,
            Q_noisy=res.y_last,
            n_iterations=n_iters,
            noise_counts=noise_counts
        )
    catch e
        println("ERROR in simulation $sim_idx : ")
        println("Error type: $(typeof(e))")
        println("Error message: $e")
        Base.showerror(stdout, e, catch_backtrace())
        println()
        flush(stdout)
        rethrow()
    end
end

# -------------------------
# Find and Display Results
# -------------------------
Q_det_values = [r.Q_det for r in results_grid]

best_idx = argmax(Q_det_values)
best_result = results_grid[best_idx]

avg_Q_det = mean(Q_det_values)
med_Q_det = median(Q_det_values)
min_Q_det = minimum(Q_det_values)
max_Q_det = maximum(Q_det_values)
std_Q_det = std(Q_det_values)

iter_values = [r.n_iterations for r in results_grid]
avg_iterations = mean(iter_values)
med_iterations = median(iter_values)
std_iterations = std(iter_values)

# Aggregate noise level usage across all simulations
total_noise_counts = Dict(σ => 0 for σ in σ_levels)
for r in results_grid
    for (σ, count) in r.noise_counts
        total_noise_counts[σ] += count
    end
end
total_calls = sum(values(total_noise_counts))

println("\n=== RESULTS SUMMARY ===")
println("Best Q_det = ", best_result.Q_det, " (Simulation ", best_result.sim_idx, ")")
println("Average Q_det = ", avg_Q_det, " ± ", std_Q_det)
println("Median Q_det = ", med_Q_det)
println("Min Q_det = ", min_Q_det)
println("Max Q_det = ", max_Q_det)
println("Average iterations = ", avg_iterations, " ± ", std_iterations)
println("Median iterations = ", med_iterations)
println("\n=== NOISE LEVEL USAGE (AGGREGATE) ===")
for σ in sort(collect(σ_levels), rev=true)
    count = total_noise_counts[σ]
    pct = 100.0 * count / total_calls
    println("σ=$(σ):  $count calls ($(round(pct, digits=1))%)")
end
println("Total:  $total_calls calls")
println("\nBest noisy measurement = ", best_result.Q_noisy)
println("\nBest u_rec = ", best_result.x_rec)

fcl_rec, fsb_rec, A_rec = u_to_params(best_result.x_rec)
println("\n=== BEST RESULT PHYSICAL PARAMETERS ===")
println("Recommended f_cl = ", fcl_rec)
println("Recommended f_sb = ", fsb_rec)
println("Recommended A    = ", A_rec)
println("Baseline   f_cl = ", f_cl0, "  f_sb = ", f_sb0, "  A = ", A0)

# -------------------------
# Summary Table
# -------------------------
println("\n=== All Results ===")
σ_headers = join(["σ=$(σ)" for σ in sort(collect(σ_levels), rev=true)], "\t")
println("Sim\tSeed\tQ_det\tQ_noisy\tIterations\t$σ_headers")
for r in results_grid
    noise_str = join([string(r.noise_counts[σ]) for σ in sort(collect(σ_levels), rev=true)], "\t")
    println("$(r.sim_idx)\t$(r.seed)\t$(r.Q_det)\t$(r.Q_noisy)\t$(r.n_iterations)\t$noise_str")
end

# -------------------------
# Write Results to File
# -------------------------
output_file = joinpath(@__DIR__, "benchmark_results_fourlevels.txt")
open(output_file, "w") do io
    println(io, "=== BENCHMARK RESULTS ===")
    println(io, "Fixed α = $α, κ = $κ")
    println(io, "Number of simulations: $num_sims")
    println(io, "")
    println(io, "=== RESULTS SUMMARY ===")
    println(io, "Best Q_det = $(best_result.Q_det) (Simulation $(best_result.sim_idx), Seed $(best_result.seed), Iterations $(best_result.n_iterations))")
    println(io, "Average Q_det = $avg_Q_det ± $std_Q_det")
    println(io, "Median Q_det = $med_Q_det")
    println(io, "Min Q_det = $min_Q_det")
    println(io, "Max Q_det = $max_Q_det")
    println(io, "Average iterations = $avg_iterations ± $std_iterations")
    println(io, "Median iterations = $med_iterations")
    println(io, "")
    println(io, "=== NOISE LEVEL USAGE (AGGREGATE) ===")
    for σ in sort(collect(σ_levels), rev=true)
        count = total_noise_counts[σ]
        pct = 100.0 * count / total_calls
        println(io, "σ=$(σ):  $count calls ($(round(pct, digits=1))%)")
    end
    println(io, "Total:  $total_calls calls")
    println(io, "")
    println(io, "Best noisy measurement = $(best_result.Q_noisy)")
    println(io, "Best u_rec = $(best_result.x_rec)")
    println(io, "")
    println(io, "=== BEST RESULT PHYSICAL PARAMETERS ===")
    println(io, "Recommended f_cl = $fcl_rec")
    println(io, "Recommended f_sb = $fsb_rec")
    println(io, "Recommended A    = $A_rec")
    println(io, "Baseline   f_cl = $f_cl0, f_sb = $f_sb0, A = $A0")
    println(io, "")
    println(io, "=== INDIVIDUAL RESULTS ===")
    σ_headers = join(["σ=$(σ)" for σ in sort(collect(σ_levels), rev=true)], "\t")
    println(io, "Sim\tSeed\tIterations\tQ_det\t$σ_headers")
    for r in results_grid
        noise_str = join([string(r.noise_counts[σ]) for σ in sort(collect(σ_levels), rev=true)], "\t")
        println(io, "$(r.sim_idx)\t$(r.seed)\t$(r.n_iterations)\t$(r.Q_det)\t$noise_str")
    end
end

println("\nResults written to: $output_file")
