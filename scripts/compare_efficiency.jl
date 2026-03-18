import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Plots
using Statistics

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

# Synthetic noisy function that looks like Ackley but maximized
function f_true(x)
    return 20 * exp(-0.2 * sqrt(0.5 * (x[1]^2 + x[2]^2))) + exp(0.5 * (cos(2π * x[1]) + cos(2π * x[2]))) - 20 - exp(1)
end

bounds = [(-2.0, 2.0), (-2.0, 2.0)]
σ_min = 0.1
σ_levels = [0.1, 0.5, 1.0, 2.0]

n_trials = 10
n_init = 5
n_iter = 30

homo_costs = []
homo_bests = []
hetero_costs = []
hetero_bests = []

for trial in 1:n_trials
    println("Running Trial $trial...")

    # 1. Homoscedastic BO
    f_homo(x) = f_true(x) + σ_min * randn()
    res_homo = CalibrationCode.bayesopt(f_homo; bounds=bounds, n_init=n_init, n_iter=n_iter, obs_noise=nothing, seed=trial)

    c_homo = Float64[]
    b_homo = Float64[]
    current_cost = 0.0
    current_best = -Inf
    for i in 1:(n_init+n_iter)
        current_cost += 1.0 / (σ_min^2)  # cost is inversely proportional to variance
        best_so_far = maximum(f_true.(eachcol(res_homo.X[:, 1:i])))
        push!(c_homo, current_cost)
        push!(b_homo, best_so_far)
    end
    push!(homo_costs, c_homo)
    push!(homo_bests, b_homo)

    # 2. Heteroscedastic BO
    f_hetero(x, σ) = f_true(x) + σ * randn()
    res_hetero = CalibrationCode.bayesopt_ucb_threshold(f_hetero; bounds=bounds, σ_levels=σ_levels, n_init=n_init, n_iter=n_iter, seed=trial, verbose=false)

    c_hetero = Float64[]
    b_hetero = Float64[]
    current_cost = 0.0
    current_best = -Inf
    for i in 1:(n_init+n_iter)
        σ_i = res_hetero.σy[i]
        current_cost += 1.0 / (σ_i^2)
        best_so_far = maximum(f_true.(eachcol(res_hetero.X[:, 1:i])))
        push!(c_hetero, current_cost)
        push!(b_hetero, best_so_far)
    end
    push!(hetero_costs, c_hetero)
    push!(hetero_bests, b_hetero)
end

println("Plotting...")
p = plot(title="Efficiency: Heteroscedastic vs Homoscedastic BO",
    xlabel="Cumulative Evaluation Cost (1 / σ²)",
    ylabel="Best True Objective Found",
    legend=:bottomright,
    xlims=(0, 2000))  # limit x axis to zoom in on early finding phase

for i in 1:n_trials
    plot!(p, homo_costs[i], homo_bests[i], linecolor=:blue, alpha=0.4, lw=2, label=(i == 1 ? "Homoscedastic (Fixed High-Fidelity)" : false))
    plot!(p, hetero_costs[i], hetero_bests[i], linecolor=:red, alpha=0.4, lw=2, label=(i == 1 ? "Heteroscedastic (Adaptive Fidelity)" : false))
end

savefig(p, joinpath(@__DIR__, "efficiency_comparison.png"))
println("Saved scripts/efficiency_comparison.png")
