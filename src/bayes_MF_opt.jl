# src/bayes_MF_opt.jl
# Multi-fidelity BO with augmented input u = [x; z] where z encodes fidelity level.

using Random
using Statistics
using Distributions
using GaussianProcesses

struct MFResult
    Xa::Matrix{Float64}                 # (d+1)×n, last row is z
    y::Vector{Float64}                  # sign-adjusted if maximize=false
    bounds::Vector{Tuple{Float64,Float64}}
    z_levels::Vector{Float64}           # length L, recommended unique, in [0,1]
    costs::Vector{Float64}              # length L
    n_init::Int
    n_iter::Int
    n_iter_actual::Int                  # actual iterations run (may be < n_iter if threshold reached)
    maximize::Bool
    x_rec::Vector{Float64}
    ℓ_rec::Int                          # recommended fidelity index (usually argmax z)
    y_rec::Float64                      # posterior mean at (x_rec, z_levels[ℓ_rec]) in user sign
end

# -------- helpers --------

@inline function _validate_bounds(bounds)
    isempty(bounds) && throw(ArgumentError("bounds must be non-empty"))
    for (i, (lo, hi)) in enumerate(bounds)
        (isfinite(lo) && isfinite(hi)) || throw(ArgumentError("bounds[$i] must be finite"))
        lo < hi || throw(ArgumentError("bounds[$i] must satisfy lo < hi (got $lo, $hi)"))
    end
    return nothing
end

@inline function _rand_in_box(rng::Random.AbstractRNG, lb::Vector{Float64}, ub::Vector{Float64})
    d = length(lb)
    x = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        x[j] = rand(rng) * (ub[j] - lb[j]) + lb[j]
    end
    return x
end

function fit_gp_stable(X::Matrix{Float64}, y::Vector{Float64};
                       obs_noise::Float64=1e-3,
                       optimize_hypers::Bool=false)
    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ

    d, _ = size(X)
    ℓ0  = fill(0.3, d)
    σf0 = 1.0
    σn0 = max(obs_noise, 1e-8)

    gp = GP(X, ystd, MeanZero(), Matern(3/2, ℓ0, σf0), σn0)
    if optimize_hypers
        try
            optimize!(gp; domean=false, noise=false)
        catch err
            @warn "GP hyperparameter optimization failed; using initial hypers" err
        end
    end
    return gp, yμ, yσ
end

@inline function _gp_predict_f1(gp, u::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(u, :, 1))
    return μ[1], σ2[1]
end

@inline function _ei(μ::Float64, σ2::Float64, fbest_std::Float64; xi::Float64=0.0)
    Δ = μ - (fbest_std + xi)
    if σ2 ≤ 1e-18
        return max(Δ, 0.0)
    end
    σ = sqrt(σ2)
    γ = Δ / σ
    return Δ * cdf(Normal(), γ) + σ * pdf(Normal(), γ)
end

"""
    bayesopt_mf(f; bounds, z_levels, costs, ...)

Multi-fidelity BO using augmented input u = [x; z], with z chosen from `z_levels`.

Objective signature: `f(x, ℓ)` where ℓ is the index into `z_levels`/`costs`.

Selection heuristic: maximize EI(u)/costs[ℓ] over random candidates.

Recommendation: argmax posterior mean at the highest fidelity (argmax z).
"""
function bayesopt_mf(f;
                     bounds::Vector{Tuple{Float64,Float64}},
                     z_levels::Vector{Float64},
                     costs::Vector{Float64},
                     n_init::Int=10,
                     n_iter::Int=40,
                     M::Int=4000,
                     xi::Float64=0.01,
                     maximize::Bool=true,
                     rng::Random.AbstractRNG=Random.default_rng(),
                     seed=nothing,
                     obs_noise::Float64=1e-3,
                     optimize_hypers::Bool=false,
                     fidelity_threshold=nothing)

    _validate_bounds(bounds)
    length(z_levels) == length(costs) || throw(DimensionMismatch("z_levels and costs must have same length"))
    L = length(z_levels)
    L ≥ 2 || throw(ArgumentError("need at least 2 fidelities"))
    all(z -> isfinite(z) && 0.0 ≤ z ≤ 1.0, z_levels) || throw(ArgumentError("z_levels must be finite and in [0,1]"))
    all(c -> isfinite(c) && c > 0, costs) || throw(ArgumentError("costs must be finite and > 0"))
    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M ≥ 1      || throw(ArgumentError("M must be ≥ 1"))
    xi ≥ 0     || throw(ArgumentError("xi must be ≥ 0"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    d  = length(bounds)
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    f_eval = maximize ? f : ((x, ℓ) -> -f(x, ℓ))

    n_total = n_init + n_iter
    Xa = Matrix{Float64}(undef, d+1, n_total)
    y_int = Vector{Float64}(undef, n_total)
    n_iter_actual = n_iter

    # init: random x and random fidelity index
    for i in 1:n_init
        x = _rand_in_box(rng_local, lb, ub)
        ℓ = rand(rng_local, 1:L)
        z = z_levels[ℓ]
        Xa[:, i] = vcat(x, z)
        y_int[i] = f_eval(x, ℓ)
    end

    for i in (n_init + 1):n_total
        gp, yμ, yσ = fit_gp_stable(Xa[:, 1:(i-1)], y_int[1:(i-1)];
                                   obs_noise=obs_noise, optimize_hypers=optimize_hypers)
        y_std = (y_int[1:(i-1)] .- yμ) ./ yσ
        fbest_std = maximum(y_std)

        best_u = Xa[:, i-1]          # just a valid default
        best_score = -Inf
        best_ℓ = 1

        for _ in 1:M
            x = _rand_in_box(rng_local, lb, ub)
            ℓ = rand(rng_local, 1:L)
            u = vcat(x, z_levels[ℓ])

            μ, σ2 = _gp_predict_f1(gp, u)
            a = _ei(μ, σ2, fbest_std; xi=xi)
            score = a / costs[ℓ]

            if score > best_score
                best_score = score
                best_u = u
                best_ℓ = ℓ
            end
        end

        Xa[:, i] = best_u
        y_int[i] = f_eval(best_u[1:d], best_ℓ)
        
        # Check fidelity threshold for early stopping (latest measurement only)
        if fidelity_threshold !== nothing
            y_latest_raw = maximize ? y_int[i] : -y_int[i]
            reached = maximize ? (y_latest_raw >= fidelity_threshold) : (y_latest_raw <= fidelity_threshold)
            if reached
                n_iter_actual = i - n_init
                break
            end
        end
    end

    # Trim arrays if early stopping occurred
    if n_iter_actual < n_iter
        Xa = Xa[:, 1:(n_init + n_iter_actual)]
        y_int = y_int[1:(n_init + n_iter_actual)]
    end
    
    # recommendation: argmax posterior mean at highest z level
    gp, yμ, yσ = fit_gp_stable(Xa, y_int; obs_noise=obs_noise, optimize_hypers=optimize_hypers)
    ℓ_hi = argmax(z_levels)
    z_hi = z_levels[ℓ_hi]

    x_rec = _rand_in_box(rng_local, lb, ub)
    best_m = -Inf
    for _ in 1:M
        x = _rand_in_box(rng_local, lb, ub)
        u = vcat(x, z_hi)
        μstd, _ = _gp_predict_f1(gp, u)
        μ = yμ + yσ * μstd
        if μ > best_m
            best_m = μ
            x_rec = x
        end
    end

    y_user = maximize ? y_int : (-y_int)
    y_rec  = maximize ? best_m : -best_m

    return MFResult(Xa, y_user, bounds, z_levels, costs, n_init, n_iter, n_iter_actual, maximize, x_rec, ℓ_hi, y_rec)
end
