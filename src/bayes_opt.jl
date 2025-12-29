# src/bayes_opt.jl
# Bayesian optimization with a GP surrogate + Expected Improvement (EI)

using Random
using Statistics
using Distributions
using GaussianProcesses

"""
    BOResult

Stores all evaluated points and values, plus a recommendation.
- `X` is d×n (each column a point)
- `y` is length-n, in the *original* objective orientation (maximize or minimize as requested)
"""
struct BOResult
    X::Matrix{Float64}                       # d×n
    y::Vector{Float64}                       # length n (original scale/orientation)
    bounds::Vector{Tuple{Float64,Float64}}
    n_init::Int
    n_iter::Int
    maximize::Bool
    x_rec::Vector{Float64}
    y_rec::Float64
end

# ----------------- helpers -----------------

@inline function _rand_in_box(rng::Random.AbstractRNG,
                             lb::AbstractVector{<:Real},
                             ub::AbstractVector{<:Real})
    d = length(lb)
    x = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        x[j] = rand(rng) * (ub[j] - lb[j]) + lb[j]
    end
    return x
end

function _validate_bounds(bounds)
    isempty(bounds) && throw(ArgumentError("bounds must be non-empty"))
    for (i, (lo, hi)) in enumerate(bounds)
        (isfinite(lo) && isfinite(hi)) || throw(ArgumentError("bounds[$i] must be finite"))
        lo < hi || throw(ArgumentError("bounds[$i] must satisfy lo < hi (got $lo, $hi)"))
    end
    return nothing
end

function _fit_gp(X::Matrix{Float64}, y::Vector{Float64};
                 obs_noise::Union{Nothing,Float64}=nothing)
    # X: d×n, y: n
    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ

    d, _ = size(X)
    ℓ0  = fill(0.3, d)
    σf0 = 1.0
    σn0 = obs_noise === nothing ? 1e-5 : obs_noise

    gp = GP(X, ystd, MeanZero(), Matern(3/2, ℓ0, σf0), σn0)

    if obs_noise === nothing
        optimize!(gp)
    else
        optimize!(gp; noise=false)
    end

    return gp, yμ, yσ
end

@inline function _gp_predict_f1(gp, x::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(x, :, 1))
    return μ[1], σ2[1]
end

@inline function _ei(gp, x::Vector{Float64}, fbest_std::Float64; xi::Float64=0.0)
    μ, σ2 = _gp_predict_f1(gp, x)
    Δ = μ - (fbest_std + xi)
    if σ2 ≤ 1e-18
        return max(Δ, 0.0)
    end
    σ = sqrt(σ2)
    γ = Δ / σ
    return Δ * cdf(Normal(), γ) + σ * pdf(Normal(), γ)
end

# ----------------- public API -----------------

"""
    bayesopt(f; bounds, n_init=8, n_iter=30, M=2000, xi=0.01, maximize=true,
             rng=Random.default_rng(), seed=nothing, obs_noise=nothing)

Bayesian optimization using a Gaussian Process surrogate and Expected Improvement.

- `f(x)` must accept a `Vector{Float64}` of length d and return a real scalar.
- If `maximize=false`, the routine minimizes `f` (internally it maximizes `-f`).
- `M` controls the number of random candidate points used to (approximately) maximize EI and posterior mean.

Returns `(result::BOResult)`.
"""
function bayesopt(f;
                  bounds::Vector{Tuple{Float64,Float64}},
                  n_init::Int=8,
                  n_iter::Int=30,
                  M::Int=2000,
                  xi::Float64=0.01,
                  maximize::Bool=true,
                  rng::Random.AbstractRNG=Random.default_rng(),
                  seed=nothing,
                  obs_noise::Union{Nothing,Float64}=nothing)

    _validate_bounds(bounds)
    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M ≥ 1      || throw(ArgumentError("M must be ≥ 1"))
    xi ≥ 0     || throw(ArgumentError("xi must be ≥ 0"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    d = length(bounds)
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    # internal objective is always maximization
    f_eval = maximize ? f : (x -> -f(x))

    n_total = n_init + n_iter
    X = Matrix{Float64}(undef, d, n_total)
    y_int = Vector{Float64}(undef, n_total)   # internal (max) values

    # initial design
    for i in 1:n_init
        x = _rand_in_box(rng_local, lb, ub)
        X[:, i] = x
        y_int[i] = f_eval(x)
    end

    # BO loop
    for i in (n_init + 1):n_total
        gp, yμ, yσ = _fit_gp(X[:, 1:(i-1)], y_int[1:(i-1)]; obs_noise=obs_noise)
        y_std = (y_int[1:(i-1)] .- yμ) ./ yσ
        fbest_std = maximum(y_std)

        best_x = _rand_in_box(rng_local, lb, ub)
        best_a = -Inf
        for _ in 1:M
            x = _rand_in_box(rng_local, lb, ub)
            a = _ei(gp, x, fbest_std; xi=xi)
            if a > best_a
                best_a = a
                best_x = x
            end
        end

        X[:, i] = best_x
        y_int[i] = f_eval(best_x)
    end

    # recommendation: argmax posterior mean (approx over M random candidates)
    gp, yμ, yσ = _fit_gp(X, y_int; obs_noise=obs_noise)
    x_rec = _rand_in_box(rng_local, lb, ub)
    best_m = -Inf
    for _ in 1:M
        x = _rand_in_box(rng_local, lb, ub)
        μstd, _ = _gp_predict_f1(gp, x)
        μ = yμ + yσ * μstd
        if μ > best_m
            best_m = μ
            x_rec = x
        end
    end

    # map back to user orientation
    y_user = maximize ? y_int : (-y_int)
    y_rec  = maximize ? best_m : -best_m

    return BOResult(X, y_user, bounds, n_init, n_iter, maximize, x_rec, y_rec)
end
