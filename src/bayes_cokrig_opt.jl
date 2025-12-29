# src/bayes_mfcokrig_opt.jl
# N-fidelity AR(1) co-kriging BO

using Random
using Statistics
using Distributions
using GaussianProcesses
using LinearAlgebra: dot

# ---------------- helpers ----------------

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

# Stable GP fit (guard optimization)
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

@inline function _gp_predict1(gp, x::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(x, :, 1))
    return μ[1], σ2[1]
end

@inline function _ei_from_moments(μ::Float64, σ2::Float64, fbest::Float64; xi::Float64=0.0)
    Δ = μ - (fbest + xi)
    if σ2 ≤ 1e-18
        return max(Δ, 0.0)
    end
    σ = sqrt(σ2)
    γ = Δ / σ
    return Δ * cdf(Normal(), γ) + σ * pdf(Normal(), γ)
end

# Estimate rho at level m using μ_{m-1}(x_m) (no need for exact matching)
function estimate_rho_from_prevmean(μprev::Vector{Float64}, ycur::Vector{Float64})
    denom = dot(μprev, μprev)
    return denom ≤ 1e-18 ? 1.0 : dot(μprev, ycur) / denom
end

# ---------------- model structs ----------------

struct MFCoKrigModel
    gps::Vector{Any}          # gp[1]=GP for level1; gpδ[m] stored in gps[m] for m>=2
    yμ::Vector{Float64}
    yσ::Vector{Float64}
    ρ::Vector{Float64}        # length N, with ρ[1]=1.0
end

"""
    predict_level(model, x, level) -> (μ, s2)

Predict mean/variance at requested fidelity level (AR(1) recursion).
"""
function predict_level(model::MFCoKrigModel, x::Vector{Float64}, level::Int)
    N = length(model.gps)
    (1 ≤ level ≤ N) || throw(ArgumentError("level must be in 1:$N"))

    # level 1
    μ1std, s21std = _gp_predict1(model.gps[1], x)
    μ = model.yμ[1] + model.yσ[1] * μ1std
    s2 = (model.yσ[1]^2) * max(s21std, 0.0)

    for m in 2:level
        # delta GP at level m is stored in gps[m]
        μδstd, s2δstd = _gp_predict1(model.gps[m], x)
        μδ = model.yμ[m] + model.yσ[m] * μδstd
        s2δ = (model.yσ[m]^2) * max(s2δstd, 0.0)

        ρm = model.ρ[m]
        μ = ρm * μ + μδ
        s2 = (ρm^2) * s2 + s2δ
    end

    return μ, s2
end

# ---------------- fit model ----------------

"""
    fit_mfcokrig(Xs, ys; obs_noises, optimize_hypers=false) -> MFCoKrigModel

- `Xs[m]` is d×n_m matrix of inputs for fidelity m (m=1..N)
- `ys[m]` is length n_m vector of observations (already sign-adjusted if needed)
- `obs_noises[m]` is scalar noise for GP at level m (used in GP fit)
"""
function fit_mfcokrig(Xs::Vector{Matrix{Float64}}, ys::Vector{Vector{Float64}};
                      obs_noises::Vector{Float64},
                      optimize_hypers::Bool=false)
    N = length(Xs)
    length(ys) == N || throw(DimensionMismatch("Xs and ys must have same length"))
    length(obs_noises) == N || throw(DimensionMismatch("obs_noises must have length N"))

    # fit level 1
    gp1, yμ1, yσ1 = fit_gp_stable(Xs[1], ys[1]; obs_noise=obs_noises[1], optimize_hypers=optimize_hypers)

    gps = Vector{Any}(undef, N)
    yμ  = Vector{Float64}(undef, N)
    yσ  = Vector{Float64}(undef, N)
    ρ   = ones(Float64, N)

    gps[1] = gp1; yμ[1] = yμ1; yσ[1] = yσ1; ρ[1] = 1.0

    # sequentially fit delta GPs
    for m in 2:N
        # compute μ_{m-1}(x) at the level-m inputs
        nm = size(Xs[m], 2)
        μprev = Vector{Float64}(undef, nm)
        for i in 1:nm
            x = vec(Xs[m][:, i])
            μp, _ = predict_level(MFCoKrigModel(gps[1:(m-1)], yμ[1:(m-1)], yσ[1:(m-1)], ρ[1:(m-1)]), x, m-1)
            μprev[i] = μp
        end

        ρm = estimate_rho_from_prevmean(μprev, ys[m])
        ρ[m] = ρm
        r = ys[m] .- ρm .* μprev

        gpδ, yμδ, yσδ = fit_gp_stable(Xs[m], r; obs_noise=obs_noises[m], optimize_hypers=optimize_hypers)
        gps[m] = gpδ; yμ[m] = yμδ; yσ[m] = yσδ
    end

    return MFCoKrigModel(gps, yμ, yσ, ρ)
end

# ---------------- BO result ----------------

struct MFCoKrigResult
    Xs::Vector{Matrix{Float64}}
    ys::Vector{Vector{Float64}}
    bounds::Vector{Tuple{Float64,Float64}}
    costs::Vector{Float64}
    n_init::Int
    n_iter::Int
    maximize::Bool
    ρ::Vector{Float64}
    x_rec::Vector{Float64}
    y_rec::Float64
end

"""
    mfcokrig_bayesopt(fs; bounds, costs, ...)

`fs[m](x)::Float64` evaluates fidelity m (m=1..N). Target is the highest fidelity N.

Selection heuristic (simple, consistent with your current approach):
- sample x candidates
- compute EI at the highest-fidelity posterior (level N)
- pick the fidelity level ℓ that maximizes EI_N(x) / costs[ℓ]
- if ℓ>1, also evaluate all lower fidelities at same x (nested design; improves stability)

Returns `MFCoKrigResult`.
"""
function mfcokrig_bayesopt(fs::Vector{Function};
                           bounds::Vector{Tuple{Float64,Float64}},
                           costs::Vector{Float64},
                           n_init::Int=12,
                           n_iter::Int=50,
                           M::Int=5000,
                           xi::Float64=0.01,
                           maximize::Bool=true,
                           rng::Random.AbstractRNG=Random.default_rng(),
                           seed=nothing,
                           obs_noises::Union{Nothing,Vector{Float64}}=nothing,
                           optimize_hypers::Bool=false,
                           p_levels::Union{Nothing,Vector{Float64}}=nothing)

    _validate_bounds(bounds)
    N = length(fs)
    N ≥ 2 || throw(ArgumentError("need at least 2 fidelities"))
    length(costs) == N || throw(DimensionMismatch("costs must have length N"))
    all(c -> isfinite(c) && c > 0, costs) || throw(ArgumentError("costs must be finite and > 0"))

    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M ≥ 1      || throw(ArgumentError("M must be ≥ 1"))
    xi ≥ 0     || throw(ArgumentError("xi must be ≥ 0"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    obs_noises === nothing && (obs_noises = fill(1e-3, N))
    length(obs_noises) == N || throw(DimensionMismatch("obs_noises must have length N"))

    # init distribution over levels (optional)
    if p_levels === nothing
        p_levels = fill(1.0 / N, N)
    end
    length(p_levels) == N || throw(DimensionMismatch("p_levels must have length N"))
    all(p -> p ≥ 0, p_levels) || throw(ArgumentError("p_levels must be ≥ 0"))
    s = sum(p_levels); s > 0 || throw(ArgumentError("p_levels must sum to > 0"))
    p_levels = p_levels ./ s

    d  = length(bounds)
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    # sign convention
    fe = maximize ? fs : [x -> -f(x) for f in fs]

    # data containers
    Xs = [Matrix{Float64}(undef, d, 0) for _ in 1:N]
    ys = [Float64[] for _ in 1:N]

    # init: sample x; choose a level; evaluate nested up to that level
    for _ in 1:n_init
        x = _rand_in_box(rng_local, lb, ub)
        ℓ = rand(rng_local, Distributions.Categorical(p_levels))
        for m in 1:ℓ
            Xs[m] = hcat(Xs[m], x)
            push!(ys[m], fe[m](x))
        end
    end

    ρ_last = ones(Float64, N)

    for _ in 1:n_iter
        model = fit_mfcokrig(Xs, ys; obs_noises=obs_noises, optimize_hypers=optimize_hypers)
        ρ_last = model.ρ

        # best observed at highest fidelity (fallback: best at highest available)
        fbest = isempty(ys[N]) ? maximum(ys[end-1]) : maximum(ys[N])

        best_x = _rand_in_box(rng_local, lb, ub)
        best_level = 1
        best_score = -Inf

        for _ in 1:M
            x = _rand_in_box(rng_local, lb, ub)
            μN, s2N = predict_level(model, x, N)
            a = _ei_from_moments(μN, s2N, fbest; xi=xi)

            # choose evaluation level by cost-normalized score
            for ℓ in 1:N
                score = a / costs[ℓ]
                if score > best_score
                    best_score = score
                    best_x = x
                    best_level = ℓ
                end
            end
        end

        # nested evaluation up to chosen level
        for m in 1:best_level
            Xs[m] = hcat(Xs[m], best_x)
            push!(ys[m], fe[m](best_x))
        end
    end

    # final recommendation: argmax posterior mean at highest fidelity
    model = fit_mfcokrig(Xs, ys; obs_noises=obs_noises, optimize_hypers=optimize_hypers)
    ρ_last = model.ρ

    best_x = _rand_in_box(rng_local, lb, ub)
    best_m = -Inf
    for _ in 1:M
        x = _rand_in_box(rng_local, lb, ub)
        μN, _ = predict_level(model, x, N)
        if μN > best_m
            best_m = μN
            best_x = x
        end
    end

    # map back to user sign
    ys_out = maximize ? ys : [(-v) for v in ys]
    y_rec  = maximize ? best_m : -best_m

    return MFCoKrigResult(Xs, ys_out, bounds, costs, n_init, n_iter, maximize, ρ_last, best_x, y_rec)
end
