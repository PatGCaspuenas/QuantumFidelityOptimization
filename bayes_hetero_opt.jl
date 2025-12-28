module BayesHeteroOpt

using Random, Statistics, LinearAlgebra

export HeteroBOResult, bayesopt_ucb_threshold, predict_latent, count_noise_levels

# ============================================================
# Heteroscedastic GP regression with known per-point noise σy_i
#   y_i = f(x_i) + ε_i,   ε_i ~ Normal(0, σy_i^2)
# Uses Matérn 3/2 kernel with ARD lengthscales.
# ============================================================

@inline function matern32(x::AbstractVector, z::AbstractVector,
                          ℓ::AbstractVector, σf::Float64)
    r2 = 0.0
    @inbounds for j in eachindex(ℓ)
        u = (x[j] - z[j]) / ℓ[j]
        r2 += u*u
    end
    r = sqrt(r2)
    a = sqrt(3.0) * r
    return (σf^2) * (1 + a) * exp(-a)
end

function buildK(X::Matrix{Float64}, ℓ::Vector{Float64}, σf::Float64)
    # X: d×n  -> K: n×n
    _, n = size(X)
    K = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        xi = view(X, :, i)
        K[i,i] = matern32(xi, xi, ℓ, σf)
        for j in (i+1):n
            kij = matern32(xi, view(X, :, j), ℓ, σf)
            K[i,j] = kij
            K[j,i] = kij
        end
    end
    return K
end

struct HeteroGP
    X::Matrix{Float64}     # d×n
    yμ::Float64
    yσ::Float64
    ℓ::Vector{Float64}
    σf::Float64
    L::LowerTriangular{Float64,Matrix{Float64}}  # chol(K + Σ)
    α::Vector{Float64}     # (K+Σ)^{-1} ystd
end

function fit_gp(X::Matrix{Float64}, y::Vector{Float64}, σy::Vector{Float64};
                ℓ::Union{Nothing,Vector{Float64}}=nothing,
                σf::Float64=1.0,
                jitter::Float64=1e-10)

    @assert size(X,2) == length(y) == length(σy)

    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ
    σstd = σy ./ yσ

    d, n = size(X)
    ℓ = ℓ === nothing ? fill(0.3, d) : ℓ

    K = buildK(X, ℓ, σf)
    @inbounds for i in 1:n
        K[i,i] += σstd[i]^2 + jitter
    end

    L = cholesky(Symmetric(K)).L
    α = L' \ (L \ ystd)

    return HeteroGP(X, yμ, yσ, ℓ, σf, L, α)
end

function predict_latent(gp::HeteroGP, x::Vector{Float64})
    X = gp.X
    _, n = size(X)

    k = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k[i] = matern32(x, view(X, :, i), gp.ℓ, gp.σf)
    end

    μstd = dot(k, gp.α)
    v = gp.L \ k
    kxx = matern32(x, x, gp.ℓ, gp.σf)
    s2std = max(kxx - dot(v, v), 0.0)

    μ  = gp.yμ + gp.yσ * μstd
    s2 = (gp.yσ^2) * s2std
    return μ, s2
end

# ============================================================
# BO result
# ============================================================

struct HeteroBOResult
    X::Matrix{Float64}                    # d×n
    y::Vector{Float64}                    # observed y (sign-adjusted if maximize=false)
    σy::Vector{Float64}                   # noise std used per observation (true units)
    bounds::Vector{Tuple{Float64,Float64}}
    σ_levels::Vector{Float64}
    n_init::Int
    maximize::Bool
    x_rec::Vector{Float64}                # recommendation: argmax posterior mean
    y_rec::Float64                        # posterior mean at x_rec (sign-adjusted)
end

# ============================================================
# Utilities
# ============================================================

@inline function rand_in_box(lb::Vector{Float64}, ub::Vector{Float64})
    d = length(lb)
    x = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        x[j] = rand() * (ub[j] - lb[j]) + lb[j]
    end
    return x
end

@inline ucb_score(μ::Float64, s2::Float64, κ::Float64) = μ + κ * sqrt(max(s2, 0.0))

function choose_sigma_threshold(s2::Float64, σ_levels::Vector{Float64}; α::Float64=0.5)
    # choose the largest σ such that σ <= α * s(x); if none exists, choose smallest σ
    thresh = α * sqrt(max(s2, 0.0))
    best = nothing
    for σ in σ_levels
        if σ <= thresh
            if best === nothing || σ > best
                best = σ
            end
        end
    end
    return best === nothing ? minimum(σ_levels) : best
end

function recommend_mean(gp::HeteroGP, bounds; M::Int=20000)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]
    best_x = nothing
    best_m = -Inf
    for _ in 1:M
        x = rand_in_box(lb, ub)
        μ, _ = predict_latent(gp, x)
        if μ > best_m
            best_m = μ
            best_x = x
        end
    end
    return best_x, best_m
end

function count_noise_levels(res::HeteroBOResult; atol::Float64=1e-12)
    counts = Dict(σ => 0 for σ in res.σ_levels)
    for s in res.σy
        j = findfirst(σ -> isapprox(s, σ; atol=atol, rtol=0), res.σ_levels)
        if j === nothing
            counts[s] = get(counts, s, 0) + 1
        else
            counts[res.σ_levels[j]] += 1
        end
    end
    return counts
end

# ============================================================
# Main algorithm:
#   x_{t+1} = argmax_x  μ(x) + κ s(x)    (GP-UCB)
#   σ_{t+1} = max {σ in levels : σ <= α s(x_{t+1}) } else min(levels)
# ============================================================

"""
    bayesopt_ucb_threshold(f; bounds, σ_levels, n_init, n_iter, M_acq, M_rec, κ, α, seed, maximize, ℓ, σf)

`f(x, σ)` must return an observation y using noise level σ.
- Uses GP-UCB to select x.
- Uses threshold rule σ <= α * s(x) to select noise level.
- Returns a `HeteroBOResult` whose recommendation is argmax posterior mean.
"""
function bayesopt_ucb_threshold(f; bounds::Vector{Tuple{Float64,Float64}},
                                σ_levels::Vector{Float64},
                                n_init::Int=8,
                                n_iter::Int=30,
                                M_acq::Int=5000,
                                M_rec::Int=20000,
                                κ::Float64=2.0,
                                α::Float64=0.5,
                                seed=nothing,
                                maximize::Bool=true,
                                ℓ::Union{Nothing,Vector{Float64}}=nothing,
                                σf::Float64=1.0)

    seed !== nothing && Random.seed!(seed)

    d  = length(bounds)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    # internal sign convention: maximize
    f_eval = maximize ? f : ((x,σ) -> -f(x,σ))

    # Data
    X  = Matrix{Float64}(undef, d, 0)
    y  = Float64[]
    σy = Float64[]

    # Init: random x, choose largest σ (cheapest) by default
    σ_init = maximum(σ_levels)
    for _ in 1:n_init
        x = rand_in_box(lb, ub)
        yi = f_eval(x, σ_init)
        X = hcat(X, x)
        push!(y, yi)
        push!(σy, σ_init)
    end

    # BO loop
    for _ in 1:n_iter
        gp = fit_gp(X, y, σy; ℓ=ℓ, σf=σf)

        # 1) choose x by GP-UCB
        best_x = nothing
        best_a = -Inf
        best_s2 = 0.0
        for _ in 1:M_acq
            x = rand_in_box(lb, ub)
            μ, s2 = predict_latent(gp, x)
            a = ucb_score(μ, s2, κ)
            if a > best_a
                best_a = a
                best_x = x
                best_s2 = s2
            end
        end

        # 2) choose σ by threshold rule
        σ_next = choose_sigma_threshold(best_s2, σ_levels; α=α)

        # 3) evaluate
        yi = f_eval(best_x, σ_next)

        X = hcat(X, best_x)
        push!(y, yi)
        push!(σy, σ_next)
    end

    # Recommendation: argmax posterior mean
    gp = fit_gp(X, y, σy; ℓ=ℓ, σf=σf)
    x_rec, m_rec = recommend_mean(gp, bounds; M=M_rec)

    # Undo sign for outputs if minimize
    y_out = maximize ? y : (-y)
    y_rec = maximize ? m_rec : -m_rec

    return HeteroBOResult(X, y_out, σy, bounds, σ_levels, n_init, maximize, x_rec, y_rec)
end

end # module
