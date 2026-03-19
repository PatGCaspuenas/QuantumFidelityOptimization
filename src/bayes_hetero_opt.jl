# src/bayes_hetero_opt.jl
# Heteroscedastic GP + GP-UCB acquisition with noise-level thresholding

using Random
using Statistics
using LinearAlgebra
using Optim

# -------------------------
# Kernel and covariance
# -------------------------

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
    # X: d×n -> K: n×n
    _, n = size(X)
    K = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        xi = view(X, :, i)
        K[i, i] = matern32(xi, xi, ℓ, σf)
        for j in (i+1):n
            kij = matern32(xi, view(X, :, j), ℓ, σf)
            K[i, j] = kij
            K[j, i] = kij
        end
    end
    return K
end

# -------------------------
# Heteroscedastic GP model
# -------------------------

struct HeteroGP
    X::Matrix{Float64}     # d×n
    yμ::Float64
    yσ::Float64
    ℓ::Vector{Float64}
    σf::Float64
    c::Float64
    L::LowerTriangular{Float64,Matrix{Float64}}  # chol(K + Σ)
    α::Vector{Float64}     # (K+Σ)^{-1} ystd
    θ::Vector{Float64}     # log-parameters
end

function _validate_gp_inputs(X, y, σy)
    size(X, 2) == length(y) || throw(DimensionMismatch("size(X,2) must equal length(y)"))
    length(y) == length(σy) || throw(DimensionMismatch("length(y) must equal length(σy)"))
    all(isfinite, y) || throw(ArgumentError("y must be finite"))
    all(s -> isfinite(s) && s ≥ 0, σy) || throw(ArgumentError("σy must be finite and ≥ 0"))
    return nothing
end

"""
    fit_heterogp(X, y, σy; ...)

Fits a heteroscedastic GP with known per-point observation noise σyᵢ and
Matérn 3/2 ARD kernel. Assumes inputs are scaled to [-1,1]^d.

Optimizes θ = (logℓ, logσf, logc) by bounded LML minimization with multi-start.
Set `learn_hypers=false` to reuse provided `θ_init` without optimization.
"""
function fit_heterogp(X::Matrix{Float64}, y::Vector{Float64}, σy::Vector{Float64};
                      jitter::Float64=1e-8,
                      learn_hypers::Bool=true,
                      learn_noise_scale::Bool=true,
                      n_restarts::Int=6,
                      θ_init::Union{Nothing,Vector{Float64}}=nothing,
                      fixed_ℓ::Union{Nothing,Vector{Float64}}=nothing,
                      ℓ_bounds::Tuple{Float64,Float64}=(0.05, 1.5),
                      σf_bounds::Tuple{Float64,Float64}=(0.3, 2.0),
                      c_bounds::Tuple{Float64,Float64}=(0.3, 3.0),
                      rng::Random.AbstractRNG=Random.default_rng())

    _validate_gp_inputs(X, y, σy)
    jitter > 0 || throw(ArgumentError("jitter must be > 0"))

    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ
    σstd0 = (σy ./ yσ)

    d, n = size(X)
    use_fixed_ℓ = fixed_ℓ !== nothing
    p_opt = use_fixed_ℓ ? (learn_noise_scale ? 2 : 1) : (learn_noise_scale ? d + 2 : d + 1)

    lower = Vector{Float64}(undef, p_opt)
    upper = Vector{Float64}(undef, p_opt)
    if use_fixed_ℓ
        lower[1] = log(σf_bounds[1]); upper[1] = log(σf_bounds[2])
        if learn_noise_scale
            lower[2] = log(c_bounds[1]); upper[2] = log(c_bounds[2])
        end
    else
        lower[1:d] .= log(ℓ_bounds[1]);  upper[1:d] .= log(ℓ_bounds[2])
        lower[d+1]  = log(σf_bounds[1]); upper[d+1]  = log(σf_bounds[2])
        if learn_noise_scale
            lower[d+2] = log(c_bounds[1]); upper[d+2] = log(c_bounds[2])
        end
    end

    function chol_from_θ(θ::Vector{Float64})
        ℓ  = use_fixed_ℓ ? fixed_ℓ : exp.(θ[1:d])
        σf = use_fixed_ℓ ? exp(θ[1]) : exp(θ[d+1])
        c  = if learn_noise_scale
            use_fixed_ℓ ? exp(θ[2]) : exp(θ[d+2])
        else
            1.0
        end

        K = buildK(X, ℓ, σf)
        σstd = c .* σstd0
        @inbounds for i in 1:n
            si = max(σstd[i], 1e-10)
            K[i, i] += si^2 + jitter
        end

        try
            return cholesky(Symmetric(K))
        catch
            return nothing
        end
    end

    function nlml(θ::Vector{Float64})
        F = chol_from_θ(θ)
        F === nothing && return Inf
        # solve (K+Σ)^{-1} y
        αtmp = F \ ystd
        return 0.5 * dot(ystd, αtmp) + sum(log, diag(F.L)) + 0.5 * n * log(2π)
    end

    # starting point
    _θ_ε = 1e-4  # small inset from boundary to avoid Fminbox boundary rejection
    θ0 = Vector{Float64}(undef, p_opt)
    if use_fixed_ℓ
        # Warm-start σf (and c) from a full-length θ_init if available
        if θ_init !== nothing && length(θ_init) >= d + 1
            θ0[1] = clamp(θ_init[d+1], lower[1] + _θ_ε, upper[1] - _θ_ε)
            if learn_noise_scale && length(θ_init) >= d + 2
                θ0[2] = clamp(θ_init[d+2], lower[2] + _θ_ε, upper[2] - _θ_ε)
            elseif learn_noise_scale
                θ0[2] = clamp(log(1.0), lower[2] + _θ_ε, upper[2] - _θ_ε)
            end
        elseif θ_init !== nothing && length(θ_init) == p_opt
            θ0 .= clamp.(θ_init, lower .+ _θ_ε, upper .- _θ_ε)
        else
            θ0[1] = clamp(log(1.0), lower[1] + _θ_ε, upper[1] - _θ_ε)
            if learn_noise_scale; θ0[2] = clamp(log(1.0), lower[2] + _θ_ε, upper[2] - _θ_ε); end
        end
    else
        if θ_init !== nothing && length(θ_init) == p_opt
            θ0 .= clamp.(θ_init, lower .+ _θ_ε, upper .- _θ_ε)
        else
            θ0[1:d] .= log(0.3)
            θ0[d+1]  = log(1.0)
            if learn_noise_scale
                θ0[d+2] = log(1.0)
            end
            θ0 .= clamp.(θ0, lower .+ _θ_ε, upper .- _θ_ε)
        end
    end

    if !learn_hypers
        F = chol_from_θ(θ0)
        F === nothing && throw(ArgumentError("Cholesky failed with fixed hyperparameters; increase jitter or adjust bounds."))
        ℓ  = use_fixed_ℓ ? fixed_ℓ : exp.(θ0[1:d])
        σf = use_fixed_ℓ ? exp(θ0[1]) : exp(θ0[d+1])
        c  = learn_noise_scale ? (use_fixed_ℓ ? exp(θ0[2]) : exp(θ0[d+2])) : 1.0
        L = F.L
        α = L' \ (L \ ystd)
        full_θ = use_fixed_ℓ ? vcat(log.(ℓ), θ0) : copy(θ0)
        return HeteroGP(X, yμ, yσ, ℓ, σf, c, L, α, full_θ)
    end

    bestθ = copy(θ0)
    bestv = nlml(bestθ)

    opts = Optim.Options(iterations=250, g_tol=1e-6, f_abstol=1e-9)

    for r in 1:n_restarts
        θstart = copy(θ0)
        if r > 1
            @inbounds for i in 1:p_opt
                θstart[i] = lower[i] + rand(rng) * (upper[i] - lower[i])
            end
        end

        res = optimize(nlml, lower, upper, θstart,
                       Fminbox(LBFGS()),
                       opts;
                       autodiff = :finite)

        θhat = Optim.minimizer(res)
        vhat = Optim.minimum(res)

        if isfinite(vhat) && vhat < bestv
            bestv = vhat
            bestθ .= θhat
        end
    end

    F = chol_from_θ(bestθ)
    F === nothing && throw(ArgumentError("Cholesky failed at optimized θ; increase jitter or tighten bounds."))
    ℓ  = use_fixed_ℓ ? fixed_ℓ : exp.(bestθ[1:d])
    σf = use_fixed_ℓ ? exp(bestθ[1]) : exp(bestθ[d+1])
    c  = learn_noise_scale ? (use_fixed_ℓ ? exp(bestθ[2]) : exp(bestθ[d+2])) : 1.0
    L = F.L
    α = L' \ (L \ ystd)
    full_θ = use_fixed_ℓ ? vcat(log.(ℓ), bestθ) : copy(bestθ)

    return HeteroGP(X, yμ, yσ, ℓ, σf, c, L, α, full_θ)
end

"""
    predict_latent(gp, x) -> (μ, s2)

Posterior mean and variance of the latent function f(x) (not including observation noise).
"""
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

# -------------------------
# BO structures + utilities
# -------------------------

struct HeteroBOResult
    X::Matrix{Float64}                    # d×n
    y::Vector{Float64}                    # sign-adjusted if maximize=false
    σy::Vector{Float64}                   # noise std per observation
    bounds::Vector{Tuple{Float64,Float64}}
    σ_levels::Vector{Float64}
    n_init::Int
    n_iter::Int
    maximize::Bool
    x_rec::Vector{Float64}
    y_rec::Float64
    n_iter_actual::Int                    # actual iterations run (< n_iter if early stopping)
    y_last::Float64                       # last noisy observation (sign-adjusted)
end

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

# μ is the fidelity of the GP at the given point. s2 is the variance. 
# The larger kappa is, the more exploration the acquisition function does.
@inline ucb_score(μ::Float64, s2::Float64, κ::Float64) = μ + κ * sqrt(max(s2, 0.0))

function choose_sigma_threshold(s2::Float64, σ_levels::Vector{Float64}; α::Float64=0.5)
    isempty(σ_levels) && throw(ArgumentError("σ_levels must be non-empty"))
    all(s -> isfinite(s) && s ≥ 0, σ_levels) || throw(ArgumentError("σ_levels must be finite and ≥ 0"))
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

function recommend_mean(gp::HeteroGP, bounds; M::Int=20000, rng::Random.AbstractRNG=Random.default_rng())
    M ≥ 1 || throw(ArgumentError("M must be ≥ 1"))
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]
    best_x = _rand_in_box(rng, lb, ub)
    best_m = -Inf
    for _ in 1:M
        x = _rand_in_box(rng, lb, ub)
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

# -------------------------
# Main algorithm
# -------------------------

"""
    bayesopt_ucb_threshold(f; bounds, σ_levels, ...)

Heteroscedastic BO with GP-UCB acquisition for x and thresholding rule for σ selection.
Objective is `f(x, σ)`.

Set `maximize=false` to minimize.
Use `seed` for determinism without affecting the global RNG.
"""
function bayesopt_ucb_threshold(f;
                               bounds::Vector{Tuple{Float64,Float64}},
                               σ_levels::Vector{Float64},
                               n_init::Int=8,
                               n_iter::Int=30,
                               M_acq::Int=5000,
                               M_rec::Int=20000,
                               κ::Float64=2.0,
                               α::Float64=0.5,
                               maximize::Bool=true,
                               hyper_every::Int=10,
                               rng::Random.AbstractRNG=Random.default_rng(),
                               seed=nothing,
                               verbose::Bool=false,
                               pretrained_θ::Union{Nothing,Vector{Float64}}=nothing,
                               freeze_mode::Symbol=:none,
                               n_freeze_iters::Int=typemax(Int),
                               fidelity_threshold::Union{Nothing,Float64}=nothing,
                               explore_frac::Float64=0.0)

    _validate_bounds(bounds)
    isempty(σ_levels) && throw(ArgumentError("σ_levels must be non-empty"))
    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M_acq ≥ 1  || throw(ArgumentError("M_acq must be ≥ 1"))
    M_rec ≥ 1  || throw(ArgumentError("M_rec must be ≥ 1"))
    κ ≥ 0      || throw(ArgumentError("κ must be ≥ 0"))
    α ≥ 0      || throw(ArgumentError("α must be ≥ 0"))
    freeze_mode ∈ (:none, :all, :lengthscales) || throw(ArgumentError("freeze_mode must be :none, :all, or :lengthscales"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    f_eval = maximize ? f : ((x, σ) -> -f(x, σ))

    d = length(bounds)
    n_total = n_init + n_iter
    X  = Matrix{Float64}(undef, d, n_total)
    y  = Vector{Float64}(undef, n_total)
    σy = Vector{Float64}(undef, n_total)

    for i in 1:n_init
        x = _rand_in_box(rng_local, lb, ub)
        σ0 = rand(rng_local, σ_levels)
        X[:, i] = x
        σy[i] = σ0
        y[i] = f_eval(x, σ0)
    end

    # Prepare freeze mode state
    fixed_ℓ_bo = if freeze_mode == :lengthscales && pretrained_θ !== nothing
        exp.(pretrained_θ[1:d])
    else
        nothing
    end

    θ_prev = pretrained_θ
    n_iter_actual = n_iter
    y_last_val = 0.0

    for it in 1:n_iter
        idx = n_init + it

        if freeze_mode == :all
            gp = fit_heterogp(X[:, 1:(idx-1)], y[1:(idx-1)], σy[1:(idx-1)];
                              θ_init=pretrained_θ,
                              learn_hypers=false,
                              learn_noise_scale=true,
                              jitter=1e-8,
                              rng=rng_local)
        else
            # For :lengthscales, hold ℓ fixed until n_freeze_iters then release to full MLE.
            # Force a refit at the transition iteration to relearn ℓ immediately.
            current_fixed_ℓ = (freeze_mode == :lengthscales && it <= n_freeze_iters) ? fixed_ℓ_bo : nothing
            releasing = (freeze_mode == :lengthscales && it == n_freeze_iters + 1)
            do_opt = (it == 1) || releasing || (hyper_every > 0 && it % hyper_every == 0)
            gp = fit_heterogp(X[:, 1:(idx-1)], y[1:(idx-1)], σy[1:(idx-1)];
                              θ_init=θ_prev,
                              fixed_ℓ=current_fixed_ℓ,
                              learn_hypers=do_opt,
                              learn_noise_scale=true,
                              n_restarts=do_opt ? 6 : 0,
                              jitter=1e-8,
                              rng=rng_local)
        end

        θ_prev = gp.θ

        best_x = _rand_in_box(rng_local, lb, ub)
        best_a = -Inf
        best_s2 = 0.0

        for _ in 1:M_acq
            x = _rand_in_box(rng_local, lb, ub)
            μ, s2 = predict_latent(gp, x)
            a = ucb_score(μ, s2, κ)
            if a > best_a
                best_a = a
                best_x = x
                best_s2 = s2
            end
        end

        σ_next = choose_sigma_threshold(best_s2, σ_levels; α=α)

        if explore_frac > 0.0 && rand(rng_local) < explore_frac
            σ_next = maximum(σ_levels)
        end

        X[:, idx] = best_x
        σy[idx] = σ_next
        y[idx] = f_eval(best_x, σ_next)
        y_last_val = y[idx]

        if verbose
            @info "it=$it best_acq=$best_a σ=$σ_next"
        end

        # Early stopping
        if fidelity_threshold !== nothing && y[idx] >= (maximize ? fidelity_threshold : -fidelity_threshold)
            n_iter_actual = it
            break
        end
    end

    n_data = n_init + n_iter_actual
    still_frozen = freeze_mode == :lengthscales && n_iter_actual <= n_freeze_iters
    final_fixed_ℓ = still_frozen ? fixed_ℓ_bo : nothing
    gp = fit_heterogp(X[:, 1:n_data], y[1:n_data], σy[1:n_data];
                      θ_init=freeze_mode == :all ? pretrained_θ : θ_prev,
                      fixed_ℓ=final_fixed_ℓ,
                      learn_hypers=freeze_mode != :all,
                      learn_noise_scale=true,
                      n_restarts=freeze_mode == :all ? 0 : 8,
                      jitter=1e-8,
                      rng=rng_local)

    x_rec, m_rec = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)

    y_out = maximize ? y : (-y)
    y_rec = maximize ? m_rec : -m_rec
    y_last_out = maximize ? y_last_val : -y_last_val

    return HeteroBOResult(X, y_out, σy, bounds, σ_levels, n_init, n_iter, maximize, x_rec, y_rec, n_iter_actual, y_last_out)
end
