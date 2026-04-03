# src/bayes_hetero_opt.jl
# Heteroscedastic GP + GP-UCB acquisition with noise-level thresholding

using Random
using Statistics
using LinearAlgebra
using Optim
using QuasiMonteCarlo

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
                      c_bounds::Tuple{Float64,Float64}=(0.05, 3.0),
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
    X::Matrix{Float64}                    # d×n  (training points only)
    y::Vector{Float64}                    # sign-adjusted if maximize=false
    σy::Vector{Float64}                   # noise std per observation
    bounds::Vector{Tuple{Float64,Float64}}
    n_shots::Int                          # shots per f call
    n_init::Int
    n_iter::Int
    maximize::Bool
    x_rec::Vector{Float64}
    y_rec::Float64
    n_iter_actual::Int                    # actual iterations run (< n_iter if early stopping)
    y_last::Float64                       # last noisy observation (sign-adjusted)
    ℓ_final::Vector{Float64}             # final GP lengthscales (original scale)
    σf_final::Float64                    # final GP signal std (original scale)
    c_final::Float64                     # final GP noise scale (original scale)
    total_shots::Int                      # total shots used across ALL f calls (incl. check calls not in X)
end

@inline function _validate_bounds(bounds)
    isempty(bounds) && throw(ArgumentError("bounds must be non-empty"))
    for (i, (lo, hi)) in enumerate(bounds)
        (isfinite(lo) && isfinite(hi)) || throw(ArgumentError("bounds[$i] must be finite"))
        lo < hi || throw(ArgumentError("bounds[$i] must satisfy lo < hi (got $lo, $hi)"))
    end
    return nothing
end

# Returns true if x is at least min_dist away (Euclidean) from all n filled columns of X.
# Used to prevent near-duplicate training points that ill-condition the kernel matrix K.
@inline function _is_far_enough(x::AbstractVector{Float64}, X::Matrix{Float64}, n::Int;
                                 min_dist::Float64=1e-4)
    @inbounds for i in 1:n
        d2 = 0.0
        for j in eachindex(x)
            u = x[j] - X[j, i]
            d2 += u * u
        end
        d2 < min_dist * min_dist && return false
    end
    return true
end

# f(x, N) must return a tuple (y::Float64, σy::Float64).
# y  is the (possibly log-transformed) fidelity observation.
# σy is the GP observation noise, computed by the caller according to the noise model:
#   :simple   → σy = 1/√N
#   :binomial → σy = √(Q(1-Q)/N), propagated through log if needed
@inline function _call_f_raw(f, x::Vector{Float64}, n::Int)
    result = f(x, n)
    result isa Tuple || throw(ArgumentError("f must return a (y, σy) tuple when called with N::Int"))
    return Float64(result[1]), Float64(result[2])
end

@inline function _rand_in_box(rng::Random.AbstractRNG, lb::Vector{Float64}, ub::Vector{Float64})
    d = length(lb)
    x = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        x[j] = rand(rng) * (ub[j] - lb[j]) + lb[j]
    end
    return x
end

"""
    _sobol_in_box(rng, lb, ub, n) -> Matrix{Float64} (d×n)

Generate `n` points in `[lb, ub]^d` using Sobol sequences with Owen scrambling.
A fresh scramble seed is drawn from `rng` on each call, so different simulations
(different `rng` seeds) and different iterations (rng has advanced) all get
distinct low-discrepancy sequences.

Owen scrambling requires n to be a power of 2: we round up and trim to the first n columns.
"""
function _sobol_in_box(rng::Random.AbstractRNG, lb::Vector{Float64}, ub::Vector{Float64}, n::Int)
    n_pow2 = max(2, nextpow(2, n))  # Owen scrambling requires power-of-2 count
    local_rng = MersenneTwister(rand(rng, UInt32))  # new seed each call → unique sequence
    pts = QuasiMonteCarlo.sample(n_pow2, lb, ub,
                                 SobolSample(R=OwenScramble(base=2, rng=local_rng)))
    return pts[:, 1:n]  # d×n
end

# μ is the fidelity of the GP at the given point. s2 is the variance.
# The larger kappa is, the more exploration the acquisition function does.
@inline ucb_score(μ::Float64, s2::Float64, κ::Float64) = μ + κ * sqrt(max(s2, 0.0))


"""
    recommend_mean(gp, bounds; M, rng) -> (x_rec, m_rec, s_rec)

Find the point with highest GP posterior mean within `bounds`.

Global search over `M` random candidates, then local refinement with
bounded L-BFGS from the best candidate.

Returns `(x_rec, m_rec, s_rec)` where `m_rec` is the posterior mean
and `s_rec` is the posterior std at `x_rec` (useful for LCB stopping checks).
"""
function recommend_mean(gp::HeteroGP, bounds; M::Int=20000, rng::Random.AbstractRNG=Random.default_rng())
    M ≥ 1 || throw(ArgumentError("M must be ≥ 1"))
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    # Global random search
    best_x = _rand_in_box(rng, lb, ub)
    μ0, s2_0 = predict_latent(gp, best_x)
    best_m = μ0
    best_s = sqrt(max(s2_0, 0.0))
    for _ in 1:M
        x = _rand_in_box(rng, lb, ub)
        μ, s2 = predict_latent(gp, x)
        if μ > best_m
            best_m = μ
            best_s = sqrt(max(s2, 0.0))
            best_x = x
        end
    end

    # Local refinement: bounded L-BFGS with finite-diff gradients.
    # Refines the best random candidate to sub-grid precision.
    try
        res = optimize(x -> begin μ, _ = predict_latent(gp, x); -μ end,
                       lb, ub, copy(best_x),
                       Fminbox(LBFGS()),
                       Optim.Options(iterations=100, g_tol=1e-5, f_abstol=1e-10);
                       autodiff=:finite)
        x_ref = Optim.minimizer(res)
        μ_ref, s2_ref = predict_latent(gp, x_ref)
        if μ_ref > best_m
            best_m = μ_ref
            best_s = sqrt(max(s2_ref, 0.0))
            best_x = x_ref
        end
    catch
        # If refinement fails (rare), fall back to random-search result
    end

    return best_x, best_m, best_s
end

# -------------------------
# Main algorithm
# -------------------------

"""
    bayesopt_ucb_threshold(f; bounds, n_shots, ...)

Heteroscedastic BO with GP-UCB acquisition for x and thresholding rule for σ selection.
Objective is `f(x, σ)`.

Set `maximize=false` to minimize.
Use `seed` for determinism without affecting the global RNG.
"""
function bayesopt_ucb_threshold(f;
                               bounds::Vector{Tuple{Float64,Float64}},
                               n_shots::Int=400,
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
                               min_iter::Int=0,
                               sigma_mode::Symbol=:simple,
                               n_checks::Int=2,
                               add_check_points::Bool=true,
                               explore_frac::Float64=0.0,
                               init_sampling::Symbol=:random,
                               acq_sampling::Symbol=:random,
                               fixed_init_seed::Union{Nothing,Int}=nothing,
                               n_restarts::Int=6)

    _validate_bounds(bounds)
    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M_acq ≥ 1  || throw(ArgumentError("M_acq must be ≥ 1"))
    M_rec ≥ 1  || throw(ArgumentError("M_rec must be ≥ 1"))
    κ ≥ 0      || throw(ArgumentError("κ must be ≥ 0"))
    α ≥ 0      || throw(ArgumentError("α must be ≥ 0"))
    freeze_mode ∈ (:none, :all, :lengthscales, :lengthscales_from, :all_from) || throw(ArgumentError("freeze_mode must be :none, :all, :lengthscales, :lengthscales_from, or :all_from"))
    min_iter ≥ 0   || throw(ArgumentError("min_iter must be ≥ 0"))
    n_shots ≥ 1   || throw(ArgumentError("n_shots must be ≥ 1"))
    sigma_mode ∈ (:simple, :binomial) || throw(ArgumentError("sigma_mode must be :simple or :binomial"))
    n_checks ∈ (1, 2) || throw(ArgumentError("n_checks must be 1 or 2"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    d = length(bounds)
    # Pre-allocate with extra capacity for x_rec check calls:
    # up to 2 check calls per BO iteration (stage-1 + stage-2 if threshold is set).
    n_cap = n_init + 3 * n_iter
    X  = Matrix{Float64}(undef, d, n_cap)
    y  = Vector{Float64}(undef, n_cap)
    σy = Vector{Float64}(undef, n_cap)
    write_idx        = 0  # tracks points written to training data
    total_shots_count = 0  # tracks ALL shots used, including check calls not added to data

    # Init RNG: fixed_init_seed pins the initial design (same points every run);
    # nothing (default) uses rng_local so init varies with the run seed.
    rng_init = fixed_init_seed === nothing ? rng_local : MersenneTwister(fixed_init_seed)
    init_pts = init_sampling === :sobol ?
        _sobol_in_box(rng_init, lb, ub, n_init) :
        nothing
    for i in 1:n_init
        x = init_sampling === :sobol ? init_pts[:, i] : _rand_in_box(rng_init, lb, ub)
        y_raw, σy_i = _call_f_raw(f, x, n_shots)
        total_shots_count += n_shots
        write_idx += 1
        X[:, write_idx] = x
        y[write_idx]  = maximize ? y_raw : -y_raw
        σy[write_idx] = σy_i
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
    frozen_ℓ_from = nothing   # captured at it==n_freeze_iters for :lengthscales_from
    frozen_θ_from = nothing   # captured at it==n_freeze_iters for :all_from

    for it in 1:n_iter
        # GP fit uses all data collected so far (init + previous acquisitions + check calls)
        if freeze_mode == :all
            gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                              θ_init=pretrained_θ,
                              learn_hypers=false,
                              learn_noise_scale=true,
                              jitter=1e-8,
                              rng=rng_local)
        elseif freeze_mode == :all_from && it > n_freeze_iters && frozen_θ_from !== nothing
            # All hyperparameters frozen at the values captured at it==n_freeze_iters
            gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                              θ_init=frozen_θ_from,
                              learn_hypers=false,
                              learn_noise_scale=false,
                              jitter=1e-8,
                              rng=rng_local)
        else
            current_fixed_ℓ = if freeze_mode == :lengthscales && it <= n_freeze_iters
                fixed_ℓ_bo
            elseif freeze_mode == :lengthscales_from && it > n_freeze_iters
                frozen_ℓ_from
            else
                nothing
            end
            releasing = (freeze_mode == :lengthscales && it == n_freeze_iters + 1)
            do_opt = (it == 1) || releasing || (hyper_every > 0 && it % hyper_every == 0)
            gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                              θ_init=θ_prev,
                              fixed_ℓ=current_fixed_ℓ,
                              learn_hypers=do_opt,
                              learn_noise_scale=true,
                              n_restarts=do_opt ? n_restarts : 0,
                              jitter=1e-8,
                              rng=rng_local)
            if freeze_mode == :lengthscales_from && it == n_freeze_iters
                frozen_ℓ_from = copy(gp.ℓ)
            end
            if freeze_mode == :all_from && it == n_freeze_iters
                frozen_θ_from = copy(gp.θ)
            end
        end

        θ_prev = gp.θ

        acq_pts = acq_sampling === :sobol ?
            _sobol_in_box(rng_local, lb, ub, M_acq) :
            nothing

        best_x = acq_sampling === :sobol ? acq_pts[:, 1] : _rand_in_box(rng_local, lb, ub)
        best_a = -Inf
        best_s2 = 0.0

        for i in 1:M_acq
            x = acq_sampling === :sobol ? acq_pts[:, i] : _rand_in_box(rng_local, lb, ub)
            μ, s2 = predict_latent(gp, x)
            a = ucb_score(μ, s2, κ)
            if a > best_a
                best_a = a
                best_x = x
                best_s2 = s2
            end
        end

        # Store acquisition point (always evaluate; add to data only if not a near-duplicate)
        y_raw, σy_i = _call_f_raw(f, best_x, n_shots)
        total_shots_count += n_shots
        if _is_far_enough(best_x, X, write_idx)
            write_idx += 1
            X[:, write_idx] = best_x
            y[write_idx]  = maximize ? y_raw : -y_raw
            σy[write_idx] = σy_i
        end
        y_last_val = maximize ? y_raw : -y_raw

        if verbose
            @info "it=$it best_acq=$best_a"
        end

        # x_rec check: evaluate the GP-recommended point every iteration when early stopping
        # is configured. The GP benefits from knowing the true value at its current best guess.
        # Stopping decision is gated on min_iter; data collection (if add_check_points) is not.
        if fidelity_threshold !== nothing
            x_rec_es, _, _ = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)

            # Stage 1 — always evaluated
            y1_raw, σy1_i = _call_f_raw(f, x_rec_es, n_shots)
            total_shots_count += n_shots
            if add_check_points && _is_far_enough(x_rec_es, X, write_idx)
                write_idx += 1
                X[:, write_idx] = x_rec_es
                y[write_idx]  = maximize ? y1_raw : -y1_raw
                σy[write_idx] = σy1_i
            end

            if it >= min_iter
                y1 = maximize ? y1_raw : -y1_raw
                reached1 = maximize ? (y1 >= fidelity_threshold) : (y1 <= fidelity_threshold)

                if reached1
                    if n_checks == 1
                        # Single-check stopping: stage 1 alone is sufficient
                        n_iter_actual = it
                        y_out_es = maximize ? y[1:write_idx] : -y[1:write_idx]
                        return HeteroBOResult(X[:, 1:write_idx], y_out_es, σy[1:write_idx],
                                              bounds, n_shots, n_init, n_iter, maximize,
                                              x_rec_es, y1_raw, n_iter_actual, y1_raw,
                                              gp.ℓ, gp.σf, gp.c, total_shots_count)
                    else
                        # Stage 2 — only when stage 1 passes; not added to data (same x)
                        y2_raw, _ = _call_f_raw(f, x_rec_es, n_shots)
                        total_shots_count += n_shots

                        y2 = maximize ? y2_raw : -y2_raw
                        reached2 = maximize ? (y2 >= fidelity_threshold) : (y2 <= fidelity_threshold)

                        if reached2
                            n_iter_actual = it
                            y_rec_es = (y1_raw + y2_raw) / 2
                            y_out_es = maximize ? y[1:write_idx] : -y[1:write_idx]
                            return HeteroBOResult(X[:, 1:write_idx], y_out_es, σy[1:write_idx],
                                                  bounds, n_shots, n_init, n_iter, maximize,
                                                  x_rec_es, y_rec_es, n_iter_actual, y_rec_es,
                                                  gp.ℓ, gp.σf, gp.c, total_shots_count)
                        end
                    end
                end
            end
        end
    end

    final_fixed_ℓ = if freeze_mode == :lengthscales && n_iter_actual <= n_freeze_iters
        fixed_ℓ_bo
    elseif freeze_mode == :lengthscales_from && n_iter_actual > n_freeze_iters && frozen_ℓ_from !== nothing
        frozen_ℓ_from
    else
        nothing
    end
    freeze_all_final = freeze_mode == :all ||
                       (freeze_mode == :all_from && n_iter_actual > n_freeze_iters && frozen_θ_from !== nothing)
    final_θ_init = if freeze_mode == :all
        pretrained_θ
    elseif freeze_mode == :all_from && frozen_θ_from !== nothing
        frozen_θ_from
    else
        θ_prev
    end
    gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                      θ_init=final_θ_init,
                      fixed_ℓ=final_fixed_ℓ,
                      learn_hypers=!freeze_all_final,
                      learn_noise_scale=!freeze_all_final,
                      n_restarts=freeze_all_final ? 0 : n_restarts + 2,
                      jitter=1e-8,
                      rng=rng_local)

    x_rec, _, _ = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
    y_rec_raw, _ = _call_f_raw(f, x_rec, n_shots)
    total_shots_count += n_shots

    y_out = maximize ? y[1:write_idx] : -y[1:write_idx]
    y_last_out = maximize ? y_last_val : -y_last_val

    return HeteroBOResult(X[:, 1:write_idx], y_out, σy[1:write_idx], bounds, n_shots,
                          n_init, n_iter, maximize, x_rec, y_rec_raw, n_iter_actual,
                          y_last_out, gp.ℓ, gp.σf, gp.c, total_shots_count)
end
