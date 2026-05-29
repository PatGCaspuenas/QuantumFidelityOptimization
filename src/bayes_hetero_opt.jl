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
    p_opt = learn_noise_scale ? d + 2 : d + 1

    lower = Vector{Float64}(undef, p_opt)
    upper = Vector{Float64}(undef, p_opt)
    lower[1:d] .= log(ℓ_bounds[1]);  upper[1:d] .= log(ℓ_bounds[2])
    lower[d+1]  = log(σf_bounds[1]); upper[d+1]  = log(σf_bounds[2])
    if learn_noise_scale
        lower[d+2] = log(c_bounds[1]); upper[d+2] = log(c_bounds[2])
    end

    function chol_from_θ(θ::Vector{Float64})
        ℓ  = exp.(θ[1:d])
        σf = exp(θ[d+1])
        c  = learn_noise_scale ? exp(θ[d+2]) : 1.0

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
        αtmp = F \ ystd
        return 0.5 * dot(ystd, αtmp) + sum(log, diag(F.L)) + 0.5 * n * log(2π)
    end

    _θ_ε = 1e-4
    θ0 = Vector{Float64}(undef, p_opt)
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

    if !learn_hypers
        F = chol_from_θ(θ0)
        F === nothing && throw(ArgumentError("Cholesky failed with fixed hyperparameters; increase jitter or adjust bounds."))
        ℓ  = exp.(θ0[1:d])
        σf = exp(θ0[d+1])
        c  = learn_noise_scale ? exp(θ0[d+2]) : 1.0
        L = F.L
        α = L' \ (L \ ystd)
        return HeteroGP(X, yμ, yσ, ℓ, σf, c, L, α, copy(θ0))
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
    ℓ  = exp.(bestθ[1:d])
    σf = exp(bestθ[d+1])
    c  = learn_noise_scale ? exp(bestθ[d+2]) : 1.0
    L = F.L
    α = L' \ (L \ ystd)

    return HeteroGP(X, yμ, yσ, ℓ, σf, c, L, α, copy(bestθ))
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
    n_shots::Union{Int,Float64}           # shots per f call; Inf means deterministic expectation
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
    total_shots::Union{Int,Float64}       # total shots used across ALL f calls (incl. check calls not in X)
end

Base.@kwdef struct PathGuardConfig
    init_design::Symbol = :random
    init_center_exclusion_radius::Float64 = 0.0
    init_sobol_skip_max::Int = 2048
    decision_q_max::Float64 = 1.0
    global_scout_period::Int = 7
    global_scout_frac::Float64 = 0.15
    stagnation_window::Int = 12
    boundary_margin::Float64 = 0.03
    ell_hi::Float64 = 1.45
    sigmaf_lo::Float64 = 0.35
    c_hi::Float64 = 2.5
    trust_start_q::Float64 = 0.70
    strong_trust_q::Float64 = 0.90
    trust_radius::Float64 = 0.25
    trust_radius_min::Float64 = 0.12
    mid_local_frac::Float64 = 0.35
    strong_local_frac::Float64 = 0.70
    min_support_count::Int = 2
    support_radius::Float64 = 0.25
    scout_batch::Int = 2
    pretrust_kappa::Float64 = 2.25
    mid_kappa::Float64 = 1.90
    exploit_kappa::Float64 = 0.50
end

function _validate_shot_count(n_shots::Real, name::String)
    n = Float64(n_shots)
    if isinf(n) && n > 0.0
        return nothing
    end
    isfinite(n) || throw(ArgumentError("$name must be a positive integer or Inf."))
    isinteger(n) || throw(ArgumentError("$name must be a positive integer or Inf, got $n_shots."))
    n ≥ 1 || throw(ArgumentError("$name must be ≥ 1, got $n_shots"))
    return nothing
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
# y  is the fidelity observation (linear Q scale).
# σy is the GP observation noise: √(Q(1-Q)/N) for binomial noise model.
@inline function _call_f_raw(f, x::Vector{Float64}, n::Real)
    result = f(x, n)
    result isa Tuple || throw(ArgumentError("f must return a (y, σy) tuple"))
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

function _rand_near_box(rng::Random.AbstractRNG, center::Vector{Float64},
                        lb::Vector{Float64}, ub::Vector{Float64}, radius::Float64)
    x = Vector{Float64}(undef, length(lb))
    @inbounds for j in eachindex(lb)
        x[j] = clamp(center[j] + (2.0 * rand(rng) - 1.0) * radius, lb[j], ub[j])
    end
    return x
end

function _latin_hypercube_points(rng::Random.AbstractRNG, lb::Vector{Float64},
                                 ub::Vector{Float64}, n_init::Int)
    X = QuasiMonteCarlo.sample(n_init, lb, ub, LatinHypercubeSample(rng))
    return [Vector{Float64}(X[:, i]) for i in 1:n_init]
end

_is_prime_int(n::Int) = n >= 2 && all(n % k != 0 for k in 2:floor(Int, sqrt(n)))

@inline _gf4_add(a::Int, b::Int) = xor(a, b)

@inline function _gf4_mul(a::Int, b::Int)
    (a == 0 || b == 0) && return 0
    # GF(4) with primitive polynomial x^2 + x + 1.
    table = ((1, 2, 3),
             (2, 3, 1),
             (3, 1, 2))
    return table[a][b]
end

@inline function _oa_symbol_sum(a::Int, b::Int, slope::Int, p::Int)
    if p == 4
        return _gf4_add(a, _gf4_mul(slope, b))
    end
    return mod(a + slope * b, p)
end

function _oa_lhs_points(rng::Random.AbstractRNG, lb::Vector{Float64},
                        ub::Vector{Float64}, n_init::Int)
    p = round(Int, sqrt(n_init))
    p * p == n_init ||
        throw(ArgumentError("orthogonal-array LHS requires n_init = p^2, got $n_init"))
    (_is_prime_int(p) || p == 4) ||
        throw(ArgumentError("orthogonal-array LHS requires prime or supported prime-power p, got p=$p"))
    d = length(lb)
    d <= p + 1 ||
        throw(ArgumentError("orthogonal-array LHS requires dimension <= p + 1; got d=$d, p=$p"))

    A = Matrix{Int}(undef, n_init, d)
    row = 0
    for a in 0:(p - 1), b in 0:(p - 1)
        row += 1
        A[row, 1] = a
        d >= 2 && (A[row, 2] = b)
        for j in 3:d
            slope = j - 2
            A[row, j] = _oa_symbol_sum(a, b, slope, p)
        end
    end

    X = Matrix{Float64}(undef, d, n_init)
    @inbounds for j in 1:d
        symbol_map = randperm(rng, p) .- 1
        symbols = [symbol_map[A[i, j] + 1] for i in 1:n_init]
        span = ub[j] - lb[j]
        for level in 0:(p - 1)
            rows = findall(==(level), symbols)
            sublevels = randperm(rng, p) .- 1
            for (k, i) in enumerate(rows)
                fine_stratum = level * p + sublevels[k]
                u = (fine_stratum + rand(rng)) / n_init
                X[j, i] = lb[j] + u * span
            end
        end
    end
    return [Vector{Float64}(X[:, i]) for i in 1:n_init]
end

function _path_guard_initial_points(rng::Random.AbstractRNG, lb::Vector{Float64},
                                    ub::Vector{Float64}, n_init::Int,
                                    cfg::PathGuardConfig)
    if cfg.init_design == :random
        return [_rand_in_box(rng, lb, ub) for _ in 1:n_init]
    end

    if cfg.init_design in (:lhs, :latin_hypercube, :latin_hypercube_random)
        return _latin_hypercube_points(rng, lb, ub, n_init)
    end

    if cfg.init_design in (:oa_lhs, :orthogonal_lhs, :orthogonal_array_lhs,
                           :orthogonal_latin_hypercube)
        return _oa_lhs_points(rng, lb, ub, n_init)
    end

    d = length(lb)
    center = 0.5 .* (lb .+ ub)
    points = Vector{Vector{Float64}}()

    if cfg.init_design in (:sobol_random, :sobol_randomized, :sobol)
        skip = rand(rng, 0:max(cfg.init_sobol_skip_max, 0))
        shift = rand(rng, d)
        need = n_init + skip + 64
        while length(points) < n_init
            qmc = QuasiMonteCarlo.sample(need, lb, ub, SobolSample())
            for i in (skip + 1):size(qmc, 2)
                x = Vector{Float64}(qmc[:, i])
                # A random digital shift keeps the low-discrepancy structure
                # while preventing every seed from seeing the same design.
                @inbounds for j in 1:d
                    span = ub[j] - lb[j]
                    u = (x[j] - lb[j]) / span
                    u = mod(u + shift[j], 1.0)
                    x[j] = lb[j] + u * span
                end
                norm(x .- center) < cfg.init_center_exclusion_radius && continue
                push!(points, x)
                length(points) >= n_init && break
            end
            need *= 2
        end
        return points[1:n_init]
    end

    throw(ArgumentError("unknown init_design=$(cfg.init_design); use :random, :latin_hypercube, :oa_lhs, or :sobol_random"))
end

function _min_dist2_to_design(x::Vector{Float64}, X::Matrix{Float64}, n::Int)
    n == 0 && return Inf
    best = Inf
    @inbounds for i in 1:n
        d2 = 0.0
        for j in eachindex(x)
            delta = x[j] - X[j, i]
            d2 += delta * delta
        end
        best = min(best, d2)
    end
    return best
end

function _maximin_candidate(rng::Random.AbstractRNG, lb::Vector{Float64},
                            ub::Vector{Float64}, X::Matrix{Float64}, n::Int;
                            M::Int=1000)
    best_x = _rand_in_box(rng, lb, ub)
    best_d2 = _min_dist2_to_design(best_x, X, n)
    for _ in 2:max(M, 2)
        x = _rand_in_box(rng, lb, ub)
        d2 = _min_dist2_to_design(x, X, n)
        if d2 > best_d2
            best_d2 = d2
            best_x = x
        end
    end
    return best_x
end

function _local_support_stats(X::Matrix{Float64}, y::Vector{Float64}, n::Int,
                              center::Vector{Float64}, radius::Float64)
    count = 0
    total = 0.0
    r2 = radius * radius
    @inbounds for i in 1:n
        d2 = 0.0
        for j in eachindex(center)
            delta = center[j] - X[j, i]
            d2 += delta * delta
        end
        if d2 <= r2
            count += 1
            total += y[i]
        end
    end
    return count, count > 0 ? total / count : -Inf
end

function _best_supported_observation(X::Matrix{Float64}, y::Vector{Float64}, n::Int,
                                     cfg::PathGuardConfig)
    best_avg = -Inf
    best_count = 0
    best_x = Vector{Float64}(undef, size(X, 1))
    @inbounds for i in 1:n
        x = Vector{Float64}(X[:, i])
        count, avg = _local_support_stats(X, y, n, x, cfg.support_radius)
        if count >= cfg.min_support_count && avg > best_avg
            best_avg = avg
            best_count = count
            best_x = x
        end
    end
    return best_x, best_avg, best_count
end

function _near_boundary(x::Vector{Float64}, lb::Vector{Float64},
                        ub::Vector{Float64}, margin::Float64)
    @inbounds for j in eachindex(x)
        span = ub[j] - lb[j]
        if x[j] - lb[j] <= margin * span || ub[j] - x[j] <= margin * span
            return true
        end
    end
    return false
end

function _gp_health_bad(gp::HeteroGP, cfg::PathGuardConfig)
    return count(>=(cfg.ell_hi), gp.ℓ) >= 2 ||
           gp.σf <= cfg.sigmaf_lo ||
           gp.c >= cfg.c_hi
end

@inline _decision_mean(μ::Float64, cfg::PathGuardConfig) =
    clamp(μ, 0.0, cfg.decision_q_max)

@inline function _path_guard_kappa(supported_q::Float64, cfg::PathGuardConfig)
    supported_q >= cfg.strong_trust_q && return cfg.exploit_kappa
    supported_q >= cfg.trust_start_q && return cfg.mid_kappa
    return cfg.pretrust_kappa
end

@inline function _path_guard_local_frac(supported_q::Float64, cfg::PathGuardConfig)
    supported_q >= cfg.strong_trust_q && return cfg.strong_local_frac
    supported_q >= cfg.trust_start_q && return cfg.mid_local_frac
    return 0.0
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

function _recommend_mean_path_guard(gp::HeteroGP, bounds, cfg::PathGuardConfig;
                                    M::Int=20000,
                                    rng::Random.AbstractRNG=Random.default_rng(),
                                    center::Union{Nothing,Vector{Float64}}=nothing,
                                    local_frac::Float64=0.0)
    M ≥ 1 || throw(ArgumentError("M must be ≥ 1"))
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]
    best_x = _rand_in_box(rng, lb, ub)
    μ0, s2_0 = predict_latent(gp, best_x)
    best_m = _decision_mean(μ0, cfg)
    best_s = sqrt(max(s2_0, 0.0))
    for _ in 1:M
        use_local = center !== nothing && rand(rng) < local_frac
        x = use_local ? _rand_near_box(rng, center, lb, ub, cfg.trust_radius) :
                        _rand_in_box(rng, lb, ub)
        μ, s2 = predict_latent(gp, x)
        μ_dec = _decision_mean(μ, cfg)
        if μ_dec > best_m
            best_m = μ_dec
            best_s = sqrt(max(s2, 0.0))
            best_x = x
        end
    end
    return best_x, best_m, best_s
end

# -------------------------
# Variable-N helper: variable mode
# -------------------------

# Adaptive measurement for :variable mode (binomial noise, linear Q scale).
# Starts with n_floor shots, accumulates in batches until:
#   - Q falls below threshold (was noise) → stop_loop=false
#   - n_max reached with Q still above threshold → stop_loop=true (confirmed above)
# Returns (y_out, σy_out, N_total_used, stop_loop).
function _adaptive_measure(f, x::Vector{Float64},
                           n_floor::Int, n_max::Int,
                           threshold::Float64,
                           maximize::Bool,
                           batch_size::Int=50)
    y1, _ = _call_f_raw(f, x, n_floor)
    Q_cur   = clamp(y1, 0.0, 1.0)
    k_total = Q_cur * Float64(n_floor)
    N_total = n_floor

    # Only activate if initial measurement is above threshold
    at_or_above = maximize ? (Q_cur >= threshold) : (Q_cur <= threshold)
    if !at_or_above
        σy_f = sqrt(max(Q_cur * (1.0 - Q_cur), 0.0) / N_total)
        return Q_cur, σy_f, N_total, false
    end

    while N_total < n_max
        Q_cur  = k_total / Float64(N_total)
        fell_below = maximize ? (Q_cur < threshold) : (Q_cur > threshold)
        fell_below && break

        Δ = min(batch_size, n_max - N_total)
        y_new, _ = _call_f_raw(f, x, Δ)
        Q_new = clamp(y_new, 0.0, 1.0)
        k_total += Q_new * Float64(Δ)
        N_total += Δ
    end

    Q_final  = k_total / Float64(N_total)
    stop_loop = maximize ? (N_total >= n_max && Q_final >= threshold) :
                           (N_total >= n_max && Q_final <= threshold)
    σy_final = sqrt(max(Q_final * (1.0 - Q_final), 0.0) / Float64(N_total))

    return Q_final, σy_final, N_total, stop_loop
end

# -------------------------
# Main algorithm
# -------------------------

"""
    bayesopt_ucb_threshold(f; bounds, n_shots, ...)

Heteroscedastic BO with GP-UCB acquisition. Noise model: binomial (√(Q(1-Q)/N)).

`use_variable_mode`:
  - `false` — fixed N_shots per acquisition; optional threshold check via two noisy evaluations.
  - `true`  — adaptive shots at x_rec each iteration; early stop when confirmed above threshold.
              Acquisition point always uses n_floor shots. Requires `fidelity_threshold`.

Set `maximize=false` to minimize.
Use `seed` for determinism without affecting the global RNG.
"""
function bayesopt_ucb_threshold(f;
                               bounds::Vector{Tuple{Float64,Float64}},
                               n_shots::Union{Int,Float64}=400,
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
                               fidelity_threshold::Union{Nothing,Float64}=nothing,
                               explore_frac::Float64=0.0,
                               path_guard::Bool=false,
                               path_guard_config::PathGuardConfig=PathGuardConfig(),
                               learn_noise_scale::Bool=true,
                               n_restarts::Int=6,
                               use_variable_mode::Bool=false,
                               n_floor::Int=50,
                               n_max_shots::Int=2000)

    _validate_bounds(bounds)
    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M_acq ≥ 1  || throw(ArgumentError("M_acq must be ≥ 1"))
    M_rec ≥ 1  || throw(ArgumentError("M_rec must be ≥ 1"))
    κ ≥ 0      || throw(ArgumentError("κ must be ≥ 0"))
    α ≥ 0      || throw(ArgumentError("α must be ≥ 0"))
    0.0 <= explore_frac <= 1.0 || throw(ArgumentError("explore_frac must be in [0, 1]"))
    _validate_shot_count(n_shots, "n_shots")
    n_floor ≥ 1 || throw(ArgumentError("n_floor must be ≥ 1"))
    n_max_shots ≥ n_floor || throw(ArgumentError("n_max_shots must be ≥ n_floor"))
    (use_variable_mode && fidelity_threshold === nothing) &&
        throw(ArgumentError("fidelity_threshold required for use_variable_mode=true"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    d = length(bounds)
    # Pre-allocate with extra capacity for x_rec check calls.
    n_cap = n_init + 3 * n_iter
    X  = Matrix{Float64}(undef, d, n_cap)
    y  = Vector{Float64}(undef, n_cap)
    σy = Vector{Float64}(undef, n_cap)
    write_idx         = 0
    total_shots_count = 0

    # Initial design: uniform random by default; optional structured design for
    # path-guard runs so initial coverage does not depend on a lucky seed.
    init_points = (path_guard || path_guard_config.init_design != :random) ?
        _path_guard_initial_points(rng_local, lb, ub, n_init, path_guard_config) :
        [_rand_in_box(rng_local, lb, ub) for _ in 1:n_init]
    for x in init_points
        y_raw, σy_i = _call_f_raw(f, x, n_shots)
        total_shots_count += n_shots
        write_idx += 1
        X[:, write_idx] = x
        y[write_idx]  = maximize ? y_raw : -y_raw
        σy[write_idx] = σy_i
    end

    θ_prev = nothing
    n_iter_actual = n_iter
    y_last_val = 0.0
    scout_remaining = 0
    last_supported_q = -Inf
    stagnation_count = 0
    boundary_streak = 0

    for it in 1:n_iter
        do_opt = (it == 1) || (hyper_every > 0 && it % hyper_every == 0)
        gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                          θ_init=θ_prev,
                          learn_hypers=do_opt,
                          learn_noise_scale=learn_noise_scale,
                          n_restarts=do_opt ? n_restarts : 0,
                          jitter=1e-8,
                          rng=rng_local)
        θ_prev = gp.θ

        supported_x, supported_q, supported_count =
            path_guard ? _best_supported_observation(X, y, write_idx, path_guard_config) :
                         (Vector{Float64}(), -Inf, 0)
        if path_guard
            if supported_q > last_supported_q + 1e-6
                last_supported_q = supported_q
                stagnation_count = 0
            else
                stagnation_count += 1
            end
            if _gp_health_bad(gp, path_guard_config) ||
               stagnation_count >= path_guard_config.stagnation_window ||
               boundary_streak >= 3
                scout_remaining = max(scout_remaining, path_guard_config.scout_batch)
            end
        end

        # Acquisition: GP-UCB by default. `explore_frac` now forces genuine
        # maximin scouts instead of being a no-op.
        force_scout = false
        if path_guard
            force_scout = scout_remaining > 0 ||
                          (path_guard_config.global_scout_period > 0 &&
                           it % path_guard_config.global_scout_period == 0) ||
                          rand(rng_local) < path_guard_config.global_scout_frac
        else
            force_scout = explore_frac > 0.0 && rand(rng_local) < explore_frac
        end

        best_x = _rand_in_box(rng_local, lb, ub)
        best_a = -Inf

        if force_scout
            best_x = _maximin_candidate(rng_local, lb, ub, X, write_idx; M=M_acq)
            μ, s2 = predict_latent(gp, best_x)
            μ_dec = path_guard ? _decision_mean(μ, path_guard_config) : μ
            best_a = ucb_score(μ_dec, s2, path_guard ? _path_guard_kappa(supported_q, path_guard_config) : κ)
            scout_remaining = max(scout_remaining - 1, 0)
        else
            κ_eff = path_guard ? _path_guard_kappa(supported_q, path_guard_config) : κ
            local_frac = path_guard ? _path_guard_local_frac(supported_q, path_guard_config) : 0.0
            for _ in 1:M_acq
                use_local = path_guard && supported_count >= path_guard_config.min_support_count &&
                            rand(rng_local) < local_frac
                x = use_local ? _rand_near_box(rng_local, supported_x, lb, ub,
                                               max(path_guard_config.trust_radius_min,
                                                   path_guard_config.trust_radius)) :
                                _rand_in_box(rng_local, lb, ub)
                μ, s2 = predict_latent(gp, x)
                μ_dec = path_guard ? _decision_mean(μ, path_guard_config) : μ
                a = ucb_score(μ_dec, s2, κ_eff)
                if a > best_a
                    best_a = a
                    best_x = x
                end
            end
        end

        # Acquisition N: n_shots for fixed mode, n_floor for variable mode
        n_acq = use_variable_mode ? n_floor : n_shots

        y_raw, σy_i = _call_f_raw(f, best_x, n_acq)
        total_shots_count += n_acq
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

        # --- variable mode: adaptive measurement at x_rec ---
        if use_variable_mode
            rec_center = path_guard && supported_count >= path_guard_config.min_support_count ? supported_x : nothing
            rec_local_frac = path_guard ? _path_guard_local_frac(supported_q, path_guard_config) : 0.0
            x_rec_cur, _, _ = path_guard ?
                _recommend_mean_path_guard(gp, bounds, path_guard_config;
                                           M=M_rec, rng=rng_local,
                                           center=rec_center,
                                           local_frac=rec_local_frac) :
                recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
            y_rec_cur, σy_rec_cur, n_rec, stop_loop = _adaptive_measure(
                f, x_rec_cur, n_floor, n_max_shots, fidelity_threshold, maximize)
            total_shots_count += n_rec

            if _is_far_enough(x_rec_cur, X, write_idx)
                write_idx += 1
                X[:, write_idx] = x_rec_cur
                y[write_idx]  = maximize ? y_rec_cur : -y_rec_cur
                σy[write_idx] = σy_rec_cur
            end

            stop_supported = true
            if path_guard
                support_n, support_avg = _local_support_stats(
                    X, y, write_idx, x_rec_cur, path_guard_config.support_radius)
                stop_supported = support_n >= path_guard_config.min_support_count &&
                                 support_avg >= path_guard_config.trust_start_q &&
                                 !_gp_health_bad(gp, path_guard_config)
            end

            if stop_loop && stop_supported
                n_iter_actual = it
                y_out = maximize ? y[1:write_idx] : -y[1:write_idx]
                return HeteroBOResult(X[:, 1:write_idx], y_out, σy[1:write_idx],
                                      bounds, n_shots, n_init, n_iter, maximize,
                                      x_rec_cur, y_rec_cur, n_iter_actual,
                                      maximize ? y_last_val : -y_last_val,
                                      gp.ℓ, gp.σf, gp.c, total_shots_count)
            end
        end

        # --- Fixed-N threshold check: two noisy evaluations at x_rec ---
        if fidelity_threshold !== nothing && !use_variable_mode
            rec_center = path_guard && supported_count >= path_guard_config.min_support_count ? supported_x : nothing
            rec_local_frac = path_guard ? _path_guard_local_frac(supported_q, path_guard_config) : 0.0
            x_rec_cur, _, _ = path_guard ?
                _recommend_mean_path_guard(gp, bounds, path_guard_config;
                                           M=M_rec, rng=rng_local,
                                           center=rec_center,
                                           local_frac=rec_local_frac) :
                recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
            y_rec_cur, σy1_i = _call_f_raw(f, x_rec_cur, n_shots)
            total_shots_count += n_shots
            if _is_far_enough(x_rec_cur, X, write_idx)
                write_idx += 1
                X[:, write_idx] = x_rec_cur
                y[write_idx]  = maximize ? y_rec_cur : -y_rec_cur
                σy[write_idx] = σy1_i
            end

            y1 = maximize ? y_rec_cur : -y_rec_cur
            if path_guard
                boundary_streak = _near_boundary(x_rec_cur, lb, ub, path_guard_config.boundary_margin) ?
                    boundary_streak + 1 : 0
            end
            stop_supported = true
            if path_guard
                support_n, support_avg = _local_support_stats(
                    X, y, write_idx, x_rec_cur, path_guard_config.support_radius)
                stop_supported = support_n >= path_guard_config.min_support_count &&
                                 support_avg >= path_guard_config.trust_start_q &&
                                 !_gp_health_bad(gp, path_guard_config)
            end
            reached1 = maximize ? (y1 >= fidelity_threshold) : (y1 <= fidelity_threshold)
            if reached1 && stop_supported
                y2_raw, _ = _call_f_raw(f, x_rec_cur, n_shots)
                total_shots_count += n_shots
                y2 = maximize ? y2_raw : -y2_raw
                reached2 = maximize ? (y2 >= fidelity_threshold) : (y2 <= fidelity_threshold)
                if reached2
                    n_iter_actual = it
                    y_rec_avg = (y_rec_cur + y2_raw) / 2
                    y_out_es = maximize ? y[1:write_idx] : -y[1:write_idx]
                    return HeteroBOResult(X[:, 1:write_idx], y_out_es, σy[1:write_idx],
                                          bounds, n_shots, n_init, n_iter, maximize,
                                          x_rec_cur, y_rec_avg, n_iter_actual,
                                          maximize ? y_last_val : -y_last_val,
                                          gp.ℓ, gp.σf, gp.c, total_shots_count)
                end
            end
        end
    end

    # Final GP fit with extra restarts
    gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                      θ_init=θ_prev,
                      learn_hypers=true,
                      learn_noise_scale=learn_noise_scale,
                      n_restarts=n_restarts + 2,
                      jitter=1e-8,
                      rng=rng_local)

    if path_guard
        supported_x, supported_q, supported_count =
            _best_supported_observation(X, y, write_idx, path_guard_config)
        rec_center = supported_count >= path_guard_config.min_support_count ? supported_x : nothing
        x_rec, _, _ = _recommend_mean_path_guard(
            gp, bounds, path_guard_config;
            M=M_rec, rng=rng_local, center=rec_center,
            local_frac=_path_guard_local_frac(supported_q, path_guard_config))
    else
        x_rec, _, _ = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
    end
    y_rec_raw, _ = _call_f_raw(f, x_rec, n_shots)
    total_shots_count += n_shots

    y_out = maximize ? y[1:write_idx] : -y[1:write_idx]
    y_last_out = maximize ? y_last_val : -y_last_val

    return HeteroBOResult(X[:, 1:write_idx], y_out, σy[1:write_idx], bounds, n_shots,
                          n_init, n_iter, maximize, x_rec, y_rec_raw, n_iter_actual,
                          y_last_out, gp.ℓ, gp.σf, gp.c, total_shots_count)
end
