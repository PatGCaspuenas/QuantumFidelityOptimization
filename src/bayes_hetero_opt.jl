# src/bayes_hetero_opt.jl
# Heteroscedastic GP + GP-UCB acquisition (random scan, 3D, fixed N)

using Random
using Statistics
using LinearAlgebra
using Optim

# Kernel and covariance
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
    L::LowerTriangular{Float64,Matrix{Float64}}
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

Heteroscedastic GP with Matérn 3/2 ARD kernel and known per-point noise σyᵢ.
Optimizes θ = (logℓ, logσf, logc) by bounded LML minimization with multi-start.
Restart strategy: r=1 warm-starts from θ₀, r=2..ceil(n/2) perturb best θ,
r>ceil(n/2) fully random in [lower, upper].
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
    ystd  = (y .- yμ) ./ yσ
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
        K  = buildK(X, ℓ, σf)
        σstd = c .* σstd0
        @inbounds for i in 1:n
            si = max(σstd[i], 1e-10)
            K[i, i] += si^2 + jitter
        end
        try return cholesky(Symmetric(K)) catch; return nothing end
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
        learn_noise_scale && (θ0[d+2] = log(1.0))
        θ0 .= clamp.(θ0, lower .+ _θ_ε, upper .- _θ_ε)
    end

    if !learn_hypers
        F = chol_from_θ(θ0)
        F === nothing && throw(ArgumentError("Cholesky failed with fixed θ; increase jitter."))
        ℓ  = exp.(θ0[1:d])
        σf = exp(θ0[d+1])
        c  = learn_noise_scale ? exp(θ0[d+2]) : 1.0
        L  = F.L
        α  = L' \ (L \ ystd)
        return HeteroGP(X, yμ, yσ, ℓ, σf, c, L, α, copy(θ0))
    end

    bestθ = copy(θ0)
    bestv = nlml(bestθ)
    opts  = Optim.Options(iterations=250, g_tol=1e-6, f_abstol=1e-9)
    n_perturb = max(0, ceil(Int, n_restarts / 2) - 1)

    for r in 1:n_restarts
        θstart = if r == 1
            copy(θ0)
        elseif r <= 1 + n_perturb
            clamp.(bestθ .+ 0.3 .* randn(rng, p_opt), lower .+ _θ_ε, upper .- _θ_ε)
        else
            θs = Vector{Float64}(undef, p_opt)
            @inbounds for i in 1:p_opt
                θs[i] = lower[i] + rand(rng) * (upper[i] - lower[i])
            end
            θs
        end

        res  = optimize(nlml, lower, upper, θstart, Fminbox(LBFGS()), opts; autodiff=:finite)
        θhat = Optim.minimizer(res)
        vhat = Optim.minimum(res)
        if isfinite(vhat) && vhat < bestv
            bestv  = vhat
            bestθ .= θhat
        end
    end

    F = chol_from_θ(bestθ)
    F === nothing && throw(ArgumentError("Cholesky failed at optimized θ; increase jitter."))
    ℓ  = exp.(bestθ[1:d])
    σf = exp(bestθ[d+1])
    c  = learn_noise_scale ? exp(bestθ[d+2]) : 1.0
    L  = F.L
    α  = L' \ (L \ ystd)
    return HeteroGP(X, yμ, yσ, ℓ, σf, c, L, α, copy(bestθ))
end

"""
    predict_latent(gp, x) -> (μ, s2)

Posterior mean and variance of the latent function f(x).
"""
function predict_latent(gp::HeteroGP, x::Vector{Float64})
    _, n = size(gp.X)
    k = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k[i] = matern32(x, view(gp.X, :, i), gp.ℓ, gp.σf)
    end
    μstd  = dot(k, gp.α)
    v     = gp.L \ k
    kxx   = matern32(x, x, gp.ℓ, gp.σf)
    s2std = max(kxx - dot(v, v), 0.0)
    return gp.yμ + gp.yσ * μstd, (gp.yσ^2) * s2std
end

# BO structures + utilities
struct HeteroBOResult
    X::Matrix{Float64}          # d × n_total  — all evaluated points
    y::Vector{Float64}          # n_total       — measured values (original space)
    σy::Vector{Float64}         # n_total       — noise estimates
    i_acq::Vector{Int}          # indices of init + UCB acquisition queries
    i_opt::Vector{Int}          # indices of GPR recommendation evaluations
    bounds::Vector{Tuple{Float64,Float64}}
    n_shots::Int
    n_init::Int
    n_iter::Int
    maximize::Bool
    x_last::Vector{Float64}     # final GP-recommended point
    y_last::Float64             # measured f(x_last)
    y_hat_last::Float64         # GP posterior mean at x_last (original space)
    n_iter_actual::Int
    ℓ_final::Vector{Float64}
    σf_final::Float64
    c_final::Float64
    total_shots::Int
end

@inline function _validate_bounds(bounds)
    isempty(bounds) && throw(ArgumentError("bounds must be non-empty"))
    for (i, (lo, hi)) in enumerate(bounds)
        (isfinite(lo) && isfinite(hi)) || throw(ArgumentError("bounds[$i] must be finite"))
        lo < hi || throw(ArgumentError("bounds[$i]: lo must be < hi"))
    end
end

@inline function _is_far_enough(x::AbstractVector{Float64}, X::Matrix{Float64}, n::Int;
                                 min_dist::Float64=1e-4)
    @inbounds for i in 1:n
        d2 = 0.0
        for j in eachindex(x)
            u = x[j] - X[j, i]; d2 += u * u
        end
        d2 < min_dist * min_dist && return false
    end
    return true
end

@inline function _call_f_raw(f, x::Vector{Float64}, n::Int)
    result = f(x, n)
    # PythonCall returns Py objects from Python callbacks — use pyconvert + 0-based indexing
    return pyconvert(Float64, result[0]), pyconvert(Float64, result[1])
end

@inline function _rand_in_box(rng::Random.AbstractRNG, lb::Vector{Float64}, ub::Vector{Float64})
    x = Vector{Float64}(undef, length(lb))
    @inbounds for j in eachindex(x)
        x[j] = rand(rng) * (ub[j] - lb[j]) + lb[j]
    end
    return x
end

@inline ucb_score(μ::Float64, s2::Float64, κ::Float64) = μ + κ * sqrt(max(s2, 0.0))

"""
    recommend_mean(gp, bounds; M, rng) -> (x_rec, m_rec, s_rec)

Find the point with highest GP posterior mean within `bounds` via random scan.
"""
function recommend_mean(gp::HeteroGP, bounds;
                        M::Int=20000,
                        rng::Random.AbstractRNG=Random.default_rng())
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    best_x  = _rand_in_box(rng, lb, ub)
    best_m, best_s2 = predict_latent(gp, best_x)

    for _ in 2:M
        x = _rand_in_box(rng, lb, ub)
        μ, s2 = predict_latent(gp, x)
        if μ > best_m
            best_m  = μ
            best_s2 = s2
            best_x  = x
        end
    end

    return best_x, best_m, sqrt(max(best_s2, 0.0))
end

# Main algorithm
"""
    bayesopt_ucb_threshold(f; bounds, n_shots, ...)

Heteroscedastic BO with GP-UCB acquisition (random scan) and Matérn 3/2 ARD kernel.

Stopping: at each iteration the GP posterior mean at the recommended point is
checked against `fidelity_threshold` (mu_one_check). If it exceeds the threshold
the run stops early without spending additional shots.
"""
function bayesopt_ucb_threshold(f;
                               bounds::Vector{Tuple{Float64,Float64}},
                               n_shots::Int=400,
                               n_init::Int=8,
                               n_iter::Int=30,
                               M_acq::Int=20000,
                               M_rec::Int=20000,
                               κ::Float64=2.0,
                               maximize::Bool=true,
                               hyper_every::Int=10,
                               rng::Random.AbstractRNG=Random.default_rng(),
                               seed=nothing,
                               verbose::Bool=false,
                               fidelity_threshold::Union{Nothing,Float64}=nothing,
                               learn_noise_scale::Bool=true,
                               n_restarts::Int=6)

    _validate_bounds(bounds)
    n_init ≥ 1  || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0  || throw(ArgumentError("n_iter must be ≥ 0"))
    κ ≥ 0       || throw(ArgumentError("κ must be ≥ 0"))
    n_shots ≥ 1 || throw(ArgumentError("n_shots must be ≥ 1"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]
    d  = length(bounds)

    # n_init + one UCB + one opt-check per iteration + 1 final recommendation
    n_cap = n_init + 2 * n_iter + 1
    X  = Matrix{Float64}(undef, d, n_cap)
    y  = Vector{Float64}(undef, n_cap)   # stored sign-flipped internally; un-flipped on return
    σy = Vector{Float64}(undef, n_cap)
    write_idx         = 0
    total_shots_count = 0
    i_acq_list = Int[]
    i_opt_list  = Int[]

    # init: random exploration
    for _ in 1:n_init
        x = _rand_in_box(rng_local, lb, ub)
        y_raw, σy_i = _call_f_raw(f, x, n_shots)
        total_shots_count += n_shots
        write_idx += 1
        X[:, write_idx] = x
        y[write_idx]    = maximize ? y_raw : -y_raw
        σy[write_idx]   = σy_i
        push!(i_acq_list, write_idx)
    end

    θ_prev        = nothing
    n_iter_actual = n_iter

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

        # UCB acquisition
        best_x = _rand_in_box(rng_local, lb, ub)
        best_μ, best_s2 = predict_latent(gp, best_x)
        best_a = ucb_score(best_μ, best_s2, κ)
        for _ in 2:M_acq
            xc = _rand_in_box(rng_local, lb, ub)
            μ, s2 = predict_latent(gp, xc)
            a = ucb_score(μ, s2, κ)
            if a > best_a
                best_a = a
                best_x = xc
            end
        end

        y_raw, σy_i = _call_f_raw(f, best_x, n_shots)
        total_shots_count += n_shots
        if _is_far_enough(best_x, X, write_idx)
            write_idx += 1
            X[:, write_idx] = best_x
            y[write_idx]    = maximize ? y_raw : -y_raw
            σy[write_idx]   = σy_i
            push!(i_acq_list, write_idx)
        end
        verbose && @info "it=$it best_acq=$best_a"

        # mu_one_check: GPR mean ≥ threshold → confirm with a real sample
        if fidelity_threshold !== nothing
            x_opt, m_opt, _ = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
            if m_opt >= fidelity_threshold
                y_opt, σy_opt = _call_f_raw(f, x_opt, n_shots)
                total_shots_count += n_shots
                if _is_far_enough(x_opt, X, write_idx)
                    write_idx += 1
                    X[:, write_idx] = x_opt
                    y[write_idx]    = maximize ? y_opt : -y_opt
                    σy[write_idx]   = σy_opt
                    push!(i_opt_list, write_idx)
                end
                y_hat_last = maximize ? m_opt : -m_opt
                y_last     = y_opt
                if y_last >= fidelity_threshold
                    n_iter_actual = it
                    y_out = maximize ? y[1:write_idx] : -y[1:write_idx]
                    return HeteroBOResult(X[:, 1:write_idx], y_out, σy[1:write_idx],
                                          i_acq_list, i_opt_list,
                                          bounds, n_shots, n_init, n_iter, maximize,
                                          x_opt, y_last, y_hat_last, n_iter_actual,
                                          gp.ℓ, gp.σf, gp.c, total_shots_count)
                end
            end
        end
    end

    # final GPR recommendation
    gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                      θ_init=θ_prev,
                      learn_hypers=true,
                      learn_noise_scale=learn_noise_scale,
                      n_restarts=n_restarts + 2,
                      jitter=1e-8,
                      rng=rng_local)

    x_last, m_last, _ = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
    y_last, σy_last   = _call_f_raw(f, x_last, n_shots)
    total_shots_count += n_shots
    if _is_far_enough(x_last, X, write_idx)
        write_idx += 1
        X[:, write_idx] = x_last
        y[write_idx]    = maximize ? y_last : -y_last
        σy[write_idx]   = σy_last
        push!(i_opt_list, write_idx)
    end

    y_hat_last = maximize ? m_last : -m_last
    y_out      = maximize ? y[1:write_idx] : -y[1:write_idx]

    return HeteroBOResult(X[:, 1:write_idx], y_out, σy[1:write_idx],
                          i_acq_list, i_opt_list,
                          bounds, n_shots, n_init, n_iter, maximize,
                          x_last, y_last, y_hat_last, n_iter_actual,
                          gp.ℓ, gp.σf, gp.c, total_shots_count)
end
