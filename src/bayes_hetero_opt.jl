# src/bayes_hetero_opt.jl
# Heteroscedastic GP + GP-UCB acquisition with noise-level thresholding

using Random
using Statistics
using LinearAlgebra
using Optim
using Dates

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

struct PretrainedHeteroGPState
    θ::Vector{Float64}
    yμ::Float64
    yσ::Float64
    bounds::Vector{Tuple{Float64,Float64}}
    σ_levels::Vector{Float64}
    learn_noise_scale::Bool
    ℓ_bounds::Tuple{Float64,Float64}
    σf_bounds::Tuple{Float64,Float64}
    c_bounds::Tuple{Float64,Float64}
    objective_mode::Symbol
    numMS::Int
    seed::Union{Nothing,Int}
    trained_at::String
end

@inline function _validate_scaling_stats(yμ::Float64, yσ::Float64)
    isfinite(yμ) || throw(ArgumentError("yμ must be finite"))
    (isfinite(yσ) && yσ > 0) || throw(ArgumentError("yσ must be finite and > 0"))
    return nothing
end

@inline function _state_to_namedtuple(s::PretrainedHeteroGPState)
    return (
        θ=copy(s.θ),
        yμ=s.yμ,
        yσ=s.yσ,
        bounds=copy(s.bounds),
        σ_levels=copy(s.σ_levels),
        learn_noise_scale=s.learn_noise_scale,
        ℓ_bounds=s.ℓ_bounds,
        σf_bounds=s.σf_bounds,
        c_bounds=s.c_bounds,
        objective_mode=s.objective_mode,
        numMS=s.numMS,
        seed=s.seed,
        trained_at=s.trained_at,
    )
end

function pretrained_state_from_namedtuple(nt)
    hasproperty(nt, :θ) || throw(ArgumentError("Missing field θ in pretrained state"))
    hasproperty(nt, :yμ) || throw(ArgumentError("Missing field yμ in pretrained state"))
    hasproperty(nt, :yσ) || throw(ArgumentError("Missing field yσ in pretrained state"))
    hasproperty(nt, :bounds) || throw(ArgumentError("Missing field bounds in pretrained state"))
    hasproperty(nt, :σ_levels) || throw(ArgumentError("Missing field σ_levels in pretrained state"))

    state = PretrainedHeteroGPState(
        Vector{Float64}(nt.θ),
        Float64(nt.yμ),
        Float64(nt.yσ),
        Vector{Tuple{Float64,Float64}}(nt.bounds),
        Vector{Float64}(nt.σ_levels),
        hasproperty(nt, :learn_noise_scale) ? Bool(nt.learn_noise_scale) : true,
        hasproperty(nt, :ℓ_bounds) ? Tuple{Float64,Float64}(nt.ℓ_bounds) : (0.05, 1.5),
        hasproperty(nt, :σf_bounds) ? Tuple{Float64,Float64}(nt.σf_bounds) : (0.3, 2.0),
        hasproperty(nt, :c_bounds) ? Tuple{Float64,Float64}(nt.c_bounds) : (0.3, 3.0),
        hasproperty(nt, :objective_mode) ? Symbol(nt.objective_mode) : :unknown,
        hasproperty(nt, :numMS) ? Int(nt.numMS) : 0,
        hasproperty(nt, :seed) ? (nt.seed === nothing ? nothing : Int(nt.seed)) : nothing,
        hasproperty(nt, :trained_at) ? String(nt.trained_at) : string(Dates.now()),
    )

    _validate_scaling_stats(state.yμ, state.yσ)
    _validate_bounds(state.bounds)
    isempty(state.σ_levels) && throw(ArgumentError("σ_levels must be non-empty"))
    return state
end

function build_pretrained_state(gp::HeteroGP;
                                bounds::Vector{Tuple{Float64,Float64}},
                                σ_levels::Vector{Float64},
                                learn_noise_scale::Bool=true,
                                ℓ_bounds::Tuple{Float64,Float64}=(0.05, 1.5),
                                σf_bounds::Tuple{Float64,Float64}=(0.3, 2.0),
                                c_bounds::Tuple{Float64,Float64}=(0.3, 3.0),
                                objective_mode::Symbol=:deterministic,
                                numMS::Int=0,
                                seed::Union{Nothing,Int}=nothing,
                                trained_at::String=string(Dates.now()))
    _validate_bounds(bounds)
    isempty(σ_levels) && throw(ArgumentError("σ_levels must be non-empty"))
    _validate_scaling_stats(gp.yμ, gp.yσ)
    return PretrainedHeteroGPState(
        copy(gp.θ),
        gp.yμ,
        gp.yσ,
        copy(bounds),
        copy(σ_levels),
        learn_noise_scale,
        ℓ_bounds,
        σf_bounds,
        c_bounds,
        objective_mode,
        numMS,
        seed,
        trained_at,
    )
end

function save_pretrained_state_script(output_path::AbstractString,
                                      state::PretrainedHeteroGPState;
                                      loader_name::AbstractString="load_pretrained_heterogp_state")
    open(output_path, "w") do io
        println(io, "# Auto-generated pretrained heteroscedastic GP state")
        println(io, "# Generated at ", state.trained_at)
        println(io)
        println(io, "function ", loader_name, "()")
        println(io, "    return (")
        println(io, "        θ=", repr(state.θ), ",")
        println(io, "        yμ=", repr(state.yμ), ",")
        println(io, "        yσ=", repr(state.yσ), ",")
        println(io, "        bounds=", repr(state.bounds), ",")
        println(io, "        σ_levels=", repr(state.σ_levels), ",")
        println(io, "        learn_noise_scale=", repr(state.learn_noise_scale), ",")
        println(io, "        ℓ_bounds=", repr(state.ℓ_bounds), ",")
        println(io, "        σf_bounds=", repr(state.σf_bounds), ",")
        println(io, "        c_bounds=", repr(state.c_bounds), ",")
        println(io, "        objective_mode=", repr(state.objective_mode), ",")
        println(io, "        numMS=", repr(state.numMS), ",")
        println(io, "        seed=", repr(state.seed), ",")
        println(io, "        trained_at=", repr(state.trained_at))
        println(io, "    )")
        println(io, "end")
    end
    return output_path
end

function load_pretrained_state_script(path::AbstractString;
                                      loader_symbol::Symbol=:load_pretrained_heterogp_state)
    m = Module()
    Base.include(m, path)
    isdefined(m, loader_symbol) || throw(ArgumentError("Loader $(loader_symbol) not found in $(path)"))
    loader = getfield(m, loader_symbol)
    # Avoid world-age errors when calling methods defined by dynamic include.
    nt = Base.invokelatest(loader)
    return pretrained_state_from_namedtuple(nt)
end

function pretrain_heterogp_deterministic(f_det;
                                         bounds::Vector{Tuple{Float64,Float64}},
                                         σ_levels::Union{Nothing,Vector{Float64}}=nothing,
                                         deployment_σ_levels::Union{Nothing,Vector{Float64}}=σ_levels,
                                         n_samples::Int=256,
                                         n_passes::Int=1,
                                         σ_train::Float64=0.0,
                                         learn_noise_scale::Bool=true,
                                         n_restarts::Int=8,
                                         ℓ_bounds::Tuple{Float64,Float64}=(0.05, 1.5),
                                         σf_bounds::Tuple{Float64,Float64}=(0.3, 2.0),
                                         c_bounds::Tuple{Float64,Float64}=(0.3, 3.0),
                                         rng::Random.AbstractRNG=Random.default_rng(),
                                         seed=nothing,
                                         numMS::Int=0,
                                         output_script_path::Union{Nothing,AbstractString}=nothing)
    _validate_bounds(bounds)
    isfinite(σ_train) || throw(ArgumentError("σ_train must be finite"))
    σ_train ≥ 0 || throw(ArgumentError("σ_train must be ≥ 0"))
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))
    n_passes ≥ 1 || throw(ArgumentError("n_passes must be ≥ 1"))

    state_σ_levels = deployment_σ_levels === nothing ? Float64[0.0] : copy(deployment_σ_levels)
    isempty(state_σ_levels) && throw(ArgumentError("deployment_σ_levels must be non-empty when provided"))
    all(s -> isfinite(s) && s ≥ 0, state_σ_levels) || throw(ArgumentError("deployment_σ_levels must be finite and ≥ 0"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]
    d = length(bounds)

    n_total = n_samples * n_passes
    X = Matrix{Float64}(undef, d, n_total)
    y = Vector{Float64}(undef, n_total)
    σy = fill(σ_train, n_total)

    θ_prev = nothing
    gp = nothing

    for pass in 1:n_passes
        i0 = (pass - 1) * n_samples + 1
        i1 = pass * n_samples

        for i in i0:i1
            x = _rand_in_box(rng_local, lb, ub)
            X[:, i] = x
            y[i] = f_det(x)
        end

        gp = fit_heterogp(X[:, 1:i1], y[1:i1], σy[1:i1];
                          jitter=1e-8,
                          learn_hypers=true,
                          learn_noise_scale=learn_noise_scale,
                          n_restarts=n_restarts,
                          θ_init=θ_prev,
                          ℓ_bounds=ℓ_bounds,
                          σf_bounds=σf_bounds,
                          c_bounds=c_bounds,
                          rng=rng_local)

        θ_prev = gp.θ
    end

    gp === nothing && throw(ArgumentError("Pretraining failed to produce a GP model"))

    state = build_pretrained_state(gp;
                                   bounds=bounds,
                                   σ_levels=state_σ_levels,
                                   learn_noise_scale=learn_noise_scale,
                                   ℓ_bounds=ℓ_bounds,
                                   σf_bounds=σf_bounds,
                                   c_bounds=c_bounds,
                                   objective_mode=:deterministic,
                                   numMS=numMS,
                                   seed=(seed === nothing ? nothing : Int(seed)))

    if output_script_path !== nothing
        save_pretrained_state_script(output_script_path, state)
    end

    return (gp=gp, state=state, n_points=n_total, n_passes=n_passes, n_samples_per_pass=n_samples)
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
                      y_stats::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                      θ_init::Union{Nothing,Vector{Float64}}=nothing,
                      ℓ_bounds::Tuple{Float64,Float64}=(0.05, 1.5),
                      σf_bounds::Tuple{Float64,Float64}=(0.3, 2.0),
                      c_bounds::Tuple{Float64,Float64}=(0.3, 3.0),
                      rng::Random.AbstractRNG=Random.default_rng())

    _validate_gp_inputs(X, y, σy)
    jitter > 0 || throw(ArgumentError("jitter must be > 0"))

    if y_stats === nothing
        yμ = mean(y)
        yσ = max(std(y), 1e-12)
    else
        yμ, yσ = y_stats
        _validate_scaling_stats(yμ, yσ)
    end
    ystd = (y .- yμ) ./ yσ
    σstd0 = (σy ./ yσ)

    d, n = size(X)
    p = learn_noise_scale ? (d + 2) : (d + 1)

    lower = Vector{Float64}(undef, p)
    upper = Vector{Float64}(undef, p)
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
        # solve (K+Σ)^{-1} y
        αtmp = F \ ystd
        return 0.5 * dot(ystd, αtmp) + sum(log, diag(F.L)) + 0.5 * n * log(2π)
    end

    # starting point
    θ0 = Vector{Float64}(undef, p)
    if θ_init !== nothing && length(θ_init) == p
        θ0 .= clamp.(θ_init, lower, upper)
    else
        θ0[1:d] .= log(0.3)
        θ0[d+1]  = log(1.0)
        if learn_noise_scale
            θ0[d+2] = log(1.0)
        end
        θ0 .= clamp.(θ0, lower, upper)
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
            @inbounds for i in 1:p
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

mutable struct HeteroBOResult
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
    n_iter_actual::Int                    # actual iterations if early stopped
    y_last::Float64                       # latest observed value in original objective scale
end

# Constructor with default value for n_iter_actual
function HeteroBOResult(X, y, σy, bounds, σ_levels, n_init, n_iter, maximize, x_rec, y_rec)
    return HeteroBOResult(X, y, σy, bounds, σ_levels, n_init, n_iter, maximize, x_rec, y_rec, 0, NaN)
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
Set `fidelity_threshold` to stop early when the latest measurement reaches this threshold
(compared in the original objective scale and respecting `maximize`).
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
                               freeze_theta::Bool=false,
                               pretrained_state::Union{Nothing,PretrainedHeteroGPState}=nothing,
                               use_pretrained_scaling::Bool=true,
                               learn_noise_scale::Bool=true,
                               ℓ_bounds::Tuple{Float64,Float64}=(0.05, 1.5),
                               σf_bounds::Tuple{Float64,Float64}=(0.3, 2.0),
                               c_bounds::Tuple{Float64,Float64}=(0.3, 3.0),
                               rng::Random.AbstractRNG=Random.default_rng(),
                               seed=nothing,
                               verbose::Bool=false,
                               fidelity_threshold=nothing)

    _validate_bounds(bounds)
    isempty(σ_levels) && throw(ArgumentError("σ_levels must be non-empty"))
    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M_acq ≥ 1  || throw(ArgumentError("M_acq must be ≥ 1"))
    M_rec ≥ 1  || throw(ArgumentError("M_rec must be ≥ 1"))
    κ ≥ 0      || throw(ArgumentError("κ must be ≥ 0"))
    α ≥ 0      || throw(ArgumentError("α must be ≥ 0"))

    if pretrained_state !== nothing
        length(pretrained_state.bounds) == length(bounds) || throw(ArgumentError("Pretrained bounds dimension mismatch"))
        pretrained_state.bounds == bounds || throw(ArgumentError("Pretrained bounds do not match runtime bounds"))
        pretrained_state.σ_levels == σ_levels || throw(ArgumentError("Pretrained σ_levels do not match runtime σ_levels"))
        if freeze_theta && length(pretrained_state.θ) != (length(bounds) + (learn_noise_scale ? 2 : 1))
            throw(ArgumentError("Pretrained θ length does not match current model configuration"))
        end
    end

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

    θ_prev = pretrained_state === nothing ? nothing : copy(pretrained_state.θ)
    scale_stats = (pretrained_state !== nothing && use_pretrained_scaling) ? (pretrained_state.yμ, pretrained_state.yσ) : nothing

    for it in 1:n_iter
        idx = n_init + it
        do_opt = (it == 1) || (hyper_every > 0 && it % hyper_every == 0)
        if freeze_theta
            do_opt = false
        end

        gp = fit_heterogp(X[:, 1:(idx-1)], y[1:(idx-1)], σy[1:(idx-1)];
                          θ_init=θ_prev,
                          learn_hypers=do_opt,
                          learn_noise_scale=learn_noise_scale,
                          n_restarts=do_opt ? 6 : 0,
                          y_stats=scale_stats,
                          ℓ_bounds=ℓ_bounds,
                          σf_bounds=σf_bounds,
                          c_bounds=c_bounds,
                          jitter=1e-8,
                          rng=rng_local)

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

        X[:, idx] = best_x
        σy[idx] = σ_next
        y_latest_eval = f_eval(best_x, σ_next)
        y[idx] = y_latest_eval

        if verbose
            @info "it=$it best_acq=$best_a σ=$σ_next"
        end
        
        # Check fidelity threshold using f_eval at x_rec (not the latest sampled point)
        # Exclude y_rec_raw == 1.0 from triggering early stop (likely noisy artifact)
        if fidelity_threshold !== nothing
            x_rec, m_rec = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
            y_rec_eval = f_eval(x_rec, σ_next)
            y_rec_raw = maximize ? y_rec_eval : -y_rec_eval
            reached = maximize ? (y_rec_raw >= fidelity_threshold) : (y_rec_raw <= fidelity_threshold)
            if reached && y_rec_raw != 1.0
                # Trim arrays to actual size and record iterations
                X = X[:, 1:idx]
                y = y[1:idx]
                σy = σy[1:idx]
                n_iter_actual = it
                if verbose
                    @info "Fidelity threshold $fidelity_threshold reached at iteration $it"
                end

                y_out = maximize ? y : (-y)
                y_rec = y_rec_raw
                result = HeteroBOResult(X, y_out, σy, bounds, σ_levels, n_init, n_iter_actual, maximize, x_rec, y_rec)
                result.n_iter_actual = n_iter_actual
                result.y_last = y_rec_raw
                return result
            end
        end
    end

    gp = fit_heterogp(X, y, σy;
                      θ_init=θ_prev,
                      learn_hypers=!freeze_theta,
                      learn_noise_scale=learn_noise_scale,
                      n_restarts=freeze_theta ? 0 : 8,
                      y_stats=scale_stats,
                      ℓ_bounds=ℓ_bounds,
                      σf_bounds=σf_bounds,
                      c_bounds=c_bounds,
                      jitter=1e-8,
                      rng=rng_local)

    x_rec, m_rec = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)

    y_out = maximize ? y : (-y)
    y_rec_eval = f_eval(x_rec, σy[end])
    y_rec = maximize ? y_rec_eval : -y_rec_eval

    y_last_eval = y_rec_eval
    y_last_raw = maximize ? y_last_eval : -y_last_eval
    result = HeteroBOResult(X, y_out, σy, bounds, σ_levels, n_init, n_iter, maximize, x_rec, y_rec)
    result.n_iter_actual = n_iter
    result.y_last = y_last_raw
    return result
end
