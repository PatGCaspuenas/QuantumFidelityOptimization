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

# Analytical ∇ₓ of Matérn 3/2 kernel: ∂k/∂x_j = -3σf² exp(-√3 r)(x_j-z_j)/ℓ_j²
function matern32_grad_x(x::AbstractVector, z::AbstractVector,
                          ℓ::AbstractVector, σf::Float64)
    r2 = 0.0
    @inbounds for j in eachindex(ℓ)
        u = (x[j] - z[j]) / ℓ[j]
        r2 += u*u
    end
    r = sqrt(r2)
    a = sqrt(3.0) * r
    g = Vector{Float64}(undef, length(x))
    if r < 1e-12
        fill!(g, 0.0)
        return g
    end
    coeff = -3.0 * (σf^2) * exp(-a)
    @inbounds for j in eachindex(ℓ)
        g[j] = coeff * (x[j] - z[j]) / (ℓ[j]^2)
    end
    return g
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
Restart strategy: r=1 warm-starts from θ₀, r=2..ceil(n/2) perturb best θ so far
(σ=0.3 in log-space), r>ceil(n/2) fully random in [lower, upper].
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

    n_perturb = max(0, ceil(Int, n_restarts / 2) - 1)  # restarts 2..ceil(n/2) perturb best

    for r in 1:n_restarts
        if r == 1
            θstart = copy(θ0)
        elseif r <= 1 + n_perturb
            # Perturb the current best with small Gaussian noise in log-space
            θstart = clamp.(bestθ .+ 0.3 .* randn(rng, p_opt), lower .+ _θ_ε, upper .- _θ_ε)
        else
            # Fully random in [lower, upper]
            θstart = Vector{Float64}(undef, p_opt)
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

"""
    predict_latent_grad(gp, x) -> (μ, s2, ∇μ, ∇s2)

Posterior mean, variance, and their analytical gradients w.r.t. x.

∇μ(x)  = yσ · ∇K(x,X) α
∇s²(x) = yσ² · (-2 · vᵀ · L⁻¹∇K)   where ∂kxx/∂x = 0 (σf is constant w.r.t. x)
"""
function predict_latent_grad(gp::HeteroGP, x::Vector{Float64})
    X = gp.X
    d, n = size(X)

    k  = Vector{Float64}(undef, n)
    ∇K = Matrix{Float64}(undef, d, n)
    @inbounds for i in 1:n
        xi = view(X, :, i)
        k[i]    = matern32(x, xi, gp.ℓ, gp.σf)
        ∇K[:,i] = matern32_grad_x(x, xi, gp.ℓ, gp.σf)
    end

    μstd  = dot(k, gp.α)
    v     = gp.L \ k
    kxx   = matern32(x, x, gp.ℓ, gp.σf)
    s2std = max(kxx - dot(v, v), 0.0)

    μ  = gp.yμ + gp.yσ * μstd
    s2 = (gp.yσ^2) * s2std

    ∇μ  = gp.yσ .* (∇K * gp.α)
    Lv  = gp.L \ ∇K   # d×n
    ∇s2 = (gp.yσ^2) .* (-2.0 .* (Lv * v))

    return μ, s2, ∇μ, ∇s2
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

@inline ucb_score(μ::Float64, s2::Float64, κ::Float64) = μ + κ * sqrt(max(s2, 0.0))


"""
    _topk_separated(candidates, scores, k, min_sep)

Greedy maximin: pick the top-scoring candidate, then repeatedly add the
candidate with highest score that is at least `min_sep` (Euclidean) from all
already-selected candidates. Returns indices into `candidates`.
"""
function _topk_separated(candidates::Vector{Vector{Float64}},
                          scores::Vector{Float64},
                          k::Int,
                          min_sep::Float64)
    n = length(candidates)
    k = min(k, n)
    selected = Int[]
    sizehint!(selected, k)

    order = sortperm(scores; rev=true)

    for idx in order
        if isempty(selected)
            push!(selected, idx)
        else
            far = true
            xi = candidates[idx]
            for s in selected
                xs = candidates[s]
                d2 = 0.0
                for j in eachindex(xi)
                    u = xi[j] - xs[j]
                    d2 += u*u
                end
                if d2 < min_sep^2
                    far = false
                    break
                end
            end
            far && push!(selected, idx)
        end
        length(selected) == k && break
    end
    return selected
end

"""
    _acq_lbfgs(gp, x0, lb, ub, κ; use_grad, n_iters) -> x_opt

Maximize UCB from x0 using bounded L-BFGS.
`use_grad=true` uses analytical GP posterior gradients; `false` uses finite differences.
"""
function _acq_lbfgs(gp::HeteroGP, x0::Vector{Float64},
                    lb::Vector{Float64}, ub::Vector{Float64},
                    κ::Float64; use_grad::Bool=false, n_iters::Int=100)
    if use_grad
        function fg!(F, G, x)
            μ, s2, ∇μ, ∇s2 = predict_latent_grad(gp, x)
            s = sqrt(max(s2, 0.0))
            if G !== nothing
                @. G = -(∇μ + (s > 1e-12 ? κ * ∇s2 / (2 * s) : zero(∇μ)))
            end
            F !== nothing && return -(μ + κ * s)
            return nothing
        end
        res = optimize(Optim.only_fg!(fg!), lb, ub, copy(x0),
                       Fminbox(LBFGS()),
                       Optim.Options(iterations=n_iters, g_tol=1e-5, f_abstol=1e-10))
    else
        res = optimize(x -> begin μ, s2 = predict_latent(gp, x); -(μ + κ * sqrt(max(s2, 0.0))) end,
                       lb, ub, copy(x0),
                       Fminbox(LBFGS()),
                       Optim.Options(iterations=n_iters, g_tol=1e-5, f_abstol=1e-10);
                       autodiff=:finite)
    end
    return Optim.minimizer(res)
end


"""
    _acquire_topk(gp, lb, ub, κ; ...) -> (best_x, best_ucb)

Acquisition maximization with top-k separated multi-start strategy:

1. Sample `M_acq` random candidates; evaluate UCB on all.
2. Greedily pick `k_acq` well-separated starts (greedy maximin, spacing ≥ `min_sep`).
3. For each start:
   - If `use_zoom`: sample `M_zoom` points in an ℓ∞ ball of radius `zoom_radius` (scaled)
     around the candidate; keep the best.
   - If `use_lbfgs_acq`: refine from the current best point via bounded L-BFGS.
     `use_grad_acq` controls analytical vs finite-difference gradients.
4. Return the overall best across all starts.

With `k_acq=1`, `use_zoom=false`, `use_lbfgs_acq=false` this reduces to plain random scan
(original baseline behavior).
"""
function _acquire_topk(gp::HeteroGP,
                        lb::Vector{Float64}, ub::Vector{Float64},
                        κ::Float64;
                        M_acq::Int=5000,
                        k_acq::Int=1,
                        min_sep::Float64=0.05,
                        use_zoom::Bool=false,
                        M_zoom::Int=200,
                        zoom_radius::Float64=0.1,
                        use_lbfgs_acq::Bool=false,
                        use_grad_acq::Bool=false,
                        rng::Random.AbstractRNG=Random.default_rng())

    # Step 1: random global candidates
    cands  = [_rand_in_box(rng, lb, ub) for _ in 1:M_acq]
    scores = Vector{Float64}(undef, M_acq)
    for i in 1:M_acq
        μ, s2 = predict_latent(gp, cands[i])
        scores[i] = ucb_score(μ, s2, κ)
    end

    # Step 2: select top-k separated starting points
    sel = _topk_separated(cands, scores, min(k_acq, M_acq), min_sep)

    best_x   = cands[sel[1]]
    best_ucb = scores[sel[1]]

    # Step 3+4: zoom and/or L-BFGS from each selected start
    for idx in sel
        x0 = cands[idx]
        a0 = scores[idx]

        if use_zoom
            zoom_best_x   = x0
            zoom_best_ucb = a0
            width = ub .- lb
            for _ in 1:M_zoom
                xz = clamp.(x0 .+ zoom_radius .* (2 .* rand(rng, length(lb)) .- 1) .* width, lb, ub)
                μz, s2z = predict_latent(gp, xz)
                az = ucb_score(μz, s2z, κ)
                if az > zoom_best_ucb
                    zoom_best_ucb = az
                    zoom_best_x   = xz
                end
            end
            x0 = zoom_best_x
            a0 = zoom_best_ucb
        end

        if use_lbfgs_acq
            x_opt = try
                _acq_lbfgs(gp, x0, lb, ub, κ; use_grad=use_grad_acq)
            catch
                x0
            end
            μ_opt, s2_opt = predict_latent(gp, x_opt)
            a0 = ucb_score(μ_opt, s2_opt, κ)
            x0 = x_opt
        end

        if a0 > best_ucb
            best_ucb = a0
            best_x   = x0
        end
    end

    return best_x, best_ucb
end


"""
    recommend_mean(gp, bounds; M, rng) -> (x_rec, m_rec, s_rec)

Find the point with highest GP posterior mean within `bounds` via global random
search (M candidates) followed by one bounded L-BFGS refinement.
"""
function recommend_mean(gp::HeteroGP, bounds; M::Int=20000, rng::Random.AbstractRNG=Random.default_rng())
    M ≥ 1 || throw(ArgumentError("M must be ≥ 1"))
    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

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
    end

    return best_x, best_m, best_s
end

# -------------------------
# Variable-N helper
# -------------------------

function _adaptive_measure(f, x::Vector{Float64},
                           n_floor::Int, n_max::Int,
                           threshold::Float64,
                           maximize::Bool,
                           batch_size::Int=50)
    y1, _ = _call_f_raw(f, x, n_floor)
    Q_cur   = clamp(y1, 0.0, 1.0)
    k_total = Q_cur * Float64(n_floor)
    N_total = n_floor

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

Acquisition is maximized via `_acquire_topk` which supports:
  - `k_acq`         — number of separated multi-start candidates (default 1)
  - `min_sep`       — minimum Euclidean separation between starts (default 0.05)
  - `use_zoom`      — random search in ℓ∞ ball around each candidate (default false)
  - `M_zoom`        — zoom sample count per candidate (default 200)
  - `zoom_radius`   — zoom ball half-width as fraction of box width (default 0.1)
  - `use_lbfgs_acq` — L-BFGS refinement from each start (default false = pure random scan)
  - `use_grad_acq`  — use analytical Matérn 3/2 gradients in L-BFGS (default false)

`use_variable_mode`:
  - `false` — fixed N_shots per acquisition.
  - `true`  — adaptive shots at x_rec; early stop when confirmed above threshold.
              Requires `fidelity_threshold`.

Set `maximize=false` to minimize. Use `seed` for reproducibility.
"""
function bayesopt_ucb_threshold(f;
                               bounds::Vector{Tuple{Float64,Float64}},
                               n_shots::Int=400,
                               n_init::Int=8,
                               n_iter::Int=30,
                               M_acq::Int=20000,
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
                               learn_noise_scale::Bool=true,
                               n_restarts::Int=6,
                               use_variable_mode::Bool=false,
                               n_floor::Int=50,
                               n_max_shots::Int=2000,
                               # Acquisition options
                               k_acq::Int=1,
                               min_sep::Float64=0.05,
                               use_zoom::Bool=false,
                               M_zoom::Int=200,
                               zoom_radius::Float64=0.1,
                               use_lbfgs_acq::Bool=false,
                               use_grad_acq::Bool=false)

    _validate_bounds(bounds)
    n_init ≥ 1 || throw(ArgumentError("n_init must be ≥ 1"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be ≥ 0"))
    M_acq ≥ 1  || throw(ArgumentError("M_acq must be ≥ 1"))
    M_rec ≥ 1  || throw(ArgumentError("M_rec must be ≥ 1"))
    κ ≥ 0      || throw(ArgumentError("κ must be ≥ 0"))
    α ≥ 0      || throw(ArgumentError("α must be ≥ 0"))
    n_shots ≥ 1   || throw(ArgumentError("n_shots must be ≥ 1"))
    n_floor ≥ 1 || throw(ArgumentError("n_floor must be ≥ 1"))
    n_max_shots ≥ n_floor || throw(ArgumentError("n_max_shots must be ≥ n_floor"))
    k_acq ≥ 1 || throw(ArgumentError("k_acq must be ≥ 1"))
    (use_variable_mode && fidelity_threshold === nothing) &&
        throw(ArgumentError("fidelity_threshold required for use_variable_mode=true"))

    rng_local = seed === nothing ? rng : MersenneTwister(seed)

    lb = Float64[b[1] for b in bounds]
    ub = Float64[b[2] for b in bounds]

    d = length(bounds)
    n_cap = n_init + 3 * n_iter
    X  = Matrix{Float64}(undef, d, n_cap)
    y  = Vector{Float64}(undef, n_cap)
    σy = Vector{Float64}(undef, n_cap)
    write_idx         = 0
    total_shots_count = 0

    for i in 1:n_init
        x = _rand_in_box(rng_local, lb, ub)
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

        best_x, best_a = _acquire_topk(gp, lb, ub, κ;
                                        M_acq=M_acq,
                                        k_acq=k_acq,
                                        min_sep=min_sep,
                                        use_zoom=use_zoom,
                                        M_zoom=M_zoom,
                                        zoom_radius=zoom_radius,
                                        use_lbfgs_acq=use_lbfgs_acq,
                                        use_grad_acq=use_grad_acq,
                                        rng=rng_local)

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

        if use_variable_mode
            x_rec_cur, _, _ = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
            y_rec_cur, σy_rec_cur, n_rec, stop_loop = _adaptive_measure(
                f, x_rec_cur, n_floor, n_max_shots, fidelity_threshold, maximize)
            total_shots_count += n_rec

            if _is_far_enough(x_rec_cur, X, write_idx)
                write_idx += 1
                X[:, write_idx] = x_rec_cur
                y[write_idx]  = maximize ? y_rec_cur : -y_rec_cur
                σy[write_idx] = σy_rec_cur
            end

            if stop_loop
                n_iter_actual = it
                y_out = maximize ? y[1:write_idx] : -y[1:write_idx]
                return HeteroBOResult(X[:, 1:write_idx], y_out, σy[1:write_idx],
                                      bounds, n_shots, n_init, n_iter, maximize,
                                      x_rec_cur, y_rec_cur, n_iter_actual,
                                      maximize ? y_last_val : -y_last_val,
                                      gp.ℓ, gp.σf, gp.c, total_shots_count)
            end
        end

        if fidelity_threshold !== nothing && !use_variable_mode
            x_rec_cur, _, _ = recommend_mean(gp, bounds; M=M_rec, rng=rng_local)
            y_rec_cur, σy1_i = _call_f_raw(f, x_rec_cur, n_shots)
            total_shots_count += n_shots
            if _is_far_enough(x_rec_cur, X, write_idx)
                write_idx += 1
                X[:, write_idx] = x_rec_cur
                y[write_idx]  = maximize ? y_rec_cur : -y_rec_cur
                σy[write_idx] = σy1_i
            end

            y1 = maximize ? y_rec_cur : -y_rec_cur
            reached1 = maximize ? (y1 >= fidelity_threshold) : (y1 <= fidelity_threshold)
            if reached1
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

    gp = fit_heterogp(X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                      θ_init=θ_prev,
                      learn_hypers=true,
                      learn_noise_scale=learn_noise_scale,
                      n_restarts=n_restarts + 2,
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
