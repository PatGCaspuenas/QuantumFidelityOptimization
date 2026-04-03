import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Statistics
using Distributed

# CONFIG
const _t              = 100.0
const _bounds         = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
const _n_pretrain     = 300
const _n_restarts     = 10
const _N              = 1000  # shots for Q_varMS; set to 0 for noiseless Q_det
const _optimize_det   = false  # if false, pretrain on Q_varMS instead of Q_det
const _log_fid        = false   # whether to apply log transform to Q for better GP fitting
const _lns            = !_optimize_det  # learn_noise_scale
# :simple  → σy = 1/sqrt(N)  (worst-case binomial, ignores Q value)
# :binomial → σy = sqrt(Q*(1-Q)/N) in Q-space, or delta-method propagated
#            through log10(1-Q) when _log_fid=true
const _sigma_mode     = :simple

const _filename_output = "pretrained_theta_3d_QvarMS_n300_rest10_N1000_simple.jl"
const _num_sims        = 1

if nprocs() == 1
    addprocs()
end

try

    @everywhere begin
        import Pkg
        Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
        include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
        using Random
    end

    # Broadcast config to all workers
    @everywhere const _t            = $(_t)
    @everywhere const _n_pretrain   = $(_n_pretrain)
    @everywhere const _n_restarts   = $(_n_restarts)
    @everywhere const _N            = $(_N)
    @everywhere const _optimize_det = $(_optimize_det)
    @everywhere const _log_fid      = $(_log_fid)
    @everywhere const _lns          = $(_lns)
    @everywhere const _sigma_mode   = $(QuoteNode(_sigma_mode))
    @everywhere const _bounds       = $(_bounds)

    # Derived physical parameters (computed on main, broadcast to workers)
    base = CalibrationCode.ideal(_t)
    _f_cl0 = base.f_cl
    _f_sb0 = base.f_sb
    _A0    = base.A

    _span_kHz = 2.0
    _span_fcl = _span_kHz * 1e3 * 2π
    _span_fsb = _span_kHz * 1e3 * 2π
    _span_A   = 1.2 * _A0 - _A0

    @everywhere const __f_cl0    = $(_f_cl0)
    @everywhere const __f_sb0    = $(_f_sb0)
    @everywhere const __A0       = $(_A0)
    @everywhere const __span_fcl = $(_span_fcl)
    @everywhere const __span_fsb = $(_span_fsb)
    @everywhere const __span_A   = $(_span_A)

    @everywhere function _u_to_params(u)
        (__f_cl0 + __span_fcl * u[1],
         __f_sb0 + __span_fsb * u[2],
         __A0    + __span_A   * u[3])
    end

    @everywhere function _sigma_y(Q_raw::Float64)
        # Returns σy in the space that the GP sees (log or linear).
        _optimize_det && return 0.0
        if _sigma_mode === :binomial
            # Binomial std in Q-space: sqrt(Q*(1-Q)/N)
            # If log transform: delta method through y = log10(1-Q)
            #   dy/dQ = -1/((1-Q)*ln(10))  =>  σ_y = σ_Q / ((1-Q)*ln(10))
            #         = sqrt(Q*(1-Q)/N) / ((1-Q)*ln(10))
            #         = sqrt(Q/(1-Q)) / (sqrt(N)*ln(10))
            one_minus_Q = max(1.0 - Q_raw, 1e-15)
            σ_Q = sqrt(max(Q_raw, 0.0) * one_minus_Q / _N)
            return _log_fid ? σ_Q / one_minus_Q : σ_Q
        else  # :simple
            return sqrt(1.0 / _N)
        end
    end

    @everywhere function _apply_log(Q::Float64)
        _log_fid ? log10(max(1.0 - Q, 1e-15)) : Q
    end

    @everywhere function _run_seed(seed::Int)
        rng = Random.MersenneTwister(seed)
        d   = length(_bounds)
        lb  = Float64[b[1] for b in _bounds]
        ub  = Float64[b[2] for b in _bounds]

        X_pre  = Matrix{Float64}(undef, d, _n_pretrain)
        y_pre  = Vector{Float64}(undef, _n_pretrain)
        σy_pre = Vector{Float64}(undef, _n_pretrain)

        for i in 1:_n_pretrain
            x = lb .+ rand(rng, d) .* (ub .- lb)
            X_pre[:, i] = x
            fcl, fsb, A = _u_to_params(x)
            Q_raw = _optimize_det ?
                CalibrationCode.Q_det(_t, fcl, fsb, A) :
                CalibrationCode.Q_varMS(_t, fcl, fsb, A; N=_N, numMS=2)
            y_pre[i]  = _apply_log(Q_raw)
            σy_pre[i] = _sigma_y(Q_raw)
        end

        gp = CalibrationCode.fit_heterogp(X_pre, y_pre, σy_pre;
            learn_hypers=true,
            learn_noise_scale=_lns,
            n_restarts=_n_restarts,
            jitter=1e-8,
            rng=rng,
        )
        return gp.θ
    end

    seeds = rand(MersenneTwister(0), 1:1000000, _num_sims)

    println("=== Pretraining GP hyperparameters (3D) — $(nworkers()) workers ===")
    println("optimize_det=$(_optimize_det), use_log_fidelity=$(_log_fid), sigma_mode=$(_sigma_mode)")
    println("n_pretrain=$(_n_pretrain), n_restarts=$(_n_restarts), num_sims=$(_num_sims)")
    flush(stdout)

    all_θ = pmap(_run_seed, seeds; batch_size=1)

    d = length(_bounds)
    d_hyp = length(all_θ[1])  # d+1 if learn_noise_scale=false, d+2 if true

    # Build param names to match actual θ length
    ℓ_names = ["logℓ$(i)" for i in 1:d]
    param_names = vcat(ℓ_names, ["logσf"], d_hyp > d+1 ? ["logc"] : String[])

    # Summary statistics across seeds
    println("\n=== Hyperparameter variability across $(length(seeds)) seeds ===")
    θ_mat = hcat(all_θ...)   # d_hyp × n_seeds
    for (i, name) in enumerate(param_names)
        row = θ_mat[i, :]
        println("  $name:  mean=$(round(mean(row), digits=4))  std=$(round(std(row), digits=4))  range=[$(round(minimum(row), digits=4)), $(round(maximum(row), digits=4))]")
    end

    pairwise_dists = Float64[]
    n = length(all_θ)
    for i in 1:n, j in (i+1):n
        push!(pairwise_dists, sqrt(sum((all_θ[i] .- all_θ[j]).^2)))
    end
    if !isempty(pairwise_dists)
        println("  ||θᵢ - θⱼ||₂: mean=$(round(mean(pairwise_dists), digits=4))  max=$(round(maximum(pairwise_dists), digits=4))")
    end

    # Save the first seed's result as the canonical pretrained file
    θ_save  = all_θ[1]
    ℓ_save  = exp.(θ_save[1:d])
    σf_save = exp(θ_save[d+1])
    c_save  = d_hyp > d+1 ? exp(θ_save[d+2]) : NaN

    output_path = joinpath(@__DIR__, "..", "data", "pretrained_theta_3d.jl")
    open(output_path, "w") do io
        println(io, "# Auto-generated pretrained GP hyperparameters for 3D quantum calibration")
        println(io, "# Generated by: julia --project=. scripts/pretrain_gp_3d.jl")
        println(io, "# θ layout: [logℓ₁, logℓ₂, logℓ₃, logσf, logc]")
        println(io, "# Seed=$(seeds[1]): ℓ=$(round.(ℓ_save, digits=4)), σf=$(round(σf_save, digits=4)), c=$(round(c_save, digits=4))")
        println(io, "pretrained_theta_3d() = ", repr(θ_save))
    end

    # Save all seeds for variability plotting
    all_path = joinpath(@__DIR__, "..", "data", _filename_output)
    open(all_path, "w") do io
        println(io, "# All-seeds GP hyperparameters for variability analysis")
        println(io, "# Generated by: julia --project=. scripts/pretrain_gp_3d.jl")
        println(io, "# θ layout: [logℓ₁, logℓ₂, logℓ₃, logσf, logc]")
        println(io, "pretrained_seeds() = $(repr(seeds))")
        println(io, "pretrained_theta_all() = $(repr(all_θ))")
    end

    println("\nSaved canonical θ  → $output_path")
    println("Saved all seeds    → $all_path")
    println("Run plot_pretrain_variability.jl to visualise hyperparameter spread.")

finally
    rmprocs(workers())
end
