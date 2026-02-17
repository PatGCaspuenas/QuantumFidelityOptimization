# scripts/plots_mf.jl
# Plots for augmented-input multi-fidelity MFResult (z in the last row of Xa)

using Plots
using Random, Statistics
using Distributions, GaussianProcesses

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

function _grid(bounds, nx, ny)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]
    xs = range(lb[1], ub[1], length=nx)
    ys = range(lb[2], ub[2], length=ny)
    return xs, ys
end

function _fit_gp_aug(Xa::Matrix{Float64}, y::Vector{Float64}; obs_noise=1e-3, optimize_hypers=false)
    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ

    d, _ = size(Xa)
    ℓ0  = fill(0.3, d)
    σf0 = 1.0
    σn0 = max(obs_noise, 1e-8)

    gp = GP(Xa, ystd, MeanZero(), Matern(3/2, ℓ0, σf0), σn0)
    if optimize_hypers
        try
            optimize!(gp; domean=false, noise=false)
        catch err
            @warn "GP optimization failed; using initial hypers" err
        end
    end
    return gp, yμ, yσ
end

@inline function _gp_predict1(gp, u::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(u, :, 1))
    return μ[1], σ2[1]
end

function plot2d_mf(res::CalibrationCode.MFResult;
                   ℓ::Int=argmax(res.z_levels),     # default: highest z
                   t::Int=size(res.Xa,2)-res.n_init,
                   nx::Int=40, ny::Int=40,
                   f_true::Union{Nothing,Function}=nothing,   # optionally f_true(x) at this fidelity
                   obs_noise::Float64=1e-3,
                   optimize_hypers::Bool=false,
                   atol_z::Float64=1e-12)

    length(res.bounds) == 2 || throw(ArgumentError("plot2d_mf supports d=2 only"))
    (1 ≤ ℓ ≤ length(res.z_levels)) || throw(ArgumentError("ℓ out of range"))
    d = length(res.bounds)
    z = res.z_levels[ℓ]

    n   = res.n_init + t
    Xan = res.Xa[:, 1:n]
    y   = res.y[1:n]

    xs, ys = _grid(res.bounds, nx, ny)
    gp, yμ, yσ = _fit_gp_aug(Xan, y; obs_noise=obs_noise, optimize_hypers=optimize_hypers)

    Zμ   = zeros(length(ys), length(xs))
    Zσ   = zeros(length(ys), length(xs))
    Zerr = f_true === nothing ? nothing : zeros(length(ys), length(xs))

    for (iy, yv) in enumerate(ys), (ix, xv) in enumerate(xs)
        u = vcat([xv, yv], z)
        μstd, σ2std = _gp_predict1(gp, u)
        m = yμ + yσ * μstd
        s = sqrt(max(σ2std, 0.0)) * yσ
        Zμ[iy, ix] = m
        Zσ[iy, ix] = s
        if Zerr !== nothing
            Zerr[iy, ix] = abs(m - f_true([xv, yv]))
        end
    end

    p_mean = contourf(xs, ys, Zμ; title="Posterior mean (ℓ=$ℓ, z=$(z))", xlabel="x₁", ylabel="x₂")

    # show sampled points, colored by fidelity z
    zcols = Xan[d+1, :]
    scatter!(p_mean, Xan[1,:], Xan[2,:]; marker_z=zcols, ms=4, label=false, colorbar_title="z")

    p_unc = contourf(xs, ys, Zσ; title="Posterior std (ℓ=$ℓ)", xlabel="x₁", ylabel="x₂")
    scatter!(p_unc, Xan[1,:], Xan[2,:]; marker_z=zcols, ms=4, label=false, colorbar_title="z")

    # best-so-far at the selected fidelity (filter by z≈z_levels[ℓ])
    idxℓ = findall(i -> isapprox(Xan[d+1, i], z; atol=atol_z, rtol=0), 1:size(Xan,2))
    best_trace = isempty(idxℓ) ? Float64[] : accumulate(max, y[idxℓ])
    p_trace = plot(best_trace; title="Best-so-far at ℓ=$ℓ", xlabel="eval@ℓ", ylabel="best y", lw=2)

    if Zerr === nothing
        return plot(p_mean, p_unc, p_trace; layout=(1,3), size=(1200,400))
    else
        p_err = contourf(xs, ys, Zerr; title="|μ(x) − f(x)| (ℓ=$ℓ)", xlabel="x₁", ylabel="x₂")
        scatter!(p_err, Xan[1,:], Xan[2,:]; marker_z=zcols, ms=4, label=false, colorbar_title="z")
        return plot(p_mean, p_unc, p_err, p_trace; layout=(2,2), size=(1000,800))
    end
end

function animate2d_mf(res::CalibrationCode.MFResult;
                      ℓ::Int=argmax(res.z_levels),
                      f_true::Union{Nothing,Function}=nothing,
                      nx::Int=40, ny::Int=40, fps::Int=10,
                      obs_noise::Float64=1e-3,
                      optimize_hypers::Bool=false)
    anim = @animate for t in 1:(size(res.Xa,2) - res.n_init)
        plot2d_mf(res; ℓ=ℓ, t=t, nx=nx, ny=ny, f_true=f_true,
                  obs_noise=obs_noise, optimize_hypers=optimize_hypers)
    end
    return anim, fps
end
