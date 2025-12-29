# scripts/plots_standard.jl
# Plots for standard (homoscedastic) BOResult

using Plots
using Random, Statistics
using Distributions, GaussianProcesses
using CalibrationCode

function _grid(bounds, nx, ny)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]
    xs = range(lb[1], ub[1], length=nx)
    ys = range(lb[2], ub[2], length=ny)
    return xs, ys
end

function _fit_gp_standard(X::Matrix{Float64}, y::Vector{Float64}; obs_noise=1e-5, optimize_hypers=false)
    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ

    d, _ = size(X)
    ℓ0  = fill(0.3, d)
    σf0 = 1.0
    σn0 = max(obs_noise, 1e-10)

    gp = GP(X, ystd, MeanZero(), Matern(3/2, ℓ0, σf0), σn0)
    if optimize_hypers
        try
            optimize!(gp)
        catch err
            @warn "GP optimization failed; using initial hypers" err
        end
    end
    return gp, yμ, yσ
end

@inline function _gp_predict1(gp, x::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(x, :, 1))
    return μ[1], σ2[1]
end

function plot2d_standard(res::CalibrationCode.BOResult;
                         t::Int=size(res.X,2)-res.n_init,
                         nx::Int=40, ny::Int=40,
                         f_true::Union{Nothing,Function}=nothing,
                         obs_noise::Float64=1e-5,
                         optimize_hypers::Bool=false)

    length(res.bounds) == 2 || throw(ArgumentError("plot2d_standard supports d=2 only"))

    n  = res.n_init + t
    Xn = res.X[:, 1:n]
    y  = res.y[1:n]

    xs, ys = _grid(res.bounds, nx, ny)
    gp, yμ, yσ = _fit_gp_standard(Xn, y; obs_noise=obs_noise, optimize_hypers=optimize_hypers)

    Zμ   = zeros(length(ys), length(xs))
    Zσ   = zeros(length(ys), length(xs))
    Zerr = f_true === nothing ? nothing : zeros(length(ys), length(xs))

    for (iy, yv) in enumerate(ys), (ix, xv) in enumerate(xs)
        μstd, σ2std = _gp_predict1(gp, [xv, yv])
        m = yμ + yσ * μstd
        s = sqrt(max(σ2std, 0.0)) * yσ
        Zμ[iy, ix] = m
        Zσ[iy, ix] = s
        if Zerr !== nothing
            Zerr[iy, ix] = abs(m - f_true([xv, yv]))
        end
    end

    p_mean = contourf(xs, ys, Zμ; title="Posterior mean", xlabel="x₁", ylabel="x₂")
    scatter!(p_mean, Xn[1,:], Xn[2,:]; ms=3, color=:white, label="samples")

    p_unc = contourf(xs, ys, Zσ; title="Posterior std", xlabel="x₁", ylabel="x₂")
    scatter!(p_unc, Xn[1,:], Xn[2,:]; ms=3, color=:white, label=false)

    best_trace = accumulate(max, y)
    p_trace = plot(best_trace; title="Best-so-far", xlabel="eval", ylabel="best y", lw=2)

    if Zerr === nothing
        return plot(p_mean, p_unc, p_trace; layout=(1,3), size=(1200,400))
    else
        p_err = contourf(xs, ys, Zerr; title="|μ(x) − f(x)|", xlabel="x₁", ylabel="x₂")
        scatter!(p_err, Xn[1,:], Xn[2,:]; ms=3, color=:white, label=false)
        return plot(p_mean, p_unc, p_err, p_trace; layout=(2,2), size=(1000,800))
    end
end

function animate2d_standard(res::CalibrationCode.BOResult;
                            f_true::Union{Nothing,Function}=nothing,
                            nx::Int=40, ny::Int=40, fps::Int=10,
                            obs_noise::Float64=1e-5,
                            optimize_hypers::Bool=false)
    anim = @animate for t in 1:(size(res.X,2) - res.n_init)
        plot2d_standard(res; t=t, nx=nx, ny=ny, f_true=f_true,
                        obs_noise=obs_noise, optimize_hypers=optimize_hypers)
    end
    return anim, fps
end
