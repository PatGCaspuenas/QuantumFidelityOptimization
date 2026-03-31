# scripts/plots_cokrig.jl
# Plots for N-fidelity AR(1) co-kriging result (MFCoKrigResult)
# Contours are drawn for the chosen level (default: highest).

using Plots
using .CalibrationCode

function _grid(bounds, nx, ny)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]
    xs = range(lb[1], ub[1], length=nx)
    ys = range(lb[2], ub[2], length=ny)
    return xs, ys
end

function _slice_data(res::CalibrationCode.MFCoKrigResult, n::Int)
    N = length(res.Xs)
    Xs = Vector{Matrix{Float64}}(undef, N)
    ys = Vector{Vector{Float64}}(undef, N)
    for m in 1:N
        nm = size(res.Xs[m], 2)
        k = min(n, nm)
        Xs[m] = res.Xs[m][:, 1:k]
        ys[m] = res.ys[m][1:k]
    end
    return Xs, ys
end

function plot2d_cokrig(res::CalibrationCode.MFCoKrigResult;
                       level::Int=length(res.Xs),          # default: highest fidelity
                       n::Union{Nothing,Int}=nothing,      # for progressive/animation: points per level
                       nx::Int=40, ny::Int=40,
                       obs_noises=nothing,
                       optimize_hypers::Bool=false,
                       show_std::Bool=false,
                       f_true::Union{Nothing,Function}=nothing,
                       show_points::Bool=true)

    length(res.bounds) == 2 || throw(ArgumentError("plot2d_cokrig supports d=2 only"))

    N = length(res.Xs)
    (1 ≤ level ≤ N) || throw(ArgumentError("level must be in 1:$N"))

    if obs_noises === nothing
        obs_noises = fill(1e-3, N)
    end
    length(obs_noises) == N || throw(DimensionMismatch("obs_noises must have length N"))

    Xs, ys = n === nothing ? (res.Xs, res.ys) : _slice_data(res, n)

    model = CalibrationCode.fit_mfcokrig(Xs, ys; obs_noises=obs_noises, optimize_hypers=optimize_hypers)

    xs, ysgrid = _grid(res.bounds, nx, ny)

    Zμ   = zeros(length(ysgrid), length(xs))
    Zσ   = show_std ? zeros(length(ysgrid), length(xs)) : nothing
    Zerr = f_true === nothing ? nothing : zeros(length(ysgrid), length(xs))

    for (iy, yv) in enumerate(ysgrid), (ix, xv) in enumerate(xs)
        μ, s2 = CalibrationCode.predict_level(model, [xv, yv], level)
        Zμ[iy, ix] = μ
        if show_std
            Zσ[iy, ix] = sqrt(max(s2, 0.0))
        end
        if Zerr !== nothing
            Zerr[iy, ix] = abs(μ - f_true([xv, yv]))
        end
    end

    p_mean = contourf(xs, ysgrid, Zμ; title="Posterior mean (level $level)", xlabel="x₁", ylabel="x₂")

    if show_points
        for m in 1:N
            Xm = Xs[m]
            if size(Xm, 2) > 0
                scatter!(p_mean, Xm[1, :], Xm[2, :]; ms=3, label="level $m")
            end
        end
    end

    plots = [p_mean]

    if show_std
        p_std = contourf(xs, ysgrid, Zσ; title="Posterior std (level $level)", xlabel="x₁", ylabel="x₂")
        if show_points
            for m in 1:N
                Xm = Xs[m]
                if size(Xm, 2) > 0
                    scatter!(p_std, Xm[1, :], Xm[2, :]; ms=3, label=false)
                end
            end
        end
        push!(plots, p_std)
    end

    if Zerr !== nothing
        p_err = contourf(xs, ysgrid, Zerr; title="|μ(x) − f(x)| (level $level)", xlabel="x₁", ylabel="x₂")
        if show_points
            for m in 1:N
                Xm = Xs[m]
                if size(Xm, 2) > 0
                    scatter!(p_err, Xm[1, :], Xm[2, :]; ms=3, label=false)
                end
            end
        end
        push!(plots, p_err)
    end

    layout = (1, length(plots))
    return plot(plots...; layout=layout, size=(420 * length(plots), 420))
end

function animate2d_cokrig(res::CalibrationCode.MFCoKrigResult;
                          level::Int=length(res.Xs),
                          nx::Int=40, ny::Int=40,
                          obs_noises=nothing,
                          optimize_hypers::Bool=false,
                          show_std::Bool=false,
                          fps::Int=10)

    N = length(res.Xs)
    maxn = maximum(size(res.Xs[m], 2) for m in 1:N)

    anim = @animate for n in 2:maxn
        plot2d_cokrig(res; level=level, n=n, nx=nx, ny=ny,
                      obs_noises=obs_noises,
                      optimize_hypers=optimize_hypers,
                      show_std=show_std)
    end
    return anim, fps
end
