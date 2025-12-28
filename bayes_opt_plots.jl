module BayesPlotting

using Plots

# Helper: common grid
function _grid(bounds, nx, ny)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]
    xs = range(lb[1], ub[1], length=nx)
    ys = range(lb[2], ub[2], length=ny)
    return xs, ys
end

# -------------------------
# Homoscedastic: BayesOpt
# -------------------------
function plot2d(res::Main.BayesOpt.BOResult;
                t::Int=size(res.X,2)-res.n_init,
                nx::Int=40, ny::Int=40,
                f_true::Union{Nothing,Function}=nothing)

    @assert length(res.bounds) == 2 "plot2d supports d=2 only"

    n  = res.n_init + t
    Xn = res.X[:, 1:n]
    y  = res.y[1:n]

    xs, ys = _grid(res.bounds, nx, ny)

    gp, yμ, yσ = Main.BayesOpt.fit_gp(Xn, y)

    Zμ   = zeros(length(ys), length(xs))
    Zσ   = zeros(length(ys), length(xs))
    Zerr = f_true === nothing ? nothing : zeros(length(ys), length(xs))

    for (iy, yv) in enumerate(ys), (ix, xv) in enumerate(xs)
        μstd, σ2std = Main.BayesOpt.gp_predict_f1(gp, [xv, yv])
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

function animate2d(res::Main.BayesOpt.BOResult;
                   f_true::Union{Nothing,Function}=nothing,
                   nx::Int=40, ny::Int=40, fps::Int=10)
    anim = @animate for t in 1:(size(res.X,2) - res.n_init)
        plot2d(res; t=t, nx=nx, ny=ny, f_true=f_true)
    end
    return anim, fps
end

# -------------------------
# Heteroscedastic: BayesHeteroOpt
# -------------------------
function plot2d(res::Main.BayesHeteroOpt.HeteroBOResult;
                t::Int=size(res.X,2)-res.n_init,
                nx::Int=40, ny::Int=40,
                f_true::Union{Nothing,Function}=nothing,
                show_noise::Bool=true)

    @assert length(res.bounds) == 2 "plot2d supports d=2 only"

    n  = res.n_init + t
    Xn = res.X[:, 1:n]
    y  = res.y[1:n]
    σn = res.σy[1:n]

    xs, ys = _grid(res.bounds, nx, ny)

    gp = Main.BayesHeteroOpt.fit_gp(Xn, y, σn)

    Zμ   = zeros(length(ys), length(xs))
    Zσ   = zeros(length(ys), length(xs))
    Zerr = f_true === nothing ? nothing : zeros(length(ys), length(xs))

    for (iy, yv) in enumerate(ys), (ix, xv) in enumerate(xs)
        μ, s2 = Main.BayesHeteroOpt.predict_latent(gp, [xv, yv])
        Zμ[iy, ix] = μ
        Zσ[iy, ix] = sqrt(max(s2, 0.0))
        if Zerr !== nothing
            Zerr[iy, ix] = abs(μ - f_true([xv, yv]))
        end
    end

    p_mean = contourf(xs, ys, Zμ; title="Posterior mean", xlabel="x₁", ylabel="x₂")
    if show_noise
        scatter!(p_mean, Xn[1,:], Xn[2,:]; marker_z=σn, ms=4, label=false, colorbar_title="σ")
    else
        scatter!(p_mean, Xn[1,:], Xn[2,:]; ms=3, color=:white, label=false)
    end

    p_unc = contourf(xs, ys, Zσ; title="Posterior std", xlabel="x₁", ylabel="x₂")
    if show_noise
        scatter!(p_unc, Xn[1,:], Xn[2,:]; marker_z=σn, ms=4, label=false, colorbar_title="σ")
    else
        scatter!(p_unc, Xn[1,:], Xn[2,:]; ms=3, color=:white, label=false)
    end

    best_trace = accumulate(max, y)
    p_trace = plot(best_trace; title="Best-so-far (observed)", xlabel="eval", ylabel="best y", lw=2)

    if Zerr === nothing
        return plot(p_mean, p_unc, p_trace; layout=(1,3), size=(1200,400))
    else
        p_err = contourf(xs, ys, Zerr; title="|μ(x) − f(x)|", xlabel="x₁", ylabel="x₂")
        if show_noise
            scatter!(p_err, Xn[1,:], Xn[2,:]; marker_z=σn, ms=4, label=false, colorbar_title="σ")
        else
            scatter!(p_err, Xn[1,:], Xn[2,:]; ms=3, color=:white, label=false)
        end
        return plot(p_mean, p_unc, p_err, p_trace; layout=(2,2), size=(1000,800))
    end
end

function animate2d(res::Main.BayesHeteroOpt.HeteroBOResult;
                   f_true::Union{Nothing,Function}=nothing,
                   nx::Int=40, ny::Int=40, fps::Int=10,
                   show_noise::Bool=true)
    anim = @animate for t in 1:(size(res.X,2) - res.n_init)
        println(t)
        plot2d(res; t=t, nx=nx, ny=ny, f_true=f_true, show_noise=show_noise)
    end
    return anim, fps
end

end # module
