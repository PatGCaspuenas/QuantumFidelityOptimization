using Random
using Distributions
using Plots

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode


include(joinpath(@__DIR__, "..", "scripts", "plots_standard.jl"))  # provides plot2d_standard/animate2d_standard

function main(; seed=1, σ=0.01)
    rng = MersenneTwister(seed)

    f_true(x) = 1.0 - (x[1] - 0.3)^2 - (x[2] + 0.2)^2
    f_noisy(x) = f_true(x) + rand(rng, Normal(0, σ))

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    out = CalibrationCode.bayesopt(f_noisy;
        bounds=bounds,
        n_init=6,
        n_iter=100,
        xi=0.01,
        maximize=true,
        seed=1,
        obs_noise=σ
    )

    if out isa Tuple
        res, x_rec, y_rec = out
    else
        res = out
        best_idx = argmax(res.y)
        x_rec = vec(res.X[:, best_idx])
        y_rec = res.y[best_idx]
    end

    # Works whether bayesopt returns BOResult OR (res, x_rec, y_rec)
    if out isa Tuple
        res, x_rec, y_rec = out
    else
        res = out
        best_idx = argmax(res.y)
        x_rec = vec(res.X[:, best_idx])
        y_rec = res.y[best_idx]   # fallback: best observed
    end

    best_idx = argmax(res.y)
    x_best = vec(res.X[:, best_idx])
    println("Best observed x = ", x_best, "   best observed y = ", res.y[best_idx])
    println("Recommended x (posterior mean argmax) = ", x_rec, "   y ≈ ", y_rec)

    p = plot2d_standard(res; f_true=f_true, nx=50, ny=50, obs_noise=σ)
    savefig(p, "figures/toy_standard_2d.png")
    println("Saved -> toy_standard_2d.png")

    anim, fps = animate2d_standard(res; f_true=f_true, nx=50, ny=50, fps=6, obs_noise=σ)
    gif(anim, "figures/toy_standard_2d.gif", fps=fps)
    println("Saved -> toy_standard_2d.gif")
end

main()

