using Random
using Distributions
using Plots

using CalibrationCode
include(joinpath(@__DIR__, "..", "scripts", "plots_standard.jl"))  # provides plot2d_standard/animate2d_standard

function main(; seed=1, σ=0.01)
    rng = MersenneTwister(seed)

    f_true(x) = 1.0 - (x[1] - 0.3)^2 - (x[2] + 0.2)^2
    f_noisy(x) = f_true(x) + rand(rng, Normal(0, σ))

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    res, x_rec, y_rec = CalibrationCode.bayesopt(f_noisy;
        bounds=bounds,
        n_init=6,
        n_iter=60,
        xi=0.01,
        maximize=true,
        seed=seed,
        obs_noise=σ
    )

    best_idx = argmax(res.y)
    x_best = vec(res.X[:, best_idx])
    println("Best observed x = ", x_best, "   best observed y = ", res.y[best_idx])
    println("Recommended x (posterior mean argmax) = ", x_rec, "   y ≈ ", y_rec)

    p = plot2d_standard(res; f_true=f_true, nx=50, ny=50, obs_noise=σ)
    savefig(p, "toy_standard_2d.png")
    println("Saved -> toy_standard_2d.png")

    anim, fps = animate2d_standard(res; f_true=f_true, nx=50, ny=50, fps=6, obs_noise=σ)
    gif(anim, "toy_standard_2d.gif", fps=fps)
    println("Saved -> toy_standard_2d.gif")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
