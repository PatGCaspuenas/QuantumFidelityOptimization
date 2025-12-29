using Random
using Distributions
using Plots

using CalibrationCode
include(joinpath(@__DIR__, "..", "scripts", "plots_hetero.jl"))  # plot2d_hetero / animate2d_hetero

function main(; seed=2)
    rng = MersenneTwister(seed)

    f_true(x) = 1.0 - (x[1] - 0.3)^2 - (x[2] + 0.2)^2

    # Objective signature required by bayesopt_ucb_threshold: f(x, σ)::Float64
    f_noisy(x, σ) = f_true(x) + rand(rng, Normal(0, σ))

    bounds   = [(-1.0, 1.0), (-1.0, 1.0)]
    σ_levels = [0.5, 0.1, 0.05, 0.02, 0.01, 0.005]

    res = CalibrationCode.bayesopt_ucb_threshold(f_noisy;
        bounds=bounds,
        σ_levels=σ_levels,
        n_init=6,
        n_iter=60,
        κ=2.0,
        α=0.5,
        seed=seed
    )

    println("Recommended x = ", res.x_rec, "   GP-mean y ≈ ", res.y_rec)

    counts = CalibrationCode.count_noise_levels(res)
    for σ in res.σ_levels
        println("σ=$(σ): ", counts[σ])
    end

    p = plot2d_hetero(res; f_true=f_true, nx=50, ny=50, fps=6, show_noise=true)
    savefig(p, "toy_hetero_2d.png")
    println("Saved -> toy_hetero_2d.png")

    anim, fps = animate2d_hetero(res; f_true=f_true, nx=50, ny=50, fps=6, show_noise=true)
    gif(anim, "toy_hetero_2d.gif", fps=fps)
    println("Saved -> toy_hetero_2d.gif")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
