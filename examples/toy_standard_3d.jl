import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using Distributions
using .CalibrationCode

function main(; seed=2, σ=0.02)
    rng = MersenneTwister(seed)

    # Smooth 3D bowl with maximum near (0.2, -0.4, 0.7)
    f_true(x) = 1.0 - (x[1] - 0.2)^2 - 2.0*(x[2] + 0.4)^2 - 0.5*(x[3] - 0.7)^2
    f_noisy(x) = f_true(x) + rand(rng, Normal(0, σ))

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]

    res = CalibrationCode.bayesopt(f_noisy;
        bounds=bounds,
        n_init=10,
        n_iter=80,
        xi=0.01,
        maximize=true,
        seed=seed,
        obs_noise=σ
    )

    best_idx = argmax(res.y)
    x_rec = vec(res.X[:, best_idx])
    y_rec = res.y[best_idx]

    x_best = vec(res.X[:, best_idx])
    println("Best observed x = ", x_best, "   best observed y = ", res.y[best_idx])
    println("Recommended x (posterior mean argmax) = ", x_rec, "   y ≈ ", y_rec)

    # Simple sanity check target (not a unit test, just quick feedback)
    println("Distance to true optimum ≈ ",
        sqrt((x_rec[1]-0.2)^2 + (x_rec[2]+0.4)^2 + (x_rec[3]-0.7)^2))
end

if !isinteractive()
    main()
end
