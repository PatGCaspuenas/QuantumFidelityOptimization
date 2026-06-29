import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using Distributions
using .CalibrationCode

function main(; seed=1, σ=0.01)
    rng = MersenneTwister(seed)

    f_true(x) = 1.0 - (x[1] - 0.3)^2 - (x[2] + 0.2)^2
    f_noisy(x) = f_true(x) + rand(rng, Normal(0, σ))

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    res = CalibrationCode.bayesopt(f_noisy;
        bounds=bounds,
        n_init=6,
        n_iter=100,
        xi=0.01,
        maximize=true,
        seed=1,
        obs_noise=σ
    )

    best_idx = argmax(res.y)
    x_rec = vec(res.X[:, best_idx])
    println("Best observed x = ", x_rec, "   y = ", res.y[best_idx])
    println("Distance to true optimum ≈ ", sqrt((x_rec[1]-0.3)^2 + (x_rec[2]+0.2)^2))
end

if !isinteractive()
    main()
end
