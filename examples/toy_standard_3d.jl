import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using .CalibrationCode

function main(; seed=2, σ=0.02)
    rng = MersenneTwister(seed)

    # Smooth 3D bowl with maximum near (0.2, -0.4, 0.7)
    f_true(x) = 1.0 - (x[1] - 0.2)^2 - 2.0*(x[2] + 0.4)^2 - 0.5*(x[3] - 0.7)^2
    # Constant (homoscedastic) noise: N(0, σ) at every point, σy known.
    f_noisy(x, _) = (f_true(x) + randn(rng) * σ, σ)

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]

    res = CalibrationCode.bayesopt_ucb(f_noisy;
        bounds=bounds,
        n_init=10,
        n_iter=80,
        seed=seed
    )

    println("Recommended x = ", res.x_rec, "   GP-mean y ≈ ", res.y_rec)
    println("Distance to true optimum ≈ ",
        sqrt((res.x_rec[1]-0.2)^2 + (res.x_rec[2]+0.4)^2 + (res.x_rec[3]-0.7)^2))
end

if !isinteractive()
    main()
end
