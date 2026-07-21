import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using .CalibrationCode

function main(; seed=1, σ=0.01)
    rng = MersenneTwister(seed)

    f_true(x) = 1.0 - (x[1] - 0.3)^2 - (x[2] + 0.2)^2
    # Constant (homoscedastic) noise: N(0, σ) at every point, σy known.
    f_noisy(x, _) = (f_true(x) + randn(rng) * σ, σ)

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    res = CalibrationCode.bayesopt_ucb(f_noisy;
        bounds=bounds,
        n_init=6,
        n_iter=100,
        seed=seed
    )

    println("Recommended x = ", res.x_rec, "   GP-mean y ≈ ", res.y_rec)
    println("Distance to true optimum ≈ ", sqrt((res.x_rec[1]-0.3)^2 + (res.x_rec[2]+0.2)^2))
end

if !isinteractive()
    main()
end
