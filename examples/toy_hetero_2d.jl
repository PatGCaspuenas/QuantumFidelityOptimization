import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using .CalibrationCode

function main(; seed=2)
    rng = MersenneTwister(seed)

    f_true(x) = clamp(1.0 - (x[1] - 0.3)^2 - (x[2] + 0.2)^2, 0.0, 1.0)

    function f_noisy(x, n)
        y0 = f_true(x)
        σy = sqrt(max(y0 * (1.0 - y0), 0.0) / Float64(n))
        y  = clamp(y0 + randn(rng) * σy, 0.0, 1.0)
        return y, σy
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    res = CalibrationCode.bayesopt_ucb(f_noisy;
        bounds=bounds,
        n_shots=200,
        n_init=6,
        n_iter=60,
        seed=seed
    )

    println("Recommended x = ", res.x_rec, "   GP-mean y ≈ ", res.y_rec)
    println("Distance to true optimum ≈ ", sqrt((res.x_rec[1]-0.3)^2 + (res.x_rec[2]+0.2)^2))
end

if !isinteractive()
    main()
end
