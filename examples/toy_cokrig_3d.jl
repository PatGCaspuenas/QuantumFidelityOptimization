using Random
using Distributions
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

function main(; seed=4)
    rng = MersenneTwister(seed)

    f_hi(x) = 1.0 - (x[1]-0.2)^2 - 2.0*(x[2]+0.4)^2 - 0.5*(x[3]-0.7)^2
    f_mid(x) = 0.93*f_hi(x) + 0.03*sin(4x[1]) - 0.02
    f_lo(x)  = 0.88*f_hi(x) + 0.06*sin(6x[1]) - 0.05

    σ = [0.06, 0.03, 0.015]
    costs = [1.0, 5.0, 25.0]

    function f_level(x, level::Int)
        level == 1 && return f_lo(x)  + rand(rng, Normal(0, σ[1]))
        level == 2 && return f_mid(x) + rand(rng, Normal(0, σ[2]))
        level == 3 && return f_hi(x)  + rand(rng, Normal(0, σ[3]))
        throw(ArgumentError("invalid level=$level"))
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]

    res = CalibrationCode.mfcokrig_bayesopt(f_level;
        bounds=bounds,
        costs=costs,
        n_init=22,
        n_iter=80,
        M=5000,
        xi=0.01,
        seed=seed,
        optimize_hypers=false
    )

    println("Recommended x = ", res.x_rec, "   predicted mean at highest fidelity ≈ ", res.y_rec)
    println("True high-fidelity at x_rec = ", f_hi(res.x_rec))
    println("Distance to true optimum ≈ ",
        sqrt((res.x_rec[1]-0.2)^2 + (res.x_rec[2]+0.4)^2 + (res.x_rec[3]-0.7)^2))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
