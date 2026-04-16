import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using Distributions
using .CalibrationCode

function main(; seed=2)
    rng = MersenneTwister(seed)

    # High-fidelity truth: optimum near (0.2, -0.4, 0.7)
    f_hi(x) = 1.0 - (x[1]-0.2)^2 - 2.0*(x[2]+0.4)^2 - 0.5*(x[3]-0.7)^2
    f_mid(x) = 0.93*f_hi(x) + 0.03*sin(4x[1]) - 0.02
    f_lo(x)  = 0.85*f_hi(x) + 0.06*sin(6x[1]) - 0.05

    σ_lo, σ_mid, σ_hi = 0.06, 0.03, 0.015

    function f_mf(x, ℓ)
        if ℓ == 1
            return f_lo(x)  + rand(rng, Normal(0, σ_lo))
        elseif ℓ == 2
            return f_mid(x) + rand(rng, Normal(0, σ_mid))
        elseif ℓ == 3
            return f_hi(x)  + rand(rng, Normal(0, σ_hi))
        else
            throw(ArgumentError("invalid fidelity index ℓ=$ℓ"))
        end
    end

    bounds   = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]
    z_levels = [0.0, 0.5, 1.0]
    costs    = [1.0, 4.0, 25.0]

    res = CalibrationCode.bayesopt_mf(f_mf;
        bounds=bounds,
        z_levels=z_levels,
        costs=costs,
        n_init=16,
        n_iter=90,
        M=5000,
        xi=0.01,
        obs_noise=1e-3,
        optimize_hypers=false,
        seed=seed
    )

    println("Recommended x = ", res.x_rec, "   GP-mean at highest fidelity ≈ ", res.y_rec)
    println("True hi-fidelity value at x_rec = ", f_hi(res.x_rec))

    println("Distance to true optimum ≈ ",
        sqrt((res.x_rec[1]-0.2)^2 + (res.x_rec[2]+0.4)^2 + (res.x_rec[3]-0.7)^2))
end

if !isinteractive()
    main()
end
