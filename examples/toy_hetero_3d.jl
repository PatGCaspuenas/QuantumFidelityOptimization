using Random
using Distributions
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

function main(; seed=3)
    rng = MersenneTwister(seed)

    # 3D maximum near (0.2, -0.4, 0.7)
    f_true(x) = 1.0 - (x[1] - 0.2)^2 - 2.0*(x[2] + 0.4)^2 - 0.5*(x[3] - 0.7)^2

    # Heteroscedastic: higher fidelity (smaller σ) is more expensive; algorithm chooses σ
    f_noisy(x, σ) = f_true(x) + rand(rng, Normal(0, σ))

    bounds   = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]
    σ_levels = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    res = CalibrationCode.bayesopt_ucb_threshold(f_noisy;
        bounds=bounds,
        σ_levels=σ_levels,
        n_init=10,
        n_iter=80,
        κ=1.75,
        α=18.0,
        seed=seed
    )

    println("Recommended x = ", res.x_rec, "   GP-mean y ≈ ", res.y_rec)

    println("Distance to true optimum ≈ ",
        sqrt((res.x_rec[1]-0.2)^2 + (res.x_rec[2]+0.4)^2 + (res.x_rec[3]-0.7)^2))

    counts = CalibrationCode.count_noise_levels(res)
    for σ in res.σ_levels
        println("σ=$(σ): ", counts[σ])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
