using Random
using Distributions
using Plots
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

include(joinpath(@__DIR__, "..", "scripts", "plots_cokrig.jl"))  # plot2d_cokrig / animate2d_cokrig

function main(; seed=2)
    rng = MersenneTwister(seed)

    # High-fidelity truth
    f_hi(x) = 1.0 - (x[1]-0.3)^2 - (x[2]+0.2)^2

    # Build 3 fidelities (low/mid/high) as biased/noisy versions of f_hi
    f_mid(x) = 0.93*f_hi(x) + 0.04*sin(5x[1]) - 0.02
    f_lo(x)  = 0.88*f_hi(x) + 0.07*sin(6x[1]) - 0.06

    σ = [0.05, 0.02, 0.01]            # noise per fidelity (1..3)
    costs = [1.0, 5.0, 20.0]          # cost per fidelity (1..3)

    # N-fidelity callable: f(x, level)
    function f_level(x, level::Int)
        level == 1 && return f_lo(x)  + rand(rng, Normal(0, σ[1]))
        level == 2 && return f_mid(x) + rand(rng, Normal(0, σ[2]))
        level == 3 && return f_hi(x)  + rand(rng, Normal(0, σ[3]))
        throw(ArgumentError("invalid level=$level"))
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    # This assumes you implemented the N-fidelity co-kriging BO:
    #   res = mfcokrig_bayesopt(f_level; bounds, costs, ...)
    # returning MFCoKrigResult with Xs, ys, x_rec, y_rec, etc.
    res = CalibrationCode.mfcokrig_bayesopt(f_level;
        bounds=bounds,
        costs=costs,
        n_init=18,
        n_iter=60,
        M=4000,
        xi=0.01,
        seed=seed,
        optimize_hypers=false
    )

    println("Recommended x = ", res.x_rec, "   predicted mean at highest fidelity ≈ ", res.y_rec)
    println("True high-fidelity at x_rec = ", f_hi(res.x_rec))

    # Posterior contours at highest fidelity + show all samples
    p = plot2d_cokrig(res; level=3, nx=50, ny=50, show_std=false, show_points=true)
    savefig(p, "toy_cokrig_2d_posterior_hi.png")
    println("Saved -> toy_cokrig_2d_posterior_hi.png")

    # Sample scatter by fidelity
    p2 = scatter(; xlabel="x₁", ylabel="x₂", title="Co-kriging MF-BO samples (3 fidelities)")
    for m in 1:3
        Xm = res.Xs[m]
        size(Xm,2) == 0 && continue
        scatter!(p2, Xm[1,:], Xm[2,:]; ms=4, label="level $m")
    end
    scatter!(p2, [res.x_rec[1]], [res.x_rec[2]]; ms=7, label="recommended")
    savefig(p2, "toy_cokrig_samples.png")
    println("Saved -> toy_cokrig_samples.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
