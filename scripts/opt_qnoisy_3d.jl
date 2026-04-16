import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using StatsBase
using .CalibrationCode

function main()
    # Baseline
    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    # spans around baseline
    span_fcl = 5e4
    span_fsb = 5e4
    span_A   = 8e4

    # u ∈ [-1,1]^3 -> physical params
    u_to_params(u) = (f_cl0 + span_fcl*u[1],
                      f_sb0 + span_fsb*u[2],
                      A0    + span_A  *u[3])

    # noise knob -> shots (tunable)
    function N_from_sigma(σ::Float64)
        N = round(Int, 1 / (σ^2))
        return clamp(N, 20, 3000)
    end

    function Q_fun(u, σ)
        fcl, fsb, A = u_to_params(u)
        return CalibrationCode.Q_noisy(t, fcl, fsb, A; N=N_from_sigma(σ))
    end

    function Q_true(u)
        fcl, fsb, A = u_to_params(u)
        return CalibrationCode.Q_det(t, fcl, fsb, A)
    end

    σ_levels = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]
    bounds   = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    resH = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
        bounds=bounds,
        σ_levels=σ_levels,
        n_init=12,
        n_iter=120,
        κ=2.0,
        α=0.5,
        seed=2
    )

    println("\n=== Hetero BO on Q_noisy (3D: f_cl,f_sb,A) ===")
    println("u_rec = ", resH.x_rec)
    println("GP-mean y ≈ ", resH.y_rec, "    Q_det at u_rec = ", Q_true(resH.x_rec))

    fcl_rec, fsb_rec, A_rec = u_to_params(resH.x_rec)
    println("Recommended f_cl = ", fcl_rec)
    println("Recommended f_sb = ", fsb_rec)
    println("Recommended A    = ", A_rec)
    println("Baseline   f_cl = ", f_cl0, "  f_sb = ", f_sb0, "  A = ", A0)

    counts = CalibrationCode.count_noise_levels(resH)
    println("\nNoise usage:")
    for σ in resH.σ_levels
        println("σ=$(σ): ", counts[σ], "   (N=$(N_from_sigma(σ)))")
    end

    return resH
end

if !isinteractive()
    main()
end
