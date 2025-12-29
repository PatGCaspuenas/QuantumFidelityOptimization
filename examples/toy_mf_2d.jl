using Random
using Distributions
using Plots

using CalibrationCode
include(joinpath(@__DIR__, "..", "scripts", "plots_mf.jl"))   # plot2d_mf / animate2d_mf (optional)

function main(; seed=1)
    rng = MersenneTwister(seed)

    # High-fidelity (expensive) truth
    f_hi(x) = 1.0 - (x[1]-0.3)^2 - (x[2]+0.2)^2

    # Mid fidelity: less biased than low, moderate noise
    f_mid(x) = 0.93*f_hi(x) + 0.04*sin(5x[1]) - 0.02

    # Low fidelity: more bias/warp, more noise
    f_lo(x)  = 0.85*f_hi(x) + 0.08*sin(6x[1]) - 0.05

    σ_lo  = 0.05
    σ_mid = 0.02
    σ_hi  = 0.01

    # Required signature: f(x, ℓ) where ℓ indexes levels
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

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    # 3 fidelities encoded as z in [0,1]
    z_levels = [0.0, 0.5, 1.0]
    costs    = [1.0, 4.0, 20.0]     # high fidelity most expensive

    res = CalibrationCode.bayesopt_mf(f_mf;
        bounds=bounds,
        z_levels=z_levels,
        costs=costs,
        n_init=12,
        n_iter=60,
        M=4000,
        xi=0.01,
        obs_noise=1e-3,
        optimize_hypers=false,
        seed=seed
    )

    println("Recommended x = ", res.x_rec, "   GP-mean at highest fidelity ≈ ", res.y_rec)
    println("True hi-fidelity value at x_rec = ", f_hi(res.x_rec))

    # --- Plot 1: posterior at highest fidelity (ℓ=3) + samples colored by z
    p1 = plot2d_mf(res; ℓ=3, f_true=f_hi, nx=50, ny=50, obs_noise=1e-3)
    savefig(p1, "toy_mf_2d_posterior_hi.png")
    println("Saved -> toy_mf_2d_posterior_hi.png")

    # --- Plot 2: sample locations by fidelity + recommended point
    Xa = res.Xa
    xs = Xa[1, :]
    ys = Xa[2, :]
    zs = Xa[3, :]

    p2 = scatter(; xlabel="x₁", ylabel="x₂", title="MF-BO samples (3 fidelities) and recommendation")
    for (ℓ, z) in enumerate(z_levels)
        idx = findall(i -> isapprox(zs[i], z; atol=1e-12, rtol=0), eachindex(zs))
        if !isempty(idx)
            scatter!(p2, xs[idx], ys[idx]; ms=4, label="ℓ=$ℓ (z=$z)")
        end
    end
    scatter!(p2, [res.x_rec[1]], [res.x_rec[2]]; ms=7, label="recommended")
    savefig(p2, "toy_mf_2d_samples.png")
    println("Saved -> toy_mf_2d_samples.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
