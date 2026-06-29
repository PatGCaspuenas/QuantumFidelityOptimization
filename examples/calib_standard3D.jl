import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using Distributions
using .CalibrationCode

function main(; seed=1, σ=0.01)
    t = 100.0
    base = CalibrationCode.ideal(t)

    # 3D: optimize f_cl, f_sb, A jointly
    f(x) = CalibrationCode.Q_noisy(t, x[1], x[2], x[3]; N=100, phase_grid=0.0:0.2:π)

    bounds = [
        (base.f_cl * 0.8, base.f_cl * 1.2),
        (base.f_sb * 0.8, base.f_sb * 1.2),
        (base.A   * 0.8, base.A   * 1.2),
    ]

    res = CalibrationCode.bayesopt(f;
        bounds=bounds,
        n_init=10,
        n_iter=100,
        xi=0.01,
        maximize=true,
        seed=seed,
        obs_noise=σ
    )

    best_idx = argmax(res.y)
    x_rec = vec(res.X[:, best_idx])
    println("Ideal (f_cl, f_sb, A) = (", base.f_cl, ", ", base.f_sb, ", ", base.A, ")")
    println("Best observed = ", x_rec, "   y = ", res.y[best_idx])
end

if !isinteractive()
    main()
end
