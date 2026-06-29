import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using Distributions
using .CalibrationCode

function main(; seed=1, σ=0.01)
    t = 100.0
    base = CalibrationCode.ideal(t)

    # 2D: optimize f_sb and A with f_cl fixed at ideal
    f(x) = CalibrationCode.Q_noisy(t, base.f_cl, x[1], x[2]; N=100, phase_grid=0.0:0.2:π)

    bounds = [(base.f_sb * 0.8, base.f_sb * 1.2), (base.A * 0.8, base.A * 1.2)]

    res = CalibrationCode.bayesopt(f;
        bounds=bounds,
        n_init=6,
        n_iter=100,
        xi=0.01,
        maximize=true,
        seed=seed,
        obs_noise=σ
    )

    best_idx = argmax(res.y)
    x_rec = vec(res.X[:, best_idx])
    println("Ideal (f_sb, A) = (", base.f_sb, ", ", base.A, ")")
    println("Best observed (f_sb, A) = ", x_rec, "   y = ", res.y[best_idx])
end

if !isinteractive()
    main()
end
