using Random
using Distributions
using Plots

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode


include(joinpath(@__DIR__, "..", "scripts", "plots_standard.jl"))  # provides plot2d_standard/animate2d_standard

function main(; seed=1, σ=0.01)
    rng = MersenneTwister(seed)
    t = 100
    init_param = CalibrationCode.ideal(t; dt=0.1)  # warmup

    f(x) = CalibrationCode.Q_noisy(t, x[1], x[2], x[3]; N=100, phase_grid=0:0.2:π, dt=0.1)
    

    bounds = [(init_param[2]*.8, init_param[2]*1.2), (init_param[3]*.8, init_param[3]*1.2), (init_param[4]*.8, init_param[4]*1.2)]

    out = CalibrationCode.bayesopt(f;
        bounds=bounds,
        n_init=6,
        n_iter=100,
        xi=0.01,
        maximize=true,
        seed=1,
        obs_noise=σ
    )

    if out isa Tuple
        res, x_rec, y_rec = out
    else
        res = out
        best_idx = argmax(res.y)
        x_rec = vec(res.X[:, best_idx])
        y_rec = res.y[best_idx]
    end

    # Works whether bayesopt returns BOResult OR (res, x_rec, y_rec)
    if out isa Tuple
        res, x_rec, y_rec = out
    else
        res = out
        best_idx = argmax(res.y)
        x_rec = vec(res.X[:, best_idx])
        y_rec = res.y[best_idx]   # fallback: best observed
    end

    best_idx = argmax(res.y)
    x_best = vec(res.X[:, best_idx])
    println("Best observed x = ", x_best, "   best observed y = ", res.y[best_idx])
    println("Recommended x (posterior mean argmax) = ", x_rec, "   y ≈ ", y_rec)

    println("ideal values: f_cl = ", init_param[2], ", f_sb = ", init_param[3], ", A = ", init_param[4])
end

main()

