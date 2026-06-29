import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using .CalibrationCode

function main(; seed=1)
    t = 100.0
    base = CalibrationCode.ideal(t)

    # 3D: optimize f_cl, f_sb, A jointly. Search in u ∈ [-1,1]³, mapped to ±20%
    # around the ideal params (matches the old physical bounds).
    span_fcl = 0.2 * base.f_cl
    span_fsb = 0.2 * base.f_sb
    span_A   = 0.2 * base.A
    function f_noisy(u, n)
        f_cl = base.f_cl + span_fcl * u[1]
        f_sb = base.f_sb + span_fsb * u[2]
        A    = base.A    + span_A   * u[3]
        y = CalibrationCode.Q_noisy(t, f_cl, f_sb, A; N=n, phase_grid=0.0:0.2:π)
        σy = sqrt(max(y * (1.0 - y), 0.0) / Float64(n))
        return y, σy
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    res = CalibrationCode.bayesopt_ucb(f_noisy;
        bounds=bounds,
        n_shots=100,
        n_init=10,
        n_iter=100,
        seed=seed
    )

    f_cl_rec = base.f_cl + span_fcl * res.x_rec[1]
    f_sb_rec = base.f_sb + span_fsb * res.x_rec[2]
    A_rec    = base.A    + span_A   * res.x_rec[3]
    println("Ideal (f_cl, f_sb, A) = (", base.f_cl, ", ", base.f_sb, ", ", base.A, ")")
    println("Recommended (f_cl, f_sb, A) = (", f_cl_rec, ", ", f_sb_rec, ", ", A_rec,
            ")   GP-mean y ≈ ", res.y_rec)
end

if !isinteractive()
    main()
end
