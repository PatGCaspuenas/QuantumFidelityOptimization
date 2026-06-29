import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using Random
using .CalibrationCode

function main(; seed=1)
    t = 100.0
    base = CalibrationCode.ideal(t)

    # 2D: optimize f_sb and A (f_cl fixed at ideal). Search in u ∈ [-1,1]²,
    # mapped to ±20% around the ideal params (matches the old physical bounds).
    span_fsb = 0.2 * base.f_sb
    span_A   = 0.2 * base.A
    function f_noisy(u, n)
        f_sb = base.f_sb + span_fsb * u[1]
        A    = base.A    + span_A   * u[2]
        y = CalibrationCode.Q_noisy(t, base.f_cl, f_sb, A; N=n, phase_grid=0.0:0.2:π)
        σy = sqrt(max(y * (1.0 - y), 0.0) / Float64(n))
        return y, σy
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    res = CalibrationCode.bayesopt_ucb(f_noisy;
        bounds=bounds,
        n_shots=100,
        n_init=6,
        n_iter=100,
        seed=seed
    )

    f_sb_rec = base.f_sb + span_fsb * res.x_rec[1]
    A_rec    = base.A    + span_A   * res.x_rec[2]
    println("Ideal (f_sb, A) = (", base.f_sb, ", ", base.A, ")")
    println("Recommended (f_sb, A) = (", f_sb_rec, ", ", A_rec, ")   GP-mean y ≈ ", res.y_rec)
end

if !isinteractive()
    main()
end
