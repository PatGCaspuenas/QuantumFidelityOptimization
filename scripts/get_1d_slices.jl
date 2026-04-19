import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf

if nprocs() == 1
    n_workers_cfg = get(ENV, "BO_N_WORKERS", "")
    n_workers_add = isempty(n_workers_cfg) ? max(1, Sys.CPU_THREADS - 1) : parse(Int, n_workers_cfg)
    addprocs(n_workers_add)
end

try
    @everywhere begin
        import Pkg
        Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
        include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

        const t = 100.0
        const base = CalibrationCode.ideal(t)
        const f_cl0 = base.f_cl
        const f_sb0 = base.f_sb
        const A0    = base.A

        const span_kHz = 2.0
        const span_fcl = span_kHz * 1e3 * 2π
        const span_fsb = span_kHz * 1e3 * 2π
        const span_A   = 1.2 * A0 - A0

        u_to_params(u) = (f_cl0 + span_fcl * u[1],
                          f_sb0 + span_fsb * u[2],
                          A0    + span_A   * u[3])

        # pt = (slice_name, u_varying, u1, u2, u3)
        function evaluate_point(pt)
            slice_name, u_val, u1, u2, u3 = pt
            fcl, fsb, A = u_to_params([u1, u2, u3])
            Q_det_val   = clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            Q_varMS_val = CalibrationCode.Q_varMS(t, fcl, fsb, A; N=1000, numMS=2)
            return (slice_name, u_val, u1, u2, u3, Q_det_val, Q_varMS_val)
        end
    end

    # ============================================================
    # BUILD EVAL POINTS
    # ============================================================
    n_pts  = 500
    u_grid = range(-1.0, 1.0, length=n_pts)

    eval_points = []

    # Slice 1: u2=u3=0, u1 varies
    for u1 in u_grid
        push!(eval_points, ("u2u3=0", u1, u1, 0.0, 0.0))
    end

    # Slice 2: u1=u3=0, u2 varies
    for u2 in u_grid
        push!(eval_points, ("u1u3=0", u2, 0.0, u2, 0.0))
    end

    println("Evaluating $(length(eval_points)) points on $(nworkers()) workers ...")
    flush(stdout)

    start_time = time()
    results = pmap(evaluate_point, eval_points; batch_size=20)
    elapsed = time() - start_time
    println("Done in $(round(elapsed, digits=2)) s.")

    # ============================================================
    # SAVE CSV
    # ============================================================
    output_file = joinpath(@__DIR__, "..", "data", "response_surface_1d_slices_N1000.csv")
    open(output_file, "w") do io
        println(io, "slice,u_val,u1_fcl,u2_fsb,u3_A,Q_det,Q_varMS")
        for r in results
            println(io, @sprintf("%s,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f",
                                 r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
        end
    end
    println("Saved → $output_file")

catch e
    println("ERROR: ")
    Base.showerror(stdout, e, catch_backtrace())
    rethrow()
finally
    rmprocs(workers())
end
