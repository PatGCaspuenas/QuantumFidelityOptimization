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
        using Random

        # Base parameters
        const t = 100.0
        const base = CalibrationCode.ideal(t)
        const f_cl0 = base.f_cl
        const f_sb0 = base.f_sb
        const A0 = base.A

        # Span parameters
        const span_kHz = 2.0
        const span_fcl = span_kHz * 1e3 * 2π
        const span_fsb = span_kHz * 1e3 * 2π
        const span_A = 1.2 * A0 - A0

        u_to_params(u) = (f_cl0 + span_fcl * u[1],
                          f_sb0 + span_fsb * u[2],
                          A0 + span_A * u[3])
                          
        # Evaluation function for the worker
        function evaluate_point(pt)
            plane_name, slice_val, u1, u2, u3 = pt
            fcl, fsb, A = u_to_params([u1, u2, u3])
            
            # Deterministic and noisy Q
            Q_det_val = clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            Q_varMS_val = CalibrationCode.Q_varMS(t, fcl, fsb, A; N=1000, numMS=2)
            
            return (plane_name, slice_val, u1, u2, u3, Q_det_val, Q_varMS_val)
        end
    end

    # ============================================================
    # GRID CONFIGURATION
    # ============================================================
    grid_res = 30                              # 30x30 grid points per 2D slice
    u_grid = range(-1.0, 1.0, length=grid_res) # The bounds of your normalized space
    slice_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]   # Positions of the slice planes
    output_file_name = "response_surface_slices_N1000.csv"
    
    println("=== Generating 2D Slices for the Response Surface ===")
    println("Grid resolution per slice: $grid_res x $grid_res")
    println("Slice plane locations: ", slice_vals)
    
    # 1. Build the list of all points to evaluate
    eval_points = []

    # XY Planes (Fix u3/A, Vary u1/fcl and u2/fsb)
    for u3 in slice_vals, u1 in u_grid, u2 in u_grid
        push!(eval_points, ("XY", u3, u1, u2, u3))
    end

    # XZ Planes (Fix u2/fsb, Vary u1/fcl and u3/A)
    for u2 in slice_vals, u1 in u_grid, u3 in u_grid
        push!(eval_points, ("XZ", u2, u1, u2, u3))
    end

    # YZ Planes (Fix u1/fcl, Vary u2/fsb and u3/A)
    for u1 in slice_vals, u2 in u_grid, u3 in u_grid
        push!(eval_points, ("YZ", u1, u1, u2, u3))
    end

    total_points = length(eval_points)
    println("Total points to evaluate: $total_points")
    println("Dispatching to $(nworkers()) workers...")
    
    start_time = time()

    # 2. Evaluate in parallel
    results_grid = pmap(evaluate_point, eval_points; batch_size=50)

    elapsed = time() - start_time
    println("Evaluation finished in $(round(elapsed, digits=2)) seconds.")
    
    # 3. Write structured results to CSV
    output_file = joinpath(@__DIR__, "..", "data", output_file_name)
    println("Writing data to $output_file...")
    
    # Using raw file I/O to avoid requiring CSV.jl / DataFrames.jl as a dependency
    open(output_file, "w") do io
        # Header
        println(io, "plane,slice_fixed_val,u1_fcl,u2_fsb,u3_A,Q_det,Q_varMS")
        
        for res in results_grid
            # Format row: (plane, slice_val, u1, u2, u3, Q_det, Q_varMS)
            str_row = @sprintf("%s,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f", 
                               res[1], res[2], res[3], res[4], res[5], res[6], res[7])
            println(io, str_row)
        end
    end

    println("\nDone! Data successfully structured for plotting.")

catch e
    println("ERROR generating slices: ")
    Base.showerror(stdout, e, catch_backtrace())
    rethrow()
finally
    rmprocs(workers())
end