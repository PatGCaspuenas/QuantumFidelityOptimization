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
        
        using LinearAlgebra
        LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS thread thrashing on workers
        
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
                          
        # Define the N values we want to sweep over
        const N_vals = 50:50:1000

        # Evaluation function for the worker
        function evaluate_point(pt)
            u1, u2 = pt
            u3 = 0.0 # Fixed Z-plane (A at baseline)
            fcl, fsb, A = u_to_params([u1, u2, u3])
            
            # Deterministic Q
            Q_det_val = clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            
            # Noisy Q for each N configuration
            Q_noisy_vals = Float64[]
            for N in N_vals
                # SWAPPED: Now using Q_noisy instead of Q_varMS
                push!(Q_noisy_vals, CalibrationCode.Q_noisy(t, fcl, fsb, A; N=N))
            end
            
            return (u1, u2, Q_det_val, Q_noisy_vals)
        end
    end

    # ============================================================
    # GRID CONFIGURATION
    # ============================================================
    grid_res = 30                              # 30x30 grid points
    u_grid = range(-1.0, 1.0, length=grid_res) # The bounds of your normalized space
    output_file_name = "response_surface_animation_data_1gate.csv"
    
    println("=== Generating 2D Z=0 Slice for N-Shot Animation ===")
    println("Grid resolution: $grid_res x $grid_res")
    println("Sweeping N from $(first(N_vals)) to $(last(N_vals)) in steps of $(step(N_vals))")
    
    # 1. Build the list of all XY points to evaluate (Z/u3 is fixed at 0.0 internally)
    eval_points = []
    for u1 in u_grid, u2 in u_grid
        push!(eval_points, (u1, u2))
    end

    total_points = length(eval_points)
    println("Total points to evaluate: $total_points")
    println("Dispatching to $(nworkers()) workers...")
    
    start_time = time()

    # 2. Evaluate in parallel
    results_grid = pmap(evaluate_point, eval_points; batch_size=20)

    elapsed = time() - start_time
    println("Evaluation finished in $(round(elapsed, digits=2)) seconds.")
    
    # 3. Write structured results to CSV
    output_file = joinpath(@__DIR__, "..", "data", output_file_name)
    println("Writing data to $output_file...")
    
    # Using raw file I/O
    open(output_file, "w") do io
        # Dynamic Header based on N_vals
        n_headers = join(["Q_noisy_N$N" for N in N_vals], ",")
        println(io, "u1_fcl,u2_fsb,Q_det,$n_headers")
        
        for res in results_grid
            u1, u2, Q_det_val, Q_noisy_vals = res
            
            # Format the array of noisy values into a comma-separated string
            noisy_vals_str = join([@sprintf("%.6f", val) for val in Q_noisy_vals], ",")
            
            # Format row: (u1, u2, Q_det, Q_noisy_N50, Q_noisy_N100, ...)
            str_row = @sprintf("%.4f,%.4f,%.6f,%s", u1, u2, Q_det_val, noisy_vals_str)
            println(io, str_row)
        end
    end

    println("\nDone! Data successfully structured for animation plotting.")

catch e
    println("ERROR generating slices: ")
    Base.showerror(stdout, e, catch_backtrace())
    rethrow()
finally
    rmprocs(workers())
end