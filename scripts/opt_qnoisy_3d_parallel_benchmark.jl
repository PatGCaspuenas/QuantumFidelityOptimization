import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Distributed
using Statistics

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
    end

    # true  → optimize Q_det (deterministic, no shots); false → optimize Q_varMS (noisy)
    optimize_det     = get(ENV, "BO_OPT_DET",   "false") == "true"   # BO_OPT_DET=true/false
    # Number of shots per acquisition call (ignored for acq in :variable mode, which uses n_floor)
    N_shots          = parse(Int, get(ENV, "BO_N_SHOTS",    "400"))   # BO_N_SHOTS=400
    # Number of initial design points
    n_initial_samples = parse(Int, get(ENV, "BO_N_INIT",   "12"))    # BO_N_INIT=12
    # How often to re-fit GP hyperparameters (every N iterations)
    hyper_every      = parse(Int, get(ENV, "BO_HYPER_EVERY","10"))    # BO_HYPER_EVERY=10
    # Number of parallel simulations
    num_sims         = parse(Int, get(ENV, "BO_NUM_SIMS",  "3"))     # BO_NUM_SIMS=40

    # Output file (written to data/)
    output_file_name =           get(ENV, "BO_OUTPUT_FILE", "benchmark_results.txt") # BO_OUTPUT_FILE=...

    # Early-stopping threshold in Q-space.
    #   "0.998"  → fixed value
    #   "auto"   → derive from N_shots as 1 − 1/N
    #   ""       → no threshold (run fixed iterations)
    _thresh_env      =           get(ENV, "BO_THRESH_Q",   "")        # BO_THRESH_Q=0.998/"auto"/""
    fidelity_threshold_Q = if isempty(_thresh_env)
        nothing
    elseif _thresh_env == "auto"
        1.0 - 1.0 / N_shots
    else
        parse(Float64, _thresh_env)
    end

    # Maximum BO iterations
    n_iter           = parse(Int, get(ENV, "BO_N_ITER",   "120"))     # BO_N_ITER=120
    # MLE multi-start restarts per hyper fit (final fit uses n_restarts+2)
    n_restarts       = parse(Int, get(ENV, "BO_N_RESTARTS",  "6"))    # BO_N_RESTARTS=6

    # Shot-allocation modes:
    #   use_variable_mode = false, decide_N_shot = false
    #       → fixed n_shots per acquisition
    #   use_variable_mode = true
    #       → evaluate acquisition points with n_floor;
    #         adaptively measure x_rec and stop early if threshold is confirmed
    #   decide_N_shot = true
    #       → choose the number of shots for each acquisition point
    #         from the GP posterior uncertainty and threshold distance

    use_variable_mode = get(ENV, "BO_VAR_N_MODE", "false") == "true"        # BO_VAR_N_MODE=true/false
    decide_N_shot     = get(ENV, "BO_DECIDE_N_SHOT", "false") == "true"    # BO_DECIDE_N_SHOT=true/false
    # Minimum shots for variable mode (also used for acquisition point)
    n_floor          = parse(Int, get(ENV, "BO_N_FLOOR",  "50"))      # BO_N_FLOOR=50
    # Maximum shots for :variable mode
    n_max_shots      = parse(Int, get(ENV, "BO_N_MAX",  "2000"))      # BO_N_MAX=2000

    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    span_kHz = 2.0
    span_fcl = span_kHz * 1e3 * 2π
    span_fsb = span_kHz * 1e3 * 2π
    span_A = 1.2 * A0 - A0

    @everywhere const t = $t
    @everywhere const f_cl0 = $f_cl0
    @everywhere const f_sb0 = $f_sb0
    @everywhere const A0 = $A0
    @everywhere const span_fcl = $span_fcl
    @everywhere const span_fsb = $span_fsb
    @everywhere const span_A = $span_A
    @everywhere const optimize_det = $optimize_det

    @everywhere u_to_params(u) = (f_cl0 + span_fcl * u[1],
        f_sb0 + span_fsb * u[2],
        A0 + span_A * u[3])

    # Binomial noise model: σy = √(Q(1-Q)/N).
    # Returns σy=0 for deterministic mode (σy is ignored by GP in Q_det runs).
    @everywhere function sigma_y_fun(Q_raw::Float64, N::Int)
        optimize_det && return 0.0
        one_minus_Q = max(1.0 - Q_raw, 1e-15)
        return sqrt(max(Q_raw, 0.0) * one_minus_Q / N)
    end

    # Returns (y, σy): linear Q scale, binomial noise.
    @everywhere function Q_fun(u, N::Int)
        fcl, fsb, A = u_to_params(u)
        Q_raw = if optimize_det
            clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
        else
            CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2)
        end
        return (clamp(Q_raw, 0.0, 1.0), sigma_y_fun(Q_raw, N))
    end

    @everywhere function Q_true(u)
        fcl, fsb, A = u_to_params(u)
        return clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    α = 1.5
    κ = 1.9

    fidelity_threshold = fidelity_threshold_Q  # linear Q scale (no log transform)

    optimization_mode = optimize_det ? "Q_det" : "Q_noisy"

    println("=== Starting Parallel Simulations (Random Seeds) ===")
    println("Optimization mode: $optimization_mode")
    println("N_shots = $N_shots, noise model = binomial")
    println("Fixed α = $α, κ = $κ")
    println("Number of simulations: $num_sims, n_iter = $n_iter, n_restarts = $n_restarts")
    println("Fidelity threshold: $(fidelity_threshold_Q === nothing ? "none (fixed $n_iter iterations)" : "Q* = $fidelity_threshold_Q (linear)")")
    println("use_variable_mode = $use_variable_mode$(use_variable_mode ? ", n_floor=$n_floor, n_max=$n_max_shots" : "")")
    println("decide_N_shot = $decide_N_shot$(decide_N_shot ? ", n_floor=$n_floor, n_max=$n_max_shots" : "")")

    start_time = time()

    # Fixed seeds from the wideOmega N400 benchmark run (reproducible baseline).
    # If num_sims > length, extra seeds are drawn deterministically from MersenneTwister(1).
    _fixed_seeds = [
        714078, 849665, 670733, 400294, 909858, 473966, 981559, 318670, 225142, 359405,
        250215, 558664, 438880,   5937, 615903, 150574, 963284, 473867, 150918, 955377,
        902578,  61646, 197255, 462583, 184672, 831702, 720308,  16729, 387749, 215846,
        312561, 749598, 631837, 746550, 709734, 181983, 279125, 965652, 419030, 888571,
    ]
    random_seeds = if num_sims <= length(_fixed_seeds)
        _fixed_seeds[1:num_sims]
    else
        extra = rand(MersenneTwister(1), 1:1000000, num_sims - length(_fixed_seeds))
        vcat(_fixed_seeds, extra)
    end

    _random_seeds        = random_seeds
    _n_iter              = n_iter
    _use_variable_mode     = use_variable_mode
    _n_floor             = n_floor
    _n_max_shots         = n_max_shots
    _n_restarts          = n_restarts
    _learn_noise_scale   = !optimize_det

    results_grid = pmap(1:num_sims; batch_size=1) do sim_idx
        try
            seed = _random_seeds[sim_idx]
            println("Starting simulation $sim_idx...")
            flush(stdout)
            res = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
                bounds=bounds,
                n_shots=N_shots,
                n_init=n_initial_samples,
                n_iter=_n_iter,
                κ=κ,
                α=α,
                seed=seed,
                maximize=true,
                fidelity_threshold=fidelity_threshold,
                explore_frac=0.0,
                hyper_every=hyper_every,
                learn_noise_scale=_learn_noise_scale,
                n_restarts=_n_restarts,
                use_variable_mode=_use_variable_mode,
                decide_N_shot=decide_N_shot,
                n_floor=_n_floor,
                n_max_shots=_n_max_shots,
                verbose=true
            )

            fcl, fsb, A = u_to_params(res.x_rec)
            Q_det_val = clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            n_iters = res.n_iter_actual > 0 ? res.n_iter_actual : _n_iter

            Q_noisy_val = res.y_last   # linear Q scale
            Q_rec_val   = res.y_rec    # GP posterior mean at recommended point

            total_shots_used = res.total_shots
            n_train = length(res.σy)

            println("Simulation $sim_idx → Q_det = $(round(Q_det_val, digits=4)), Q_rec = $(round(Q_rec_val, digits=4)), iterations: $n_iters, total_shots: $total_shots_used")
            flush(stdout)

            (
                sim_idx=sim_idx,
                seed=seed,
                x_rec=res.x_rec,
                Q_rec=Q_rec_val,
                Q_det=Q_det_val,
                Q_noisy=Q_noisy_val,
                n_iterations=n_iters,
                total_shots=total_shots_used,
                n_train=n_train,
                ℓ_final=res.ℓ_final,
                σf_final=res.σf_final,
                c_final=res.c_final,
            )
        catch e
            println("ERROR in simulation $sim_idx : ")
            println("Error type: $(typeof(e))")
            println("Error message: $e")
            Base.showerror(stdout, e, catch_backtrace())
            println()
            flush(stdout)
            rethrow()
        end
    end

    Q_det_values = [r.Q_det for r in results_grid]

    best_idx = argmax(Q_det_values)
    best_result = results_grid[best_idx]

    avg_Q_det = mean(Q_det_values)
    med_Q_det = median(Q_det_values)
    min_Q_det = minimum(Q_det_values)
    max_Q_det = maximum(Q_det_values)
    std_Q_det = std(Q_det_values)

    iter_values = [r.n_iterations for r in results_grid]
    avg_iterations = mean(iter_values)
    med_iterations = median(iter_values)
    std_iterations = std(iter_values)

    total_train = sum(r.n_train for r in results_grid)

    println("\n=== RESULTS SUMMARY ===")
    println("Optimization mode: $optimization_mode")
    println("Best Q_det = ", best_result.Q_det, " (Simulation ", best_result.sim_idx, ")")
    println("Average Q_det = ", avg_Q_det, " ± ", std_Q_det)
    println("Median Q_det = ", med_Q_det)
    println("Min Q_det = ", min_Q_det)
    println("Max Q_det = ", max_Q_det)
    println("Average iterations = ", avg_iterations, " ± ", std_iterations)
    println("Median iterations = ", med_iterations)
    println("Total training points across all sims = $total_train")
    println("\nBest noisy Q (last obs) = ", best_result.Q_noisy)
    println("Best GP-recommended Q   = ", best_result.Q_rec)
    println("\nBest final GP hyperparameters:")
    println("  ℓ  = ", round.(best_result.ℓ_final, digits=4))
    println("  σf = ", round(best_result.σf_final, digits=4))
    println("  c  = ", round(best_result.c_final,  digits=4))
    println("\nBest u_rec = ", best_result.x_rec)

    fcl_rec, fsb_rec, A_rec = u_to_params(best_result.x_rec)
    println("\n=== BEST RESULT PHYSICAL PARAMETERS ===")
    println("Recommended f_cl = ", fcl_rec)
    println("Recommended f_sb = ", fsb_rec)
    println("Recommended A    = ", A_rec)
    println("Baseline   f_cl = ", f_cl0, "  f_sb = ", f_sb0, "  A = ", A0)

    elapsed_seconds = time() - start_time
    elapsed_total = round(Int, elapsed_seconds)
    elapsed_hours = elapsed_total ÷ 3600
    elapsed_minutes = (elapsed_total % 3600) ÷ 60
    elapsed_secs = elapsed_total % 60
    elapsed_hms = string(elapsed_hours, ":", lpad(string(elapsed_minutes), 2, '0'), ":", lpad(string(elapsed_secs), 2, '0'))

    output_file = joinpath(@__DIR__, "..", "data", output_file_name)
    open(output_file, "w") do io
        println(io, "=== BENCHMARK RESULTS ===")
        println(io, "Optimization mode: $optimization_mode, N_shots = $N_shots, noise model = binomial")
        println(io, "Fixed α = $α, κ = $κ")
        println(io, "Number of simulations: $num_sims, n_iter = $n_iter, n_restarts = $n_restarts")
        println(io, "Fidelity threshold: $(fidelity_threshold_Q === nothing ? "none" : "Q* = $fidelity_threshold_Q")")
        println(io, "use_variable_mode = $use_variable_mode$(use_variable_mode ? ", n_floor=$n_floor, n_max=$n_max_shots" : "")")
        println(io, "Elapsed time: $elapsed_hms")
        println(io, "")
        println(io, "=== RESULTS SUMMARY ===")
        println(io, "Best Q_det = $(best_result.Q_det) (Simulation $(best_result.sim_idx), Seed $(best_result.seed), Iterations $(best_result.n_iterations))")
        println(io, "Average Q_det = $avg_Q_det ± $std_Q_det")
        println(io, "Median Q_det = $med_Q_det")
        println(io, "Min Q_det = $min_Q_det")
        println(io, "Max Q_det = $max_Q_det")
        println(io, "Average iterations = $avg_iterations ± $std_iterations")
        println(io, "Median iterations = $med_iterations")
        println(io, "")
        println(io, "Total training points across all sims = $total_train")
        println(io, "")
        println(io, "Best noisy Q (last obs) = $(best_result.Q_noisy)")
        println(io, "Best GP-recommended Q   = $(best_result.Q_rec)")
        println(io, "Best final GP hyperparameters:")
        println(io, "  ℓ  = $(round.(best_result.ℓ_final, digits=4))")
        println(io, "  σf = $(round(best_result.σf_final, digits=4))")
        println(io, "  c  = $(round(best_result.c_final,  digits=4))")
        println(io, "Best u_rec = $(best_result.x_rec)")
        println(io, "")
        println(io, "=== BEST RESULT PHYSICAL PARAMETERS ===")
        println(io, "Recommended f_cl = $fcl_rec")
        println(io, "Recommended f_sb = $fsb_rec")
        println(io, "Recommended A    = $A_rec")
        println(io, "Baseline   f_cl = $f_cl0, f_sb = $f_sb0, A = $A0")
        println(io, "")
        println(io, "=== All Results ===")
        ℓ_headers = join(["ℓ$i" for i in 1:length(bounds)], "\t")
        println(io, "Sim\tSeed\tIterations\tTotalShots\tNTrain\tQ_det\tQ_noisy\tQ_rec\t$ℓ_headers\tσf\tc")
        for r in results_grid
            ℓ_str = join(round.(r.ℓ_final, digits=4), "\t")
            println(io, "$(r.sim_idx)\t$(r.seed)\t$(r.n_iterations)\t$(r.total_shots)\t$(r.n_train)\t$(r.Q_det)\t$(r.Q_noisy)\t$(r.Q_rec)\t$ℓ_str\t$(round(r.σf_final,digits=4))\t$(round(r.c_final,digits=4))")
        end
    end

    println("\nResults written to: $output_file")

finally
    rmprocs(workers())
end
