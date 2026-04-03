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

    # ============================================================
    # CONFIG — edit these flags to change behaviour.
    # Any value can be overridden at runtime via ENV variables
    # (used by run_benchmark_sweep.jl). ENV keys listed in comments.
    # ============================================================
    use_log_fidelity     = get(ENV, "BO_LOG_FID",      "false") == "true"   # BO_LOG_FID=true/false
    optimize_det         = get(ENV, "BO_OPT_DET",      "false") == "true"   # BO_OPT_DET=true/false
    N_shots              = parse(Int,   get(ENV, "BO_N_SHOTS",    "400"))    # BO_N_SHOTS=400
    sigma_mode           = Symbol(      get(ENV, "BO_SIGMA_MODE", "simple")) # BO_SIGMA_MODE=simple/binomial
    use_pretrained       = get(ENV, "BO_USE_PRETRAIN",  "false") == "true"  # BO_USE_PRETRAIN=true/false
    pretrain_file_name   =              get(ENV, "BO_PRETRAIN_FILE", "pretrained_theta_3d.jl") # BO_PRETRAIN_FILE=filename
    freeze_mode_str      =              get(ENV, "BO_FREEZE_MODE", "none")  # BO_FREEZE_MODE=none/lengthscales/lengthscales_from/all
    n_freeze_iters       = parse(Int,   get(ENV, "BO_N_FREEZE",   "20"))    # BO_N_FREEZE=20
    n_initial_samples    = parse(Int,   get(ENV, "BO_N_INIT",     "12"))    # BO_N_INIT=12
    hyper_every          = parse(Int,   get(ENV, "BO_HYPER_EVERY","10"))    # BO_HYPER_EVERY=10
    num_sims             = parse(Int,   get(ENV, "BO_NUM_SIMS",   "40"))    # BO_NUM_SIMS=40
    output_file_name     =              get(ENV, "BO_OUTPUT_FILE", "benchmark_results_N400.txt") # BO_OUTPUT_FILE=...
    # Sampling strategies: :random or :sobol (Sobol + Owen scramble)
    #   init_sampling → how to draw the n_init design points
    #   acq_sampling  → how to draw the M_acq acquisition candidates each iteration
    init_sampling        = Symbol(      get(ENV, "BO_INIT_SAMPLING", "random")) # BO_INIT_SAMPLING=random/sobol
    acq_sampling         = Symbol(      get(ENV, "BO_ACQ_SAMPLING",  "random")) # BO_ACQ_SAMPLING=random/sobol

    # Early-stopping threshold in Q-space (e.g. "0.998"), or "" to derive from N_shots.
    _thresh_env          =              get(ENV, "BO_THRESH_Q",   "")       # BO_THRESH_Q=0.998 or ""
    fidelity_threshold_Q = isempty(_thresh_env) ? nothing : parse(Float64, _thresh_env)
    # Minimum iterations before early stopping is allowed (GP needs data to be trustworthy).
    min_iter             = parse(Int,   get(ENV, "BO_MIN_ITER",          "1"))    # BO_MIN_ITER=1
    # Number of independent checks before declaring convergence: 1 (single) or 2 (double).
    n_checks             = parse(Int,   get(ENV, "BO_N_CHECKS",          "1"))    # BO_N_CHECKS=1/2
    # Whether to add x_rec check evaluations to GP training data (always with distance guard).
    add_check_points     = get(ENV, "BO_ADD_CHECK_POINTS", "false") == "true"      # BO_ADD_CHECK_POINTS=true/false
    explore_frac         = parse(Float64, get(ENV, "BO_EXPLORE_FRAC",      "0.15")) # BO_EXPLORE_FRAC=0.15 (fraction of acquisition candidates drawn from exploration distribution instead of surrogate)
    # Fixed seed for init points: all runs share the same initial design when set.
    # "" or unset → init points vary with each run's seed (default behavior).
    _fixed_init_env      =              get(ENV, "BO_FIXED_INIT_SEED", "")        # BO_FIXED_INIT_SEED=1 or ""
    fixed_init_seed      = isempty(_fixed_init_env) ? nothing : parse(Int, _fixed_init_env)
    n_restarts           = parse(Int,   get(ENV, "BO_N_RESTARTS",       "6"))     # BO_N_RESTARTS=6 (MLE multi-start restarts per hyper fit; final fit uses n_restarts+2)
    # ============================================================

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

    @everywhere const use_log_fidelity = $use_log_fidelity
    @everywhere const optimize_det = $optimize_det
    @everywhere const sigma_mode = $(QuoteNode(sigma_mode))
    @everywhere u_to_params(u) = (f_cl0 + span_fcl * u[1],
        f_sb0 + span_fsb * u[2],
        A0 + span_A * u[3])

    # log10(Q): maps Q→1 to 0 (bounded above), noise shrinks at optimum — better than log10(1-Q).
    @everywhere function apply_log_fidelity(Q::Float64)
        use_log_fidelity ? log10(max(Q, 1e-15)) : Q
    end

    # Returns σy in GP space for a measurement with Q_raw fidelity using N shots.
    # :simple   → 1/√N  (worst-case, ignores Q)
    # :binomial → √(Q(1-Q)/N) in linear space;
    #             delta-method through y=log10(Q): dy/dQ = 1/(Q·ln10) → σ_y = σ_Q / (Q·ln10)
    #             (noise shrinks near Q=1, unlike the log10(1-Q) formulation)
    @everywhere function sigma_y_fun(Q_raw::Float64, N::Int)
        optimize_det && return 0.0
        if sigma_mode === :binomial
            one_minus_Q = max(1.0 - Q_raw, 1e-15)
            σ_Q = sqrt(max(Q_raw, 0.0) * one_minus_Q / N)
            return use_log_fidelity ? σ_Q / (max(Q_raw, 1e-15) * log(10)) : σ_Q
        else  # :simple
            return 1.0 / sqrt(N)
        end
    end

    # Returns (y, σy) where y is the (log-)fidelity and σy is the GP observation noise.
    # N is passed directly from bayesopt_ucb_threshold; no captured N_shots const needed.
    @everywhere function Q_fun(u, N::Int)
        fcl, fsb, A = u_to_params(u)
        Q_raw = if optimize_det
            CalibrationCode.Q_det(t, fcl, fsb, A)
        else
            CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2)
        end
        return (apply_log_fidelity(Q_raw), sigma_y_fun(Q_raw, N))
    end

    @everywhere function Q_true(u)
        fcl, fsb, A = u_to_params(u)
        Q = CalibrationCode.Q_det(t, fcl, fsb, A)
        return apply_log_fidelity(Q)
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    α = 1.5
    κ = 1.9
    # Resolve threshold: user value takes priority, otherwise derive from N_shots.
    Q_thresh = fidelity_threshold_Q !== nothing ? Float64(fidelity_threshold_Q) : 1.0 - 1.0 / N_shots
    fidelity_threshold = use_log_fidelity ? log10(max(Q_thresh, 1e-15)) : Q_thresh

    # --- Resolve pretrained θ and freeze_mode from config flags ---
    pretrain_file = joinpath(@__DIR__, "..", "data", pretrain_file_name)

    pretrained_θ = if use_pretrained
        if isfile(pretrain_file)
            include(pretrain_file)
            pretrained_theta_3d()
        else
            @warn "Pretrained file not found: $pretrain_file — falling back to no pretraining"
            nothing
        end
    else
        nothing
    end

    freeze_mode = Symbol(freeze_mode_str)

    optimization_mode = optimize_det ? "Q_det" : "Q_noisy"

    println("=== Starting Parallel Simulations (Random Seeds) ===")
    println("Optimization mode: $optimization_mode")
    println("N_shots = $N_shots, sigma_mode = $sigma_mode, use_log_fidelity = $use_log_fidelity")
    println("init_sampling = $init_sampling, acq_sampling = $acq_sampling, fixed_init_seed = $(fixed_init_seed === nothing ? "none (per-run)" : fixed_init_seed)")
    println("Fixed α = $α, κ = $κ")
    println("Number of simulations: $num_sims")
    thresh_src = fidelity_threshold_Q !== nothing ? "user-specified" : "derived from N_shots"
    println("Fidelity threshold: Q* = $Q_thresh ($thresh_src) → GP value = $fidelity_threshold $(use_log_fidelity ? "(log10 scale)" : "(linear scale)")")
    println("Min iterations before early stopping: $min_iter, n_checks=$n_checks, add_check_points=$add_check_points")
    println("freeze_mode = $freeze_mode, n_freeze_iters = $n_freeze_iters, use_pretrained = $use_pretrained")
    println("n_restarts = $n_restarts (BO loop), $(n_restarts + 2) (final fit)")
    println("Maximize: true (log10(Q) mode also maximizes, approaching 0 from below)")

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

    # Capture all config as locals so pmap serializes values, not globals.
    _explore_frac = explore_frac
    _pretrained_θ = pretrained_θ
    _freeze_mode = freeze_mode
    _n_freeze_iters = n_freeze_iters
    _random_seeds = random_seeds
    _explore_frac = explore_frac
    _init_sampling = init_sampling
    _acq_sampling  = acq_sampling
    _min_iter          = min_iter
    _n_checks          = n_checks
    _add_check_points  = add_check_points
    _fixed_init_seed   = fixed_init_seed
    _n_restarts        = n_restarts

    results_grid = pmap(1:num_sims; batch_size=1) do sim_idx
        try
            seed = _random_seeds[sim_idx]
            println("Starting simulation...")
            flush(stdout)
            res = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
                bounds=bounds,
                n_shots=N_shots,
                sigma_mode=sigma_mode,
                n_init=n_initial_samples,
                n_iter=120,
                κ=κ,
                α=α,
                seed=seed,
                maximize=true,
                fidelity_threshold=fidelity_threshold,
                min_iter=_min_iter,
                n_checks=_n_checks,
                add_check_points=_add_check_points,
                pretrained_θ=_pretrained_θ,
                freeze_mode=_freeze_mode,
                n_freeze_iters=_n_freeze_iters,
                explore_frac=_explore_frac,
                hyper_every=hyper_every,
                init_sampling=_init_sampling,
                acq_sampling=_acq_sampling,
                fixed_init_seed=_fixed_init_seed,
                n_restarts=_n_restarts,
            )

            # Always calculate true Q_det in original scale for reporting
            fcl, fsb, A = u_to_params(res.x_rec)
            Q_det_val = CalibrationCode.Q_det(t, fcl, fsb, A)
            n_iters = res.n_iter_actual > 0 ? res.n_iter_actual : 120

            # Convert y values back to Q scale if log_fidelity was used (y = log10(Q) → Q = 10^y)
            # y_last = last acquisition sample; y_rec = GP posterior mean at recommended point
            Q_noisy_val = use_log_fidelity ? 10^res.y_last : res.y_last
            Q_rec_val   = use_log_fidelity ? 10^res.y_rec  : res.y_rec

            # Total shots: all f calls including check calls not added to training data
            total_shots_used = res.total_shots
            n_train = length(res.σy)   # training points added to GP (may be < total calls)

            println("Simulation $sim_idx → Q_det = $(Q_det_val), Q_rec = $(round(Q_rec_val, digits=4)), iterations: $n_iters, total_shots: $total_shots_used")
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
    optimization_mode = optimize_det ? "Q_det" : "Q_noisy"
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

    println("\n=== All Results ===")
    ℓ_headers = join(["ℓ$i" for i in 1:length(bounds)], "\t")
    println("Sim\tSeed\tQ_det\tQ_noisy\tQ_rec\tIterations\tTotalShots\tNTrain\t$ℓ_headers\tσf\tc")
    for r in results_grid
        ℓ_str = join(round.(r.ℓ_final, digits=4), "\t")
        println("$(r.sim_idx)\t$(r.seed)\t$(r.Q_det)\t$(r.Q_noisy)\t$(r.Q_rec)\t$(r.n_iterations)\t$(r.total_shots)\t$(r.n_train)\t$ℓ_str\t$(round(r.σf_final,digits=4))\t$(round(r.c_final,digits=4))")
    end

    elapsed_seconds = time() - start_time
    elapsed_total = round(Int, elapsed_seconds)
    elapsed_hours = elapsed_total ÷ 3600
    elapsed_minutes = (elapsed_total % 3600) ÷ 60
    elapsed_secs = elapsed_total % 60
    elapsed_hms = string(elapsed_hours, ":", lpad(string(elapsed_minutes), 2, '0'), ":", lpad(string(elapsed_secs), 2, '0'))
    output_file = joinpath(@__DIR__, "..", "data", output_file_name)
    open(output_file, "w") do io
        println(io, "=== BENCHMARK RESULTS ===")
        println(io, "Optimization mode: $optimization_mode, N_shots = $N_shots, sigma_mode = $sigma_mode, use_log_fidelity = $use_log_fidelity")
        println(io, "Fixed α = $α, κ = $κ")
        println(io, "Number of simulations: $num_sims")
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
