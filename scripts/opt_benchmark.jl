import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Distributed
using Statistics
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

    # Variable-N mode:
    #   false → fixed N_shots per acquisition
    #   true  → variable mode: adaptive shots at x_rec; early stop when confirmed above threshold
    use_variable_mode  = get(ENV, "BO_VAR_N_MODE", "false") == "true"   # BO_VAR_N_MODE=true/false
    # Minimum shots for variable mode (also used for acquisition point)
    n_floor          = parse(Int, get(ENV, "BO_N_FLOOR",  "50"))      # BO_N_FLOOR=50
    # Maximum shots for :variable mode
    n_max_shots      = parse(Int, get(ENV, "BO_N_MAX",  "2000"))      # BO_N_MAX=2000

    # Objective mode:
    #   "2ms"             → Q_varMS(numMS=2), maximize F                     (default, original behavior)
    #   "2ms_log"         → Q_varMS(numMS=2), minimize log10(1-F)
    #   "3ms_balance"     → Q_varMS(numMS=3), maximize expected-population score
    #   "3ms_balance_log" → Q_varMS(numMS=3), minimize log10(1-F)
    #   "jacobian"        → Q_ms_sequence with searched subgates, maximize
    objective_mode   =           get(ENV, "BO_OBJECTIVE_MODE", "2ms")  # BO_OBJECTIVE_MODE=2ms/2ms_log/...
    # 4D mode: adds inter-gate phase φ as 4th optimization dimension
    use_4d           = get(ENV, "BO_USE_4D", "false") == "true"        # BO_USE_4D=true/false

    _is_log_mode = objective_mode in ("2ms_log", "3ms_balance_log")
    _do_maximize = !_is_log_mode

    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    span_kHz = 2.0
    span_fcl = span_kHz * 1e3 * 2π
    span_fsb = span_kHz * 1e3 * 2π
    span_A = 1.2 * A0 - A0
    span_phi = π / 10

    # Load Jacobian-searched subgates and compute expected populations if needed
    _jacobian_subgates, _jac_expected_gg, _jac_expected_ee = if objective_mode == "jacobian"
        _search_path = joinpath(@__DIR__, "..", "data", "ms_sequence_search_result.jl")
        isfile(_search_path) || error("Search result not found: $_search_path — run scripts/ms_sequence_search.jl first")
        _raw = include(_search_path)
        _subs = [CalibrationCode.MSSubgate(sg.theta, sg.phi) for sg in _raw.best_overall.subgates]
        _I_center = Float64(_raw.I_center)
        _pulses = CalibrationCode.build_closed_loop_ms_sequence(t, f_cl0, f_sb0, _I_center, _subs)
        _pops = CalibrationCode.populations_ms_sequence(_pulses)
        println("Jacobian sequence expected pops: gg=$(round(_pops.gg, digits=6)), ee=$(round(_pops.ee, digits=6)), odd=$(round(_pops.eg + _pops.ge, digits=6))")
        (_subs, Float64(_pops.gg), Float64(_pops.ee))
    else
        (CalibrationCode.MSSubgate[], NaN, NaN)
    end

    @everywhere const t = $t
    @everywhere const f_cl0 = $f_cl0
    @everywhere const f_sb0 = $f_sb0
    @everywhere const A0 = $A0
    @everywhere const span_fcl = $span_fcl
    @everywhere const span_fsb = $span_fsb
    @everywhere const span_A = $span_A
    @everywhere const span_phi = $span_phi
    @everywhere const optimize_det = $optimize_det
    @everywhere const _objective_mode = $objective_mode
    @everywhere const _use_4d = $use_4d
    @everywhere const _is_log_mode = $_is_log_mode
    @everywhere const _jac_subgates = $_jacobian_subgates
    @everywhere const _jac_exp_gg = $_jac_expected_gg
    @everywhere const _jac_exp_ee = $_jac_expected_ee

    @everywhere const _phase_drift = Ref(0.0)

    @everywhere function u_to_params(u)
        fcl = f_cl0 + span_fcl * u[1]
        fsb = f_sb0 + span_fsb * u[2]
        A   = A0    + span_A   * u[3]
        if _use_4d
            phi = span_phi * u[4]
            return (fcl, fsb, A, phi)
        end
        return (fcl, fsb, A)
    end

    @everywhere function _eval_raw(fcl, fsb, A, N::Int; phi::Float64=0.0)
        if _objective_mode == "3ms_balance" || _objective_mode == "3ms_balance_log"
            return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=3,
                relative_phase=phi, phase_drift=_phase_drift[])
        elseif _objective_mode == "jacobian"
            return clamp(CalibrationCode.Q_ms_sequence(t, fcl, fsb, A, _jac_subgates;
                N=N, expected_gg=_jac_exp_gg, expected_ee=_jac_exp_ee,
                relative_phase=phi, phase_drift=_phase_drift[]), 0.0, 1.0)
        elseif _use_4d
            return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2,
                relative_phase=phi, phase_drift=_phase_drift[])
        else
            if optimize_det
                return clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            else
                return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2)
            end
        end
    end

    @everywhere function _to_objective(F_raw::Float64)
        F = clamp(F_raw, 0.0, 1.0)
        _is_log_mode && return log10(max(1.0 - F, 1e-10))
        return F
    end

    @everywhere function _sigma_for_objective(F_raw::Float64, N::Int)
        optimize_det && return 0.0
        F = clamp(F_raw, 0.0, 1.0)
        σ_F = sqrt(max(F * (1.0 - F), 0.0) / N)
        if _is_log_mode
            infid = max(1.0 - F, 1e-10)
            return σ_F / (infid * log(10))
        end
        return σ_F
    end

    @everywhere function Q_fun(u, N::Int)
        params = u_to_params(u)
        phi = _use_4d ? params[4] : 0.0
        F_raw = _eval_raw(params[1], params[2], params[3], N; phi=phi)
        return (_to_objective(F_raw), _sigma_for_objective(F_raw, N))
    end

    @everywhere function Q_true(u)
        params = u_to_params(u)
        fcl, fsb, A = params[1], params[2], params[3]
        return clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
    end

    bounds = use_4d ? [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)] : [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]

    α = 1.5
    κ = 1.9

    fidelity_threshold = if _is_log_mode && fidelity_threshold_Q !== nothing
        log10(max(1.0 - fidelity_threshold_Q, 1e-10))
    else
        fidelity_threshold_Q
    end

    optimization_mode = "$(objective_mode)$(use_4d ? "_4d" : "")$(optimize_det ? "_det" : "")"

    println("=== Starting Parallel Simulations (Random Seeds) ===")
    println("Objective mode: $objective_mode, 4D: $use_4d, maximize: $_do_maximize")
    println("N_shots = $N_shots, noise model = binomial")
    println("Fixed α = $α, κ = $κ")
    println("Number of simulations: $num_sims, n_iter = $n_iter, n_restarts = $n_restarts")
    println("Fidelity threshold: $(fidelity_threshold_Q === nothing ? "none (fixed $n_iter iterations)" : "Q* = $fidelity_threshold_Q (linear)")")
    println("use_variable_mode = $use_variable_mode$(use_variable_mode ? ", n_floor=$n_floor, n_max=$n_max_shots" : "")")

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
    _use_variable_mode   = use_variable_mode
    _n_floor             = n_floor
    _n_max_shots         = n_max_shots
    _n_restarts          = n_restarts
    _learn_noise_scale   = !optimize_det

    results_grid = pmap(1:num_sims; batch_size=1) do sim_idx
        try
            seed = _random_seeds[sim_idx]
            if _use_4d
                _phase_drift[] = (rand(MersenneTwister()) * 2.0 - 1.0) * span_phi
                println("Starting simulation $sim_idx (phase_drift = $(round(_phase_drift[], digits=4)))...")
            else
                _phase_drift[] = 0.0
                println("Starting simulation $sim_idx...")
            end
            flush(stdout)
            local drift_val = _phase_drift[]
            res = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
                bounds=bounds,
                n_shots=N_shots,
                n_init=n_initial_samples,
                n_iter=_n_iter,
                κ=κ,
                α=α,
                seed=seed,
                maximize=_do_maximize,
                fidelity_threshold=fidelity_threshold,
                explore_frac=0.0,
                hyper_every=hyper_every,
                learn_noise_scale=_learn_noise_scale,
                n_restarts=_n_restarts,
                use_variable_mode=_use_variable_mode,
                n_floor=_n_floor,
                n_max_shots=_n_max_shots,
            )

            _params = u_to_params(res.x_rec)
            fcl, fsb, A = _params[1], _params[2], _params[3]
            phi_rec = _use_4d ? _params[4] : 0.0
            Q_det_val = clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            n_iters = res.n_iter_actual > 0 ? res.n_iter_actual : _n_iter

            Q_noisy_val = res.y_last   # linear Q scale
            Q_rec_val   = res.y_rec    # GP posterior mean at recommended point

            total_shots_used = res.total_shots
            n_train = length(res.σy)

            if _use_4d
                println("Simulation $sim_idx → Q_det = $(round(Q_det_val, digits=4)), Q_rec = $(round(Q_rec_val, digits=4)), phi_rec = $(round(phi_rec, digits=4)), drift = $(round(drift_val, digits=4)), iterations: $n_iters, total_shots: $total_shots_used")
            else
                println("Simulation $sim_idx → Q_det = $(round(Q_det_val, digits=4)), Q_rec = $(round(Q_rec_val, digits=4)), iterations: $n_iters, total_shots: $total_shots_used")
            end
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
                phase_drift=drift_val,
                phi_rec=phi_rec,
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

    _best_params = u_to_params(best_result.x_rec)
    fcl_rec, fsb_rec, A_rec = _best_params[1], _best_params[2], _best_params[3]
    println("\n=== BEST RESULT PHYSICAL PARAMETERS ===")
    @printf("Recommended f_cl = %.15e\n", fcl_rec)
    @printf("Recommended f_sb = %.15e\n", fsb_rec)
    @printf("Recommended A    = %.15e\n", A_rec)
    if use_4d
        println("Recommended phi  = ", _best_params[4])
    end
    @printf("Baseline   f_cl = %.15e  f_sb = %.15e  A = %.15e\n", f_cl0, f_sb0, A0)

    elapsed_seconds = time() - start_time
    elapsed_total = round(Int, elapsed_seconds)
    elapsed_hours = elapsed_total ÷ 3600
    elapsed_minutes = (elapsed_total % 3600) ÷ 60
    elapsed_secs = elapsed_total % 60
    elapsed_hms = string(elapsed_hours, ":", lpad(string(elapsed_minutes), 2, '0'), ":", lpad(string(elapsed_secs), 2, '0'))

    _output_dir = get(ENV, "BO_OUTPUT_DIR", joinpath(@__DIR__, "..", "data"))
    output_file = joinpath(_output_dir, output_file_name)
    open(output_file, "w") do io
        println(io, "=== BENCHMARK RESULTS ===")
        println(io, "Optimization mode: $optimization_mode, objective: $objective_mode, 4D: $use_4d, maximize: $_do_maximize")
        println(io, "N_shots = $N_shots, noise model = binomial")
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
        @printf(io, "Recommended f_cl = %.15e\n", fcl_rec)
        @printf(io, "Recommended f_sb = %.15e\n", fsb_rec)
        @printf(io, "Recommended A    = %.15e\n", A_rec)
        use_4d && println(io, "Recommended phi  = $(_best_params[4])")
        @printf(io, "Baseline   f_cl = %.15e  f_sb = %.15e  A = %.15e\n", f_cl0, f_sb0, A0)
        println(io, "")
        println(io, "=== All Results ===")
        ℓ_headers = join(["ℓ$i" for i in 1:length(bounds)], "\t")
        phase_headers = use_4d ? "\tdrift\tphi_rec\tdelta_phi" : ""
        println(io, "Sim\tSeed\tIterations\tTotalShots\tNTrain\tQ_det\tQ_noisy\tQ_rec\t$ℓ_headers\tσf\tc$phase_headers")
        for r in results_grid
            ℓ_str = join(round.(r.ℓ_final, digits=4), "\t")
            phase_str = use_4d ? "\t$(round(r.phase_drift, digits=6))\t$(round(r.phi_rec, digits=6))\t$(round(r.phi_rec - r.phase_drift, digits=6))" : ""
            println(io, "$(r.sim_idx)\t$(r.seed)\t$(r.n_iterations)\t$(r.total_shots)\t$(r.n_train)\t$(r.Q_det)\t$(r.Q_noisy)\t$(r.Q_rec)\t$ℓ_str\t$(round(r.σf_final,digits=4))\t$(round(r.c_final,digits=4))$phase_str")
        end
    end

    println("\nResults written to: $output_file")

finally
    rmprocs(workers())
end
