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

    # true  → optimize Q_det (deterministic, no shots); false → optimize noisy Q
    optimize_det      = get(ENV, "BO_OPT_DET",   "false") == "true"
    N_shots           = parse(Int, get(ENV, "BO_N_SHOTS",    "400"))
    n_initial_samples = parse(Int, get(ENV, "BO_N_INIT",   "12"))
    hyper_every       = parse(Int, get(ENV, "BO_HYPER_EVERY","10"))
    num_sims          = parse(Int, get(ENV, "BO_NUM_SIMS",  "3"))
    output_file_name  =           get(ENV, "BO_OUTPUT_FILE", "benchmark_results.txt")

    # Early-stopping threshold: "0.998" | "auto" (1-1/N) | "" (none)
    _thresh_env = get(ENV, "BO_THRESH_Q", "")
    fidelity_threshold_Q = if isempty(_thresh_env)
        nothing
    elseif _thresh_env == "auto"
        1.0 - 1.0 / N_shots
    else
        parse(Float64, _thresh_env)
    end

    n_iter       = parse(Int,   get(ENV, "BO_N_ITER",      "120"))
    n_restarts   = parse(Int,   get(ENV, "BO_N_RESTARTS",  "10"))

    use_variable_mode = get(ENV, "BO_VAR_N_MODE", "false") == "true"
    n_floor           = parse(Int, get(ENV, "BO_N_FLOOR",  "50"))
    n_max_shots       = parse(Int, get(ENV, "BO_N_MAX",  "2000"))

    # Objective mode:
    #   "2ms"             → Q_varMS(numMS=2)              [default]
    #   "3ms"             → Q_varMS(numMS=3)
    #   "2ms_log"         → Q_varMS(numMS=2), minimize log10(1-F)
    #   "3ms_balance"     → Q_varMS_balance(numMS=3)
    #   "3ms_balance_log" → Q_varMS_balance(numMS=3), minimize log10(1-F)
    objective_mode = get(ENV, "BO_OBJECTIVE_MODE", "2ms")
    use_4d         = get(ENV, "BO_USE_4D", "false") == "true"

    _valid_modes = ("2ms", "3ms", "2ms_log", "3ms_balance", "3ms_balance_log")
    objective_mode in _valid_modes || error("Unknown BO_OBJECTIVE_MODE=$objective_mode. Valid: $(_valid_modes)")

    _is_log_mode = objective_mode in ("2ms_log", "3ms_balance_log")
    _do_maximize = !_is_log_mode

    # Acquisition configuration
    k_acq         = parse(Int,   get(ENV, "BO_K_ACQ",          "1"))
    min_sep       = parse(Float64, get(ENV, "BO_MIN_SEP",       "0.05"))
    use_zoom      = get(ENV, "BO_USE_ZOOM",      "false") == "true"
    M_zoom        = parse(Int,   get(ENV, "BO_M_ZOOM",          "200"))
    zoom_radius   = parse(Float64, get(ENV, "BO_ZOOM_RADIUS",   "0.1"))
    use_lbfgs_acq = get(ENV, "BO_USE_LBFGS_ACQ", "false") == "true"
    use_grad_acq  = get(ENV, "BO_USE_GRAD_ACQ",  "false") == "true"

    # High-N Q_noisy evaluation at x_rec after BO (benchmark metric)
    N_noisy_high = parse(Int, get(ENV, "BO_Q_NOISY_N", "5000"))

    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    span_kHz = 2.0
    span_fcl = span_kHz * 1e3 * 2π
    span_fsb = span_kHz * 1e3 * 2π
    span_A   = 1.2 * A0 - A0
    span_phi = π / 10

    @everywhere const t        = $t
    @everywhere const f_cl0    = $f_cl0
    @everywhere const f_sb0    = $f_sb0
    @everywhere const A0       = $A0
    @everywhere const span_fcl = $span_fcl
    @everywhere const span_fsb = $span_fsb
    @everywhere const span_A   = $span_A
    @everywhere const span_phi = $span_phi
    @everywhere const optimize_det     = $optimize_det
    @everywhere const _objective_mode  = $objective_mode
    @everywhere const _use_4d          = $use_4d
    @everywhere const _is_log_mode     = $_is_log_mode
    @everywhere const _N_noisy_high    = $N_noisy_high

    @everywhere const _phase_drift = Ref(0.0)

    @everywhere function u_to_params(u)
        fcl = f_cl0 + span_fcl * u[1]
        fsb = f_sb0 + span_fsb * u[2]
        A   = A0    + span_A   * u[3]
        _use_4d && return (fcl, fsb, A, span_phi * u[4])
        return (fcl, fsb, A)
    end

    @everywhere function _eval_raw(fcl, fsb, A, N::Int; phi::Float64=0.0)
        if _objective_mode == "3ms_balance" || _objective_mode == "3ms_balance_log"
            return CalibrationCode.Q_varMS_balance(t, fcl, fsb, A; N=N, numMS=3,
                relative_phase=phi, phase_drift=_phase_drift[])
        elseif _objective_mode == "3ms"
            return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=3,
                relative_phase=phi, phase_drift=_phase_drift[])
        elseif _use_4d   # 2ms 4D
            return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2,
                relative_phase=phi, phase_drift=_phase_drift[])
        else             # 2ms 3D (default)
            optimize_det && return clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            return CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2)
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

    bounds = use_4d ? [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)] :
                      [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]

    α_bo = 1.5
    κ_bo = 1.9

    fidelity_threshold = if _is_log_mode && fidelity_threshold_Q !== nothing
        log10(max(1.0 - fidelity_threshold_Q, 1e-10))
    else
        fidelity_threshold_Q
    end

    acq_label = if !use_lbfgs_acq && !use_zoom
        "baseline(k=$k_acq,random)"
    elseif use_zoom && !use_lbfgs_acq
        "zoom(k=$k_acq,r=$zoom_radius,M=$M_zoom)"
    elseif use_lbfgs_acq && !use_grad_acq
        "lbfgs(k=$k_acq,finite_diff)"
    else
        "lbfgs_grad(k=$k_acq,analytical)"
    end

    optimization_mode = "$(objective_mode)$(use_4d ? "_4d" : "")$(optimize_det ? "_det" : "")"

    println("=== Starting Parallel Simulations (Random Seeds) ===")
    println("Objective mode: $objective_mode, 4D: $use_4d, maximize: $_do_maximize")
    println("N_shots = $N_shots, noise model = binomial")
    println("Fixed α = $α_bo, κ = $κ_bo")
    println("Number of simulations: $num_sims, n_iter = $n_iter, n_restarts = $n_restarts")
    println("Fidelity threshold: $(fidelity_threshold_Q === nothing ? "none (fixed $n_iter iterations)" : "Q* = $fidelity_threshold_Q (linear)")")
    println("use_variable_mode = $use_variable_mode$(use_variable_mode ? ", n_floor=$n_floor, n_max=$n_max_shots" : "")")
    println("Acquisition: $acq_label")
    println("Q_noisy benchmark N = $N_noisy_high")

    start_time = time()

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

    _random_seeds      = random_seeds
    _n_iter            = n_iter
    _use_variable_mode = use_variable_mode
    _n_floor           = n_floor
    _n_max_shots       = n_max_shots
    _n_restarts        = n_restarts
    _learn_noise_scale = !optimize_det
    _k_acq             = k_acq
    _min_sep           = min_sep
    _use_zoom          = use_zoom
    _M_zoom            = M_zoom
    _zoom_radius       = zoom_radius
    _use_lbfgs_acq     = use_lbfgs_acq
    _use_grad_acq      = use_grad_acq

    results_grid = pmap(1:num_sims; batch_size=1) do sim_idx
        try
            seed = _random_seeds[sim_idx]
            Random.seed!(seed)   # seeds global RNG so Q-function draws (StatsBase) are reproducible
            if _use_4d
                _phase_drift[] = (rand() * 2.0 - 1.0) * span_phi
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
                κ=κ_bo,
                α=α_bo,
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
                k_acq=_k_acq,
                min_sep=_min_sep,
                use_zoom=_use_zoom,
                M_zoom=_M_zoom,
                zoom_radius=_zoom_radius,
                use_lbfgs_acq=_use_lbfgs_acq,
                use_grad_acq=_use_grad_acq,
            )

            _params   = u_to_params(res.x_rec)
            fcl, fsb, A = _params[1], _params[2], _params[3]
            phi_rec   = _use_4d ? _params[4] : 0.0
            Q_det_val = clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            n_iters   = res.n_iter_actual > 0 ? res.n_iter_actual : _n_iter

            # High-N Q_noisy at recommended point (single-gate parity scan benchmark)
            Q_noisy_hN_val = try
                clamp(CalibrationCode.Q_noisy(t, fcl, fsb, A;
                    phi_1=phi_rec, phi_2=0.0, N=_N_noisy_high), 0.0, 1.0)
            catch
                NaN
            end

            Q_noisy_val   = res.y_last   # last acquisition obs (linear Q scale)
            Q_rec_val     = res.y_rec    # GP posterior mean at recommended point
            total_shots_used = res.total_shots
            n_train = length(res.σy)

            if _use_4d
                println("Simulation $sim_idx → Q_det=$(round(Q_det_val,digits=4))  Q_noisy_hN=$(round(Q_noisy_hN_val,digits=4))  phi_rec=$(round(phi_rec,digits=4))  drift=$(round(drift_val,digits=4))  iters=$n_iters  shots=$total_shots_used")
            else
                println("Simulation $sim_idx → Q_det=$(round(Q_det_val,digits=4))  Q_noisy_hN=$(round(Q_noisy_hN_val,digits=4))  iters=$n_iters  shots=$total_shots_used")
            end
            flush(stdout)

            (
                sim_idx=sim_idx,
                seed=seed,
                x_rec=res.x_rec,
                Q_rec=Q_rec_val,
                Q_det=Q_det_val,
                Q_noisy_last=Q_noisy_val,
                Q_noisy_hN=Q_noisy_hN_val,
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

    Q_det_values    = [r.Q_det     for r in results_grid]
    Q_noisy_values  = [r.Q_noisy_hN for r in results_grid]

    best_idx    = argmax(Q_det_values)
    best_result = results_grid[best_idx]

    avg_Q_det = mean(Q_det_values);   med_Q_det = median(Q_det_values)
    min_Q_det = minimum(Q_det_values); max_Q_det = maximum(Q_det_values)
    std_Q_det = std(Q_det_values)

    valid_qn = filter(isfinite, Q_noisy_values)
    avg_Q_noisy = isempty(valid_qn) ? NaN : mean(valid_qn)
    med_Q_noisy = isempty(valid_qn) ? NaN : median(valid_qn)
    std_Q_noisy = isempty(valid_qn) ? NaN : std(valid_qn)

    iter_values     = [r.n_iterations for r in results_grid]
    avg_iterations  = mean(iter_values); med_iterations = median(iter_values)
    std_iterations  = std(iter_values)

    total_train = sum(r.n_train for r in results_grid)

    println("\n=== RESULTS SUMMARY ===")
    println("Optimization mode: $optimization_mode, acq: $acq_label")
    println("Best Q_det = $(best_result.Q_det)  (Simulation $(best_result.sim_idx))")
    println("Average Q_det = $avg_Q_det ± $std_Q_det")
    println("Median Q_det = $med_Q_det")
    println("Min/Max Q_det = $min_Q_det / $max_Q_det")
    println("Average Q_noisy (N=$N_noisy_high) = $avg_Q_noisy ± $std_Q_noisy")
    println("Median  Q_noisy (N=$N_noisy_high) = $med_Q_noisy")
    println("Average iterations = $avg_iterations ± $std_iterations")
    println("Median iterations  = $med_iterations")
    println("Total training points across all sims = $total_train")

    elapsed_seconds = time() - start_time
    elapsed_total   = round(Int, elapsed_seconds)
    elapsed_hms = string(elapsed_total ÷ 3600, ":",
                         lpad(string((elapsed_total % 3600) ÷ 60), 2, '0'), ":",
                         lpad(string(elapsed_total % 60), 2, '0'))

    _output_dir = get(ENV, "BO_OUTPUT_DIR", joinpath(@__DIR__, "..", "data"))
    output_file = joinpath(_output_dir, output_file_name)
    open(output_file, "w") do io
        println(io, "=== BENCHMARK RESULTS ===")
        println(io, "Optimization mode: $optimization_mode, objective: $objective_mode, 4D: $use_4d, maximize: $_do_maximize")
        println(io, "N_shots = $N_shots, noise model = binomial")
        println(io, "Fixed α = $α_bo, κ = $κ_bo")
        println(io, "Number of simulations: $num_sims, n_iter = $n_iter, n_restarts = $n_restarts")
        println(io, "Fidelity threshold: $(fidelity_threshold_Q === nothing ? "none" : "Q* = $fidelity_threshold_Q")")
        println(io, "use_variable_mode = $use_variable_mode$(use_variable_mode ? ", n_floor=$n_floor, n_max=$n_max_shots" : "")")
        println(io, "Acquisition: $acq_label  (k_acq=$k_acq, min_sep=$min_sep, use_zoom=$use_zoom, M_zoom=$M_zoom, zoom_radius=$zoom_radius, use_lbfgs_acq=$use_lbfgs_acq, use_grad_acq=$use_grad_acq)")
        println(io, "Q_noisy benchmark N = $N_noisy_high")
        println(io, "Elapsed time: $elapsed_hms")
        println(io, "")
        println(io, "=== RESULTS SUMMARY ===")
        println(io, "Best Q_det = $(best_result.Q_det) (Simulation $(best_result.sim_idx), Seed $(best_result.seed), Iterations $(best_result.n_iterations))")
        println(io, "Average Q_det = $avg_Q_det ± $std_Q_det")
        println(io, "Median Q_det = $med_Q_det")
        println(io, "Min Q_det = $min_Q_det")
        println(io, "Max Q_det = $max_Q_det")
        println(io, "Average Q_noisy_hN (N=$N_noisy_high) = $avg_Q_noisy ± $std_Q_noisy")
        println(io, "Median  Q_noisy_hN (N=$N_noisy_high) = $med_Q_noisy")
        println(io, "Average iterations = $avg_iterations ± $std_iterations")
        println(io, "Median iterations  = $med_iterations")
        println(io, "Total training points across all sims = $total_train")
        println(io, "")
        println(io, "=== BEST RESULT PHYSICAL PARAMETERS ===")
        _best_params = u_to_params(best_result.x_rec)
        fcl_rec, fsb_rec, A_rec = _best_params[1], _best_params[2], _best_params[3]
        @printf(io, "Recommended f_cl = %.15e\n", fcl_rec)
        @printf(io, "Recommended f_sb = %.15e\n", fsb_rec)
        @printf(io, "Recommended A    = %.15e\n", A_rec)
        use_4d && println(io, "Recommended phi  = $(_best_params[4])")
        @printf(io, "Baseline   f_cl = %.15e  f_sb = %.15e  A = %.15e\n", f_cl0, f_sb0, A0)
        println(io, "Best final GP hyperparameters:")
        println(io, "  ℓ  = $(round.(best_result.ℓ_final, digits=4))")
        println(io, "  σf = $(round(best_result.σf_final, digits=4))")
        println(io, "  c  = $(round(best_result.c_final,  digits=4))")
        println(io, "Best u_rec = $(best_result.x_rec)")
        println(io, "")
        println(io, "=== All Results ===")
        ℓ_headers     = join(["ℓ$i" for i in 1:length(bounds)], "\t")
        phase_headers = use_4d ? "\tdrift\tphi_rec\tdelta_phi" : ""
        println(io, "Sim\tSeed\tIterations\tTotalShots\tNTrain\tQ_det\tQ_noisy_last\tQ_noisy_hN\tQ_rec\t$ℓ_headers\tσf\tc$phase_headers")
        for r in results_grid
            ℓ_str     = join(round.(r.ℓ_final, digits=4), "\t")
            phase_str = use_4d ? "\t$(round(r.phase_drift,digits=6))\t$(round(r.phi_rec,digits=6))\t$(round(r.phi_rec - r.phase_drift,digits=6))" : ""
            println(io, "$(r.sim_idx)\t$(r.seed)\t$(r.n_iterations)\t$(r.total_shots)\t$(r.n_train)\t$(r.Q_det)\t$(r.Q_noisy_last)\t$(r.Q_noisy_hN)\t$(r.Q_rec)\t$ℓ_str\t$(round(r.σf_final,digits=4))\t$(round(r.c_final,digits=4))$phase_str")
        end
    end

    println("\nResults written to: $output_file")

finally
    rmprocs(workers())
end
