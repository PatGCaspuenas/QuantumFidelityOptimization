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

    optimize_det      = get(ENV, "BO_OPT_DET",      "false") == "true"
    N_shots           = parse(Int, get(ENV, "BO_N_SHOTS",     "400"))
    n_initial_samples = parse(Int, get(ENV, "BO_N_INIT",      "12"))
    hyper_every       = parse(Int, get(ENV, "BO_HYPER_EVERY", "10"))
    num_sims          = parse(Int, get(ENV, "BO_NUM_SIMS",    "3"))
    output_file_name  =           get(ENV, "BO_OUTPUT_FILE",  "benchmark_results.txt")
    n_iter            = parse(Int, get(ENV, "BO_N_ITER",      "120"))
    n_restarts        = parse(Int, get(ENV, "BO_N_RESTARTS",  "10"))

    # Early-stopping threshold: "0.998" | "auto" (1-1/N) | "" (none)
    _thresh_env = get(ENV, "BO_THRESH_Q", "")
    fidelity_threshold = if isempty(_thresh_env)
        nothing
    elseif _thresh_env == "auto"
        1.0 - 1.0 / N_shots
    else
        parse(Float64, _thresh_env)
    end

    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    span_kHz = 2.0
    span_fcl = span_kHz * 1e3 * 2π
    span_fsb = span_kHz * 1e3 * 2π
    span_A   = 1.2 * A0 - A0

    @everywhere const t             = $t
    @everywhere const f_cl0         = $f_cl0
    @everywhere const f_sb0         = $f_sb0
    @everywhere const A0            = $A0
    @everywhere const span_fcl      = $span_fcl
    @everywhere const span_fsb      = $span_fsb
    @everywhere const span_A        = $span_A
    @everywhere const optimize_det  = $optimize_det
    @everywhere const _obj_mode     = $objective_mode

    @everywhere function u_to_params(u)
        return (f_cl0 + span_fcl * u[1],
                f_sb0 + span_fsb * u[2],
                A0    + span_A   * u[3])
    end

    @everywhere function Q_fun(u, N::Int)
        fcl, fsb, A = u_to_params(u)
        optimize_det && return clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0), 0.0
        F, σ = CalibrationCode.Q_varMS_σ(t, fcl, fsb, A; N=N, numMS=2)
        return clamp(F, 0.0, 1.0), σ
    end

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    α_bo = 1.5
    κ_bo = 1.9

    noise_model    = "binomial"

    println("=== Starting Parallel Simulations (Random Seeds) ===")
    println("Objective: $objective_mode, noise model: $noise_model")
    println("N_shots = $N_shots, n_init = $n_initial_samples, n_iter = $n_iter, n_restarts = $n_restarts")
    println("κ = $κ_bo, acq = ucb_random_scan")
    println("Fidelity threshold: $(fidelity_threshold === nothing ? "none" : "Q* = $fidelity_threshold (mu_one_check)")")
    println("Number of simulations: $num_sims")

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
    _n_restarts        = n_restarts
    _learn_noise_scale = !optimize_det
    _fidelity_threshold = fidelity_threshold

    results_grid = pmap(1:num_sims; batch_size=1) do sim_idx
        try
            seed = _random_seeds[sim_idx]
            Random.seed!(seed)
            println("Starting simulation $sim_idx...")
            flush(stdout)

            res = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
                bounds=bounds,
                n_shots=N_shots,
                n_init=n_initial_samples,
                n_iter=_n_iter,
                κ=κ_bo,
                α=α_bo,
                seed=seed,
                maximize=true,
                fidelity_threshold=_fidelity_threshold,
                hyper_every=hyper_every,
                learn_noise_scale=_learn_noise_scale,
                n_restarts=_n_restarts,
            )

            fcl, fsb, A = u_to_params(res.x_rec)
            Q_det_val = clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
            n_iters   = res.n_iter_actual > 0 ? res.n_iter_actual : _n_iter

            println("Simulation $sim_idx → Q_det=$(round(Q_det_val,digits=4))  iters=$n_iters  shots=$(res.total_shots)")
            flush(stdout)

            (
                sim_idx    = sim_idx,
                seed       = seed,
                x_rec      = res.x_rec,
                Q_rec      = res.y_rec,
                Q_det      = Q_det_val,
                Q_last     = res.y_last,
                n_iterations = n_iters,
                total_shots  = res.total_shots,
                n_train    = length(res.σy),
                ℓ_final    = res.ℓ_final,
                σf_final   = res.σf_final,
                c_final    = res.c_final,
            )
        catch e
            println("ERROR in simulation $sim_idx: $(typeof(e)): $e")
            Base.showerror(stdout, e, catch_backtrace())
            println()
            flush(stdout)
            rethrow()
        end
    end

    Q_det_values = [r.Q_det for r in results_grid]
    best_idx     = argmax(Q_det_values)
    best_result  = results_grid[best_idx]

    avg_Q_det = mean(Q_det_values); med_Q_det = median(Q_det_values)
    min_Q_det = minimum(Q_det_values); max_Q_det = maximum(Q_det_values)
    std_Q_det = std(Q_det_values)

    iter_values    = [r.n_iterations for r in results_grid]
    avg_iterations = mean(iter_values); med_iterations = median(iter_values)
    std_iterations = std(iter_values)
    total_train    = sum(r.n_train for r in results_grid)

    println("\n=== RESULTS SUMMARY ===")
    println("Best Q_det = $(best_result.Q_det)  (Simulation $(best_result.sim_idx))")
    println("Average Q_det = $avg_Q_det ± $std_Q_det")
    println("Median Q_det = $med_Q_det")
    println("Min/Max Q_det = $min_Q_det / $max_Q_det")
    println("Average iterations = $avg_iterations ± $std_iterations")
    println("Median iterations  = $med_iterations")
    println("Total training points = $total_train")

    elapsed_seconds = time() - start_time
    elapsed_total   = round(Int, elapsed_seconds)
    elapsed_hms = string(elapsed_total ÷ 3600, ":",
                         lpad(string((elapsed_total % 3600) ÷ 60), 2, '0'), ":",
                         lpad(string(elapsed_total % 60), 2, '0'))

    _output_dir = get(ENV, "BO_OUTPUT_DIR", joinpath(@__DIR__, "data"))
    output_file = joinpath(_output_dir, output_file_name)
    open(output_file, "w") do io
        println(io, "=== BENCHMARK RESULTS ===")
        println(io, "Objective: $objective_mode, noise model: $noise_model, maximize: true")
        println(io, "N_shots = $N_shots, n_init = $n_initial_samples, n_iter = $n_iter, n_restarts = $n_restarts")
        println(io, "κ = $κ_bo, acq = ucb_random_scan$a_bound_label")
        println(io, "Fidelity threshold: $(fidelity_threshold === nothing ? "none" : "Q* = $fidelity_threshold (mu_one_check)")")
        println(io, "Elapsed time: $elapsed_hms")
        println(io, "")
        println(io, "=== RESULTS SUMMARY ===")
        println(io, "Best Q_det = $(best_result.Q_det) (Sim $(best_result.sim_idx), Seed $(best_result.seed), Iters $(best_result.n_iterations))")
        println(io, "Average Q_det = $avg_Q_det ± $std_Q_det")
        println(io, "Median Q_det = $med_Q_det")
        println(io, "Min Q_det = $min_Q_det")
        println(io, "Max Q_det = $max_Q_det")
        println(io, "Average iterations = $avg_iterations ± $std_iterations")
        println(io, "Median iterations  = $med_iterations")
        println(io, "Total training points = $total_train")
        println(io, "")
        println(io, "=== BEST RESULT PHYSICAL PARAMETERS ===")
        fcl_rec, fsb_rec, A_rec = u_to_params(best_result.x_rec)
        @printf(io, "Recommended f_cl = %.15e\n", fcl_rec)
        @printf(io, "Recommended f_sb = %.15e\n", fsb_rec)
        @printf(io, "Recommended A    = %.15e\n", A_rec)
        @printf(io, "Baseline   f_cl = %.15e  f_sb = %.15e  A = %.15e\n", f_cl0, f_sb0, A0)
        println(io, "Best u_rec = $(best_result.x_rec)")
        println(io, "Best GP hyperparameters: ℓ=$(round.(best_result.ℓ_final,digits=4))  σf=$(round(best_result.σf_final,digits=4))  c=$(round(best_result.c_final,digits=4))")
        println(io, "")
        println(io, "=== All Results ===")
        println(io, "Sim\tSeed\tIterations\tTotalShots\tNTrain\tQ_det\tQ_last\tQ_rec\tℓ1\tℓ2\tℓ3\tσf\tc")
        for r in results_grid
            ℓ_str = join(round.(r.ℓ_final, digits=4), "\t")
            println(io, "$(r.sim_idx)\t$(r.seed)\t$(r.n_iterations)\t$(r.total_shots)\t$(r.n_train)\t$(r.Q_det)\t$(r.Q_last)\t$(r.Q_rec)\t$ℓ_str\t$(round(r.σf_final,digits=4))\t$(round(r.c_final,digits=4))")
        end
    end

    println("\nResults written to: $output_file")

finally
    rmprocs(workers())
end
