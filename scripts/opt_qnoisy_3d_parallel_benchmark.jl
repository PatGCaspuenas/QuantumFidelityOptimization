using Random
using Distributed
using Statistics
import Pkg


if nprocs() == 1
    addprocs()
end

try

    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()

    @everywhere include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
    @everywhere using .CalibrationCode

    use_log_fidelity = false
    optimize_det = true

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

    @everywhere u_to_params(u) = (f_cl0 + span_fcl * u[1],
        f_sb0 + span_fsb * u[2],
        A0 + span_A * u[3])

    @everywhere function N_from_sigma(σ::Float64)
        N = round(Int, 1 / (σ^2))
        return clamp(N, 20, 10000)
    end

    @everywhere function apply_log_fidelity(Q::Float64)
        if use_log_fidelity
            return log(1.0 - Q, 10)
        else
            return Q
        end
    end

    @everywhere function Q_fun(u, σ)
        fcl, fsb, A = u_to_params(u)
        if optimize_det
            Q = CalibrationCode.Q_det(t, fcl, fsb, A)
        else
            Q = CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N_from_sigma(σ), numMS=2)#CalibrationCode.Q_noisy(t, fcl, fsb, A; N=N_from_sigma(σ))#
        end
        return apply_log_fidelity(Q)
    end

    @everywhere function Q_true(u)
        fcl, fsb, A = u_to_params(u)
        Q = CalibrationCode.Q_det(t, fcl, fsb, A)
        return apply_log_fidelity(Q)
    end


    σ_levels = [0.1412, 0.1, 0.06, 0.04472]
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    α = 1.5
    κ = 1.9
    num_sims = 5
    fidelity_threshold = 1 - 1 / N_from_sigma(σ_levels[end])

    # --- Pretrained GP hyperparameters ---
    # freeze_mode options:
    #   :none          — no pretraining, full MLE from scratch (original behavior)
    #   :lengthscales  — fix ℓ from pretrained θ, adapt σf and noise scale
    #   :all           — fix all hyperparameters (ℓ, σf, c) throughout BO
    #
    # Run `julia --project=. scripts/pretrain_gp_3d.jl` first to generate the pretrained file.
    pretrain_file = joinpath(@__DIR__, "data", "pretrained_theta_3d.jl")
    freeze_mode = :lengthscales
    n_freeze_iters = 40  # fix ℓ for first 40 iterations, then release to full MLE

    pretrained_θ = if isfile(pretrain_file)
        include(pretrain_file)
        pretrained_theta_3d()
    else
        @warn "Pretrained file not found: $pretrain_file — running without pretraining (freeze_mode ignored)"
        nothing
    end

    optimization_mode = optimize_det ? "Q_det" : "Q_noisy"

    println("=== Starting Parallel Simulations (Random Seeds) ===")
    println("Optimization mode: $optimization_mode")
    println("Fixed α = $α, κ = $κ")
    println("Number of simulations: $num_sims")
    if fidelity_threshold !== nothing && !use_log_fidelity
        println("Fidelity threshold: $fidelity_threshold")
    end

    start_time = time()

    random_seeds = rand(1:1000000, num_sims)

    # Capture pretraining config as locals so pmap serializes values, not globals.
    explore_frac = 0.15
    _pretrained_θ = pretrained_θ
    _freeze_mode = freeze_mode
    _n_freeze_iters = n_freeze_iters
    _random_seeds = random_seeds
    _explore_frac = explore_frac

    results_grid = pmap(1:num_sims; batch_size=1) do sim_idx
        try
            seed = _random_seeds[sim_idx]
            println("Starting simulation...")
            flush(stdout)
            res = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
                bounds=bounds,
                σ_levels=σ_levels,
                n_init=12,
                n_iter=120,
                κ=κ,
                α=α,
                seed=seed,
                fidelity_threshold=fidelity_threshold,
                pretrained_θ=_pretrained_θ,
                freeze_mode=_freeze_mode,
                n_freeze_iters=_n_freeze_iters,
                explore_frac=_explore_frac,
            )

            # Always calculate true Q_det in original scale for reporting
            fcl, fsb, A = u_to_params(res.x_rec)
            Q_det_val = CalibrationCode.Q_det(t, fcl, fsb, A)
            n_iters = res.n_iter_actual > 0 ? res.n_iter_actual : 120

            # Count noise level usage
            σy_used = res.σy[1:n_iters+12]  # n_init=12 + actual iterations
            noise_counts = Dict(σ => count(==(σ), σy_used) for σ in σ_levels)

            println("Simulation $sim_idx → Q_det = $(Q_det_val), metric = $(res.y_last), iterations: $n_iters")
            flush(stdout)

            (
                sim_idx=sim_idx,
                seed=seed,
                x_rec=res.x_rec,
                y_rec=res.y_rec,
                Q_det=Q_det_val,
                Q_noisy=res.y_last,
                n_iterations=n_iters,
                noise_counts=noise_counts
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

    total_noise_counts = Dict(σ => 0 for σ in σ_levels)
    for r in results_grid
        for (σ, count) in r.noise_counts
            total_noise_counts[σ] += count
        end
    end
    total_calls = sum(values(total_noise_counts))

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
    println("\n=== NOISE LEVEL USAGE (AGGREGATE) ===")
    for σ in sort(collect(σ_levels), rev=true)
        count = total_noise_counts[σ]
        pct = 100.0 * count / total_calls
        println("σ=$(σ):  $count calls ($(round(pct, digits=1))%)")
    end
    println("Total:  $total_calls calls")
    println("\nBest noisy measurement = ", best_result.Q_noisy)
    println("\nBest u_rec = ", best_result.x_rec)

    fcl_rec, fsb_rec, A_rec = u_to_params(best_result.x_rec)
    println("\n=== BEST RESULT PHYSICAL PARAMETERS ===")
    println("Recommended f_cl = ", fcl_rec)
    println("Recommended f_sb = ", fsb_rec)
    println("Recommended A    = ", A_rec)
    println("Baseline   f_cl = ", f_cl0, "  f_sb = ", f_sb0, "  A = ", A0)

    println("\n=== All Results ===")
    σ_headers = join(["σ=$(σ)" for σ in sort(collect(σ_levels), rev=true)], "\t")
    println("Sim\tSeed\tQ_det\tQ_noisy\tIterations\t$σ_headers")
    for r in results_grid
        noise_str = join([string(r.noise_counts[σ]) for σ in sort(collect(σ_levels), rev=true)], "\t")
        println("$(r.sim_idx)\t$(r.seed)\t$(r.Q_det)\t$(r.Q_noisy)\t$(r.n_iterations)\t$noise_str")
    end

    elapsed_seconds = time() - start_time
    elapsed_total = round(Int, elapsed_seconds)
    elapsed_hours = elapsed_total ÷ 3600
    elapsed_minutes = (elapsed_total % 3600) ÷ 60
    elapsed_secs = elapsed_total % 60
    elapsed_hms = string(elapsed_hours, ":", lpad(string(elapsed_minutes), 2, '0'), ":", lpad(string(elapsed_secs), 2, '0'))
    output_file = joinpath(@__DIR__, "benchmark_results_test.txt")
    open(output_file, "w") do io
        println(io, "=== BENCHMARK RESULTS ===")
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
        println(io, "=== NOISE LEVEL USAGE (AGGREGATE) ===")
        for σ in sort(collect(σ_levels), rev=true)
            count = total_noise_counts[σ]
            pct = 100.0 * count / total_calls
            println(io, "σ=$(σ):  $count calls ($(round(pct, digits=1))%)")
        end
        println(io, "Total:  $total_calls calls")
        println(io, "")
        println(io, "Best noisy measurement = $(best_result.Q_noisy)")
        println(io, "Best u_rec = $(best_result.x_rec)")
        println(io, "")
        println(io, "=== BEST RESULT PHYSICAL PARAMETERS ===")
        println(io, "Recommended f_cl = $fcl_rec")
        println(io, "Recommended f_sb = $fsb_rec")
        println(io, "Recommended A    = $A_rec")
        println(io, "Baseline   f_cl = $f_cl0, f_sb = $f_sb0, A = $A0")
        println(io, "")
        println(io, "=== All Results ===")
        σ_headers = join(["σ=$(σ)" for σ in sort(collect(σ_levels), rev=true)], "\t")
        println(io, "Sim\tSeed\tIterations\tQ_det\t$σ_headers")
        for r in results_grid
            noise_str = join([string(r.noise_counts[σ]) for σ in sort(collect(σ_levels), rev=true)], "\t")
            println(io, "$(r.sim_idx)\t$(r.seed)\t$(r.n_iterations)\t$(r.Q_det)\t$noise_str")
        end
    end

    println("\nResults written to: $output_file")

finally
    rmprocs(workers())
end
