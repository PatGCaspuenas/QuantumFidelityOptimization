# Data generator for BO trace dirs feeding paper figures.
# Driven by run_main.sh.
#
# Outputs:
#   data/traces_*/trace_seed<seed>_ucb.csv   (per-seed BO trace)
#   data/traces_*/seed_progress.csv          (summary row per seed)
#   data/benchmark_trace_N<N>_<tag>.txt      (aggregate statistics)

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf
using Random
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO_ROOT, "data")
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", joinpath(DATA_DIR, "traces"))
const SCRIPT_DATA_DIR = get(ENV, "SCRIPT_DATA_DIR", joinpath(DATA_DIR))

function env_int(name::String, default::Int)
    value = parse(Int, get(ENV, name, string(default)))
    value >= 1 || throw(ArgumentError("$name must be >= 1, got $value"))
    return value
end

function env_shot_count(name::String, default::Int)
    raw = strip(get(ENV, name, string(default)))
    lowercase(raw) in ("inf", "+inf", "infinity", "+infinity") && return Inf
    value = parse(Int, raw)
    value >= 1 || throw(ArgumentError("$name must be >= 1 or Inf, got $raw"))
    return value
end

function env_float(name::String, default::Float64)
    value = parse(Float64, get(ENV, name, string(default)))
    isfinite(value) || throw(ArgumentError("$name must be finite, got $value"))
    return value
end

function env_seed_list(name::String)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int[]
    return [parse(Int, strip(part)) for part in split(raw, ",") if !isempty(strip(part))]
end

sample_std(values) = length(values) > 1 ? std(values) : 0.0
log_infid(q) = log10(max(1.0 - clamp(Float64(q), 0.0, 1.0), 1e-6))

const N_WORKERS = env_int("N_WORKERS", 4)
if nprocs() == 1
    addprocs(N_WORKERS)
end

try
    @everywhere begin
        import Pkg
        Pkg.activate(joinpath($REPO_ROOT); io=devnull)
        include(joinpath($REPO_ROOT, "src", "CalibrationCode.jl"))
        using Random
        using Printf
    end

    # --- configuration (main process) ---
    N_SHOTS = env_shot_count("N_SHOTS", 100)
    NUM_SIMS = env_int("NUM_SIMS", 40)
    N_INIT = env_int("N_INIT", 12)
    N_ITER = env_int("N_ITER", 120)
    N_RESTARTS = env_int("N_RESTARTS", 6)
    HYPER_EVERY = env_int("HYPER_EVERY", 10)
    M_ACQ = env_int("M_ACQ", 5000)
    M_REC = env_int("M_REC", 20000)
    KAPPA = env_float("KAPPA", 1.96)
    JITTER_MAX = env_float("CENTER_JITTER_U_MAX", 0.0)
    JITTER_MAX >= 0.0 || throw(ArgumentError("CENTER_JITTER_U_MAX must be >= 0"))
    BOUND_SCALE = env_float("BOUND_SCALE", 1.0)
    BOUND_SCALE > 0.0 || throw(ArgumentError("BOUND_SCALE must be > 0"))
    FREQ_SPAN_KHZ = env_float("FREQ_SPAN_KHZ", 10.0)
    FREQ_SPAN_KHZ > 0.0 || throw(ArgumentError("FREQ_SPAN_KHZ must be > 0"))
    OUTPUT_TAG = get(ENV, "OUTPUT_TAG", "")

    mkpath(OUTPUT_DIR)
    mkpath(SCRIPT_DATA_DIR)

    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = Float64(base.f_cl), Float64(base.f_sb), Float64(base.A)
    span_freq = FREQ_SPAN_KHZ * 1e3   # Hz; same span for the f_cl and f_sb axes
    span_A = 1.2 * A0 - A0

    trace_header = [
        "iter", "total_shots", "n_train",
        "m_rec", "m_rec_raw", "s_rec",
        "x_rec_u1", "x_rec_u2", "x_rec_u3",
        "x_acq_u1", "x_acq_u2", "x_acq_u3",
        "m_acq", "s_acq", "acq_score", "y_acq",
        "q_det_rec", "q_det_acq",
        "center_jitter_u1", "center_jitter_u2", "center_jitter_u3",
        "true_opt_u1", "true_opt_u2", "true_opt_u3",
    ]
    seed_progress_header = [
        "seed", "N_shots", "total_shots", "n_train",
        "q_det", "q_rec",
        "center_jitter_u1", "center_jitter_u2", "center_jitter_u3",
        "true_opt_u1", "true_opt_u2", "true_opt_u3",
        "ell1", "ell2", "ell3", "sigma_f", "c", "elapsed_seconds",
    ]
    progress_csv = joinpath(OUTPUT_DIR, "seed_progress.csv")

    # Single config bundle broadcast to all workers.
    cfg = (
        t=t, fcl0=f_cl0, fsb0=f_sb0, a0=A0,
        span_freq=span_freq, span_a=span_A,
        n_shots=N_SHOTS, n_init=N_INIT, n_iter=N_ITER, n_restarts=N_RESTARTS,
        hyper_every=HYPER_EVERY, m_acq=M_ACQ, m_rec=M_REC, kappa=KAPPA,
        jitter_max=JITTER_MAX, bound_scale=BOUND_SCALE,
        output_dir=OUTPUT_DIR, progress_csv=progress_csv,
        trace_header=trace_header, progress_header=seed_progress_header,
    )
    @everywhere const CFG = $cfg

    @everywhere function csv_field(value)
        if value isa AbstractFloat
            isnan(value) && return "NaN"
            isinf(value) && return value > 0 ? "Inf" : "-Inf"
            return @sprintf("%.17g", value)
        elseif value isa Bool
            return value ? "true" : "false"
        end
        return string(value)
    end

    # One BO run for a single seed. The per-iteration convergence trace is
    # collected by an `on_iter` callback handed to bayesopt_ucb, so the loop
    # itself lives in src/bayes_opt.jl. The u→physical-parameter mapping and the
    # per-seed center jitter are study-specific and stay local here.
    @everywhere function run_seed(seed::Int)
        Random.seed!(seed)   # seeds the global RNG used by Q_varMS projection sampling
        jitter = CFG.jitter_max <= 0.0 ? (0.0, 0.0, 0.0) :
            ((2rand() - 1) * CFG.jitter_max,
             (2rand() - 1) * CFG.jitter_max,
             (2rand() - 1) * CFG.jitter_max)

        u_to_params(u) = (
            fcl = CFG.fcl0 + CFG.span_freq * (Float64(u[1]) + jitter[1]),
            fsb = CFG.fsb0 + CFG.span_freq * (Float64(u[2]) + jitter[2]),
            A   = CFG.a0   + CFG.span_a    * (Float64(u[3]) + jitter[3]),
        )
        objective(u, N) = begin
            p = u_to_params(u)
            CalibrationCode.Q_varMS(CFG.t, p.fcl, p.fsb, p.A; N=N)
        end
        qdet(u) = objective(u, Inf)[1]
        bounds = [(-CFG.bound_scale, CFG.bound_scale) for _ in 1:3]

        rows = NamedTuple[]
        record = st -> begin
            x_rec, m_rec_raw, s_rec =
                CalibrationCode.recommend_mean(st.gp, bounds; M=CFG.m_rec, rng=st.rng)
            push!(rows, (
                iter=st.it, total_shots=st.total_shots, n_train=st.n_train,
                m_rec=clamp(m_rec_raw, 0.0, 1.0), m_rec_raw=m_rec_raw, s_rec=s_rec,
                x_rec_u1=x_rec[1], x_rec_u2=x_rec[2], x_rec_u3=x_rec[3],
                x_acq_u1=st.x_acq[1], x_acq_u2=st.x_acq[2], x_acq_u3=st.x_acq[3],
                m_acq=st.m_acq, s_acq=st.s_acq, acq_score=st.acq_score, y_acq=st.y_acq,
                q_det_rec=qdet(x_rec), q_det_acq=qdet(st.x_acq),
                center_jitter_u1=jitter[1], center_jitter_u2=jitter[2], center_jitter_u3=jitter[3],
                true_opt_u1=-jitter[1], true_opt_u2=-jitter[2], true_opt_u3=-jitter[3],
            ))
        end

        res = CalibrationCode.bayesopt_ucb(objective; bounds=bounds,
            n_shots=CFG.n_shots, n_init=CFG.n_init, n_iter=CFG.n_iter,
            M_acq=CFG.m_acq, M_rec=CFG.m_rec, κ=CFG.kappa,
            hyper_every=CFG.hyper_every, n_restarts=CFG.n_restarts,
            seed=seed, on_iter=record)

        return (
            seed=seed, trace_rows=rows,
            x_rec=res.x_rec, y_rec=res.y_rec,
            total_shots=res.total_shots, n_train=size(res.X, 2),
            q_det=qdet(res.x_rec), jitter_u=collect(jitter),
            ell=res.ℓ_final, sigma_f=res.σf_final, c=res.c_final,
        )
    end

    @everywhere function write_seed_trace_csv(result)
        mkpath(CFG.output_dir)
        path = joinpath(CFG.output_dir, "trace_seed$(result.seed)_ucb.csv")
        open(path, "w") do io
            println(io, join(CFG.trace_header, ","))
            for row in result.trace_rows
                println(io, join((csv_field(getproperty(row, Symbol(name))) for name in CFG.trace_header), ","))
            end
        end
    end

    @everywhere function with_progress_lock(f::Function)
        lockdir = CFG.progress_csv * ".lockdir"
        attempts = 0
        while true
            try
                mkdir(lockdir)
                break
            catch
                attempts += 1
                if attempts % 200 == 0 && isdir(lockdir)
                    try rm(lockdir; recursive=true, force=true) catch end
                end
                sleep(0.05)
            end
        end
        try
            return f()
        finally
            try rm(lockdir; recursive=true, force=true) catch end
        end
    end

    @everywhere function append_seed_progress_csv(result, seed_elapsed::Float64)
        row = [
            result.seed, CFG.n_shots,
            result.total_shots, result.n_train, result.q_det, result.y_rec,
            result.jitter_u[1], result.jitter_u[2], result.jitter_u[3],
            -result.jitter_u[1], -result.jitter_u[2], -result.jitter_u[3],
            result.ell[1], result.ell[2], result.ell[3], result.sigma_f, result.c,
            seed_elapsed,
        ]
        with_progress_lock() do
            open(CFG.progress_csv, "a") do io
                println(io, join(csv_field.(row), ","))
            end
        end
    end

    @everywhere function write_seed_outputs(result, seed_elapsed::Float64)
        write_seed_trace_csv(result)
        append_seed_progress_csv(result, seed_elapsed)
    end

    fixed_seeds = [
        714078, 849665, 670733, 400294, 909858, 473966, 981559, 318670, 225142, 359405,
        250215, 558664, 438880,   5937, 615903, 150574, 963284, 473867, 150918, 955377,
        902578,  61646, 197255, 462583, 184672, 831702, 720308,  16729, 387749, 215846,
        312561, 749598, 631837, 746550, 709734, 181983, 279125, 965652, 419030, 888571,
        567906, 968356, 682041, 184183, 228167, 644045, 795829,  55163, 175307, 218892,
        155832, 262269, 273956, 828743, 656460, 893076, 503506, 245877, 359706, 246715,
        690925, 696535, 795298, 535117, 310012, 266070, 510043, 364834,  50978, 983225,
         20510,  39528, 518624, 279217, 769063, 603072, 265386, 191889, 461212, 707673,
         29044, 621551, 460176, 632485, 705471, 617573, 950438, 607415, 321202, 421887,
        337617, 666520, 205394, 889047,    526,  29498, 831472, 810138, 105026, 647322,
    ]
    requested_seeds = env_seed_list("SEEDS")
    seeds = if !isempty(requested_seeds)
        requested_seeds
    elseif NUM_SIMS <= length(fixed_seeds)
        fixed_seeds[1:NUM_SIMS]
    else
        vcat(fixed_seeds, rand(MersenneTwister(1), 1:1_000_000, NUM_SIMS - length(fixed_seeds)))
    end
    NUM_SIMS = length(seeds)

    println("=== Trace benchmark export ===")
    println("N=$N_SHOTS  seeds=$NUM_SIMS  workers=$(nworkers())  n_init=$N_INIT  n_iter=$N_ITER")
    println("kappa=$KAPPA  init_design=latin_hypercube")
    println("bound_scale=$BOUND_SCALE  freq_span_kHz=$FREQ_SPAN_KHZ")
    println("seeds=$(join(seeds, ","))")

    open(progress_csv, "w") do io
        println(io, join(seed_progress_header, ","))
    end

    start_time = time()

    results = pmap(1:NUM_SIMS; batch_size=1) do idx
        seed = seeds[idx]
        println("Starting trace seed $seed ($idx/$NUM_SIMS)")
        flush(stdout)
        seed_start = time()
        result = run_seed(seed)
        seed_elapsed = time() - seed_start
        write_seed_outputs(result, seed_elapsed)
        result = merge(result, (seed_elapsed=seed_elapsed,))
        println("Finished seed $seed: Q_det=$(round(result.q_det, digits=5)), elapsed_s=$(round(seed_elapsed, digits=2))")
        flush(stdout)
        return result
    end

    elapsed = time() - start_time
    q_values = [r.q_det for r in results]
    shot_values = [r.total_shots for r in results]
    seed_elapsed_values = [r.seed_elapsed for r in results]
    output_stem = isempty(OUTPUT_TAG) ? "benchmark_trace_N$(N_SHOTS)" : "benchmark_trace_N$(N_SHOTS)_$(OUTPUT_TAG)"
    output_path = joinpath(SCRIPT_DATA_DIR, "$(output_stem).txt")
    open(output_path, "w") do io
        println(io, "=== TRACE BENCHMARK RESULTS ===")
        println(io, "N_shots = $N_SHOTS")
        println(io, "n_init = $N_INIT")
        println(io, "n_iter = $N_ITER")
        println(io, "num_sims = $NUM_SIMS")
        println(io, "workers = $(nworkers())")
        println(io, "kappa = $KAPPA")
        println(io, "init_design = latin_hypercube")
        println(io, "bound_scale = $BOUND_SCALE")
        println(io, "freq_span_kHz = $FREQ_SPAN_KHZ")
        println(io, "center_jitter_u_max = $JITTER_MAX")
        println(io, "M_acq = $M_ACQ")
        println(io, "M_rec = $M_REC")
        println(io, "elapsed_seconds = $elapsed")
        println(io, "")
        println(io, "=== RESULTS SUMMARY ===")
        println(io, "Average Q_det = $(mean(q_values)) +/- $(sample_std(q_values))")
        println(io, "Median Q_det = $(median(q_values))")
        println(io, "Min Q_det = $(minimum(q_values))")
        println(io, "Max Q_det = $(maximum(q_values))")
        println(io, "Average log10(1-Q_det) = $(mean(log_infid.(q_values))) +/- $(sample_std(log_infid.(q_values)))")
        println(io, "Average total shots = $(mean(shot_values))")
        println(io, "Average seed elapsed seconds = $(mean(seed_elapsed_values)) +/- $(sample_std(seed_elapsed_values))")
        println(io, "")
        println(io, "=== All Results ===")
        println(io, "Sim\tSeed\tTotalShots\tNTrain\tQ_det\tQ_rec\tcenter_jitter_u1\tcenter_jitter_u2\tcenter_jitter_u3\ttrue_opt_u1\ttrue_opt_u2\ttrue_opt_u3\tell1\tell2\tell3\tsigma_f\tc\tElapsedSeconds")
        for (idx, r) in enumerate(results)
            println(io, join(csv_field.([
                idx, r.seed,
                r.total_shots, r.n_train, r.q_det, r.y_rec,
                r.jitter_u[1], r.jitter_u[2], r.jitter_u[3],
                -r.jitter_u[1], -r.jitter_u[2], -r.jitter_u[3],
                r.ell[1], r.ell[2], r.ell[3], r.sigma_f, r.c, r.seed_elapsed,
            ]), "\t"))
        end
    end

    println("Trace files written to: $OUTPUT_DIR")
    println("Summary written to: $output_path")
finally
    rmprocs(workers())
end
