# Data generator for BO trace dirs feeding paper figures.
# Driven by run_traces.sh.
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
const TRACE_DIR = get(ENV, "TRACE_OUTPUT_DIR", joinpath(DATA_DIR, "traces"))
const SCRIPT_DATA_DIR = get(ENV, "TRACE_SCRIPT_DATA_DIR", joinpath(DATA_DIR))
const TWO_PI_KHZ = 2π * 1e3

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

function env_optional_float(name::String)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return nothing
    value = parse(Float64, raw)
    isfinite(value) || throw(ArgumentError("$name must be finite, got $value"))
    return value
end

env_bool(name::String, default::Bool) =
    lowercase(get(ENV, name, default ? "true" : "false")) in ("1", "true", "yes", "on")

function env_seed_list(name::String)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int[]
    return [parse(Int, strip(part)) for part in split(raw, ",") if !isempty(strip(part))]
end

function csv_field(value)
    if value isa AbstractFloat
        isnan(value) && return "NaN"
        isinf(value) && return value > 0 ? "Inf" : "-Inf"
        return @sprintf("%.17g", value)
    elseif value isa Bool
        return value ? "true" : "false"
    end
    return string(value)
end

sample_std(values) = length(values) > 1 ? std(values) : 0.0
log_infid(q) = log10(max(1.0 - clamp(Float64(q), 0.0, 1.0), 1e-6))

const N_WORKERS = env_int("TRACE_N_WORKERS", 4)
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
        using Distributions
        using StatsBase
    end

    N_SHOTS = env_shot_count("TRACE_N_SHOTS", 100)
    NUM_SIMS = env_int("TRACE_NUM_SIMS", 40)
    N_INIT = env_int("TRACE_N_INIT", 12)
    N_ITER = env_int("TRACE_N_ITER", 120)
    N_RESTARTS = env_int("TRACE_N_RESTARTS", 6)
    HYPER_EVERY = env_int("TRACE_HYPER_EVERY", 10)
    M_ACQ = env_int("TRACE_M_ACQ", 5000)
    M_REC = env_int("TRACE_M_REC", 20000)
    KAPPA = env_float("TRACE_KAPPA", 1.9)
    SCORE_MODE = Symbol(lowercase(get(ENV, "TRACE_SCORE_MODE", "standard")))
    SCORE_MODE in (:standard, :odd_penalty, :full_l1) ||
        throw(ArgumentError("TRACE_SCORE_MODE must be standard, odd_penalty, or full_l1, got $SCORE_MODE"))
    INIT_DESIGN = Symbol(lowercase(get(ENV, "TRACE_INIT_DESIGN", "random")))
    CENTER_JITTER_U_MAX = env_float("TRACE_CENTER_JITTER_U_MAX", 0.0)
    CENTER_JITTER_U_MAX >= 0.0 || throw(ArgumentError("TRACE_CENTER_JITTER_U_MAX must be >= 0"))
    BOUND_SCALE = env_float("TRACE_BOUND_SCALE", 1.0)
    BOUND_SCALE > 0.0 || throw(ArgumentError("TRACE_BOUND_SCALE must be > 0"))
    FREQ_SPAN_KHZ_OVERRIDE = env_optional_float("TRACE_FREQ_SPAN_KHZ")
    (FREQ_SPAN_KHZ_OVERRIDE === nothing || FREQ_SPAN_KHZ_OVERRIDE > 0.0) ||
        throw(ArgumentError("TRACE_FREQ_SPAN_KHZ must be > 0 when provided"))
    FIXED_INIT_SEED = parse(Int, get(ENV, "TRACE_FIXED_INIT_SEED", "0"))
    FIXED_INIT_SEED >= 0 || throw(ArgumentError("TRACE_FIXED_INIT_SEED must be >= 0"))
    OUTPUT_TAG = get(ENV, "TRACE_OUTPUT_TAG", "")
    STREAM_SEED_OUTPUTS = env_bool("TRACE_STREAM_SEED_OUTPUTS", false)

    mkpath(TRACE_DIR)
    mkpath(SCRIPT_DATA_DIR)

    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = Float64(base.f_cl), Float64(base.f_sb), Float64(base.A)
    span_kHz = 2.0
    span_fcl = FREQ_SPAN_KHZ_OVERRIDE === nothing ? span_kHz * TWO_PI_KHZ : FREQ_SPAN_KHZ_OVERRIDE * 1e3
    span_fsb = FREQ_SPAN_KHZ_OVERRIDE === nothing ? span_kHz * TWO_PI_KHZ : FREQ_SPAN_KHZ_OVERRIDE * 1e3
    freq_span_mode = FREQ_SPAN_KHZ_OVERRIDE === nothing ? "legacy_2pi_khz" : "raw_khz"
    freq_span_raw_kHz = span_fcl / 1e3
    freq_span_2pi_kHz = span_fcl / TWO_PI_KHZ
    span_A = 1.2 * A0 - A0

    @everywhere const _TRACE_T = $t
    @everywhere const _TRACE_FCL0 = $f_cl0
    @everywhere const _TRACE_FSB0 = $f_sb0
    @everywhere const _TRACE_A0 = $A0
    @everywhere const _TRACE_SPAN_FCL = $span_fcl
    @everywhere const _TRACE_SPAN_FSB = $span_fsb
    @everywhere const _TRACE_SPAN_A = $span_A
    @everywhere const _TRACE_N_SHOTS = $N_SHOTS
    @everywhere const _TRACE_N_INIT = $N_INIT
    @everywhere const _TRACE_N_ITER = $N_ITER
    @everywhere const _TRACE_N_RESTARTS = $N_RESTARTS
    @everywhere const _TRACE_HYPER_EVERY = $HYPER_EVERY
    @everywhere const _TRACE_M_ACQ = $M_ACQ
    @everywhere const _TRACE_M_REC = $M_REC
    @everywhere const _TRACE_KAPPA = $KAPPA
    @everywhere const _TRACE_SCORE_MODE = $(QuoteNode(SCORE_MODE))
    @everywhere const _TRACE_INIT_DESIGN = $(QuoteNode(INIT_DESIGN))
    @everywhere const _TRACE_CENTER_JITTER_U_MAX = $CENTER_JITTER_U_MAX
    @everywhere const _TRACE_BOUND_SCALE = $BOUND_SCALE
    @everywhere const _TRACE_FIXED_INIT_SEED = $FIXED_INIT_SEED
    @everywhere const _TRACE_CENTER_JITTER_U = Ref([0.0, 0.0, 0.0])

    @everywhere function trace_u_to_params(u)
        jitter_u = _TRACE_CENTER_JITTER_U[]
        return (
            fcl=_TRACE_FCL0 + _TRACE_SPAN_FCL * (Float64(u[1]) + jitter_u[1]),
            fsb=_TRACE_FSB0 + _TRACE_SPAN_FSB * (Float64(u[2]) + jitter_u[2]),
            A=_TRACE_A0 + _TRACE_SPAN_A * (Float64(u[3]) + jitter_u[3]),
        )
    end

    @everywhere function trace_sample_center_jitter!(rng::Random.AbstractRNG)
        if _TRACE_CENTER_JITTER_U_MAX <= 0.0
            _TRACE_CENTER_JITTER_U[] = [0.0, 0.0, 0.0]
        else
            _TRACE_CENTER_JITTER_U[] = [
                (2.0 * rand(rng) - 1.0) * _TRACE_CENTER_JITTER_U_MAX,
                (2.0 * rand(rng) - 1.0) * _TRACE_CENTER_JITTER_U_MAX,
                (2.0 * rand(rng) - 1.0) * _TRACE_CENTER_JITTER_U_MAX,
            ]
        end
        return _TRACE_CENTER_JITTER_U[]
    end

    @everywhere function trace_varms_weights(u)
        p = trace_u_to_params(u)
        subgates = [
            CalibrationCode.MSSubgate(pi / 2, 0.0),
            CalibrationCode.MSSubgate(pi / 2, 0.0),
        ]
        pulses = CalibrationCode.build_closed_loop_ms_sequence(_TRACE_T, p.fcl, p.fsb, p.A, subgates)
        pops = CalibrationCode.populations_ms_sequence(pulses)
        weights = Float64[
            max(pops.gg, 0.0),
            max(pops.eg, 0.0),
            max(pops.ge, 0.0),
            max(pops.ee, 0.0),
        ]
        weights ./= sum(weights)
        return weights
    end

    @everywhere function trace_score_from_probabilities(p_ss::Float64, p_sd::Float64,
                                                        p_ds::Float64, p_dd::Float64)
        if _TRACE_SCORE_MODE == :odd_penalty
            return clamp(p_dd - p_ss - p_sd - p_ds, 0.0, 1.0)
        elseif _TRACE_SCORE_MODE == :full_l1
            l1 = abs(p_ss) + abs(p_sd) + abs(p_ds) + abs(p_dd - 1.0)
            return clamp(1.0 - 0.5 * l1, 0.0, 1.0)
        end
        return clamp(p_dd - p_ss, 0.0, 1.0)
    end

    @everywhere function trace_score_noise_std(weights::Vector{Float64}, N::Real)
        n = Float64(N)
        isinf(n) && return 0.0
        p_ss, p_sd, p_ds, p_dd = weights
        if _TRACE_SCORE_MODE == :odd_penalty
            raw = p_dd - p_ss - p_sd - p_ds
            var_one = 1.0 - raw * raw
        elseif _TRACE_SCORE_MODE == :full_l1
            var_one = p_dd * (1.0 - p_dd)
        else
            raw = p_dd - p_ss
            var_one = p_dd + p_ss - raw * raw
        end
        return sqrt(max(var_one, 0.0) / n)
    end

    @everywhere function trace_score_from_weights(weights::Vector{Float64}, N::Real)
        n = Float64(N)
        if isinf(n)
            return trace_score_from_probabilities(weights[1], weights[2], weights[3], weights[4])
        end
        n_int = Int(n)
        counts = rand(Distributions.Multinomial(n_int, weights))
        p_ss = counts[1] / n_int
        p_sd = counts[2] / n_int
        p_ds = counts[3] / n_int
        p_dd = counts[4] / n_int
        return trace_score_from_probabilities(p_ss, p_sd, p_ds, p_dd)
    end

    @everywhere function trace_q_fun(u, N::Real)
        weights = trace_varms_weights(u)
        y = trace_score_from_weights(weights, N)
        σ = trace_score_noise_std(weights, N)
        return y, σ
    end

    @everywhere trace_q_det(u) = trace_score_from_weights(trace_varms_weights(u), Inf)

    @everywhere function trace_rand_in_box(rng::Random.AbstractRNG,
                                           lb::Vector{Float64},
                                           ub::Vector{Float64})
        x = Vector{Float64}(undef, length(lb))
        @inbounds for j in eachindex(lb)
            x[j] = rand(rng) * (ub[j] - lb[j]) + lb[j]
        end
        return x
    end

    @everywhere function trace_far_enough(x::Vector{Float64}, X::Matrix{Float64}, n::Int;
                                          min_dist::Float64=1e-4)
        @inbounds for i in 1:n
            d2 = 0.0
            for j in eachindex(x)
                delta = x[j] - X[j, i]
                d2 += delta * delta
            end
            d2 < min_dist * min_dist && return false
        end
        return true
    end

    @everywhere function trace_initial_points(rng::Random.AbstractRNG,
                                              lb::Vector{Float64},
                                              ub::Vector{Float64})
        if _TRACE_INIT_DESIGN != :random
            return CalibrationCode._path_guard_initial_points(
                rng, lb, ub, _TRACE_N_INIT,
                CalibrationCode.PathGuardConfig(init_design=_TRACE_INIT_DESIGN))
        end
        return [trace_rand_in_box(rng, lb, ub) for _ in 1:_TRACE_N_INIT]
    end

    @everywhere function trace_fill_initial_design!(X::Matrix{Float64}, y::Vector{Float64},
                                                    σy::Vector{Float64},
                                                    rng::Random.AbstractRNG,
                                                    lb::Vector{Float64},
                                                    ub::Vector{Float64})
        write_idx = 0
        shots = 0
        for x in trace_initial_points(rng, lb, ub)
            y_raw, σ_raw = trace_q_fun(x, _TRACE_N_SHOTS)
            shots += _TRACE_N_SHOTS
            write_idx += 1
            X[:, write_idx] = x
            y[write_idx] = y_raw
            σy[write_idx] = σ_raw
        end
        return write_idx, shots
    end

    @everywhere trace_ucb_score(μ::Float64, s2::Float64, κ::Float64) = μ + κ * sqrt(max(s2, 0.0))

    @everywhere function trace_run_seed(seed::Int)
        init_seed = _TRACE_FIXED_INIT_SEED > 0 ? _TRACE_FIXED_INIT_SEED : seed
        Random.seed!(init_seed)
        rng = MersenneTwister(init_seed)
        jitter_u = trace_sample_center_jitter!(rng)
        bounds = [(-_TRACE_BOUND_SCALE, _TRACE_BOUND_SCALE),
                  (-_TRACE_BOUND_SCALE, _TRACE_BOUND_SCALE),
                  (-_TRACE_BOUND_SCALE, _TRACE_BOUND_SCALE)]
        lb = Float64[b[1] for b in bounds]
        ub = Float64[b[2] for b in bounds]
        n_cap = _TRACE_N_INIT + _TRACE_N_ITER + 2
        X = Matrix{Float64}(undef, 3, n_cap)
        y = Vector{Float64}(undef, n_cap)
        σy = Vector{Float64}(undef, n_cap)
        total_shots = 0

        write_idx, init_shots = trace_fill_initial_design!(X, y, σy, rng, lb, ub)
        total_shots += init_shots
        if _TRACE_FIXED_INIT_SEED > 0
            Random.seed!(seed)
            rng = MersenneTwister(seed)
        end

        θ_prev = nothing
        trace_rows = NamedTuple[]

        for it in 1:_TRACE_N_ITER
            do_opt = (it == 1) || (_TRACE_HYPER_EVERY > 0 && it % _TRACE_HYPER_EVERY == 0)
            gp = CalibrationCode.fit_heterogp(
                X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
                θ_init=θ_prev,
                learn_hypers=do_opt,
                learn_noise_scale=true,
                n_restarts=do_opt ? _TRACE_N_RESTARTS : 0,
                jitter=1e-8,
                rng=rng,
            )
            θ_prev = gp.θ

            x_acq = trace_rand_in_box(rng, lb, ub)
            best_a = -Inf
            m_acq = NaN
            s_acq = NaN
            for _ in 1:_TRACE_M_ACQ
                x = trace_rand_in_box(rng, lb, ub)
                μ, s2 = CalibrationCode.predict_latent(gp, x)
                a = trace_ucb_score(μ, s2, _TRACE_KAPPA)
                if a > best_a
                    best_a = a
                    x_acq = x
                    m_acq = μ
                    s_acq = sqrt(max(s2, 0.0))
                end
            end

            y_acq, σ_acq = trace_q_fun(x_acq, _TRACE_N_SHOTS)
            total_shots += _TRACE_N_SHOTS
            if trace_far_enough(x_acq, X, write_idx)
                write_idx += 1
                X[:, write_idx] = x_acq
                y[write_idx] = y_acq
                σy[write_idx] = σ_acq
            end

            x_rec, m_rec_raw, s_rec = CalibrationCode.recommend_mean(gp, bounds; M=_TRACE_M_REC, rng=rng)
            push!(trace_rows, (
                iter=it,
                total_shots=total_shots,
                n_train=write_idx,
                m_rec=clamp(m_rec_raw, 0.0, 1.0),
                m_rec_raw=m_rec_raw,
                s_rec=s_rec,
                x_rec_u1=x_rec[1],
                x_rec_u2=x_rec[2],
                x_rec_u3=x_rec[3],
                x_acq_u1=x_acq[1],
                x_acq_u2=x_acq[2],
                x_acq_u3=x_acq[3],
                m_acq=m_acq,
                s_acq=s_acq,
                acq_score=best_a,
                y_acq=y_acq,
                q_det_rec=trace_q_det(x_rec),
                q_det_acq=trace_q_det(x_acq),
                center_jitter_u1=jitter_u[1],
                center_jitter_u2=jitter_u[2],
                center_jitter_u3=jitter_u[3],
                true_opt_u1=-jitter_u[1],
                true_opt_u2=-jitter_u[2],
                true_opt_u3=-jitter_u[3],
            ))
        end

        gp = CalibrationCode.fit_heterogp(
            X[:, 1:write_idx], y[1:write_idx], σy[1:write_idx];
            θ_init=θ_prev,
            learn_hypers=true,
            learn_noise_scale=true,
            n_restarts=_TRACE_N_RESTARTS + 2,
            jitter=1e-8,
            rng=rng,
        )
        final_x, _, _ = CalibrationCode.recommend_mean(gp, bounds; M=_TRACE_M_REC, rng=rng)
        final_y_rec, _ = trace_q_fun(final_x, _TRACE_N_SHOTS)
        total_shots += _TRACE_N_SHOTS
        return (
            seed=seed,
            trace_rows=trace_rows,
            x_rec=final_x,
            y_rec=final_y_rec,
            total_shots=total_shots,
            n_train=write_idx,
            q_det=trace_q_det(final_x),
            jitter_u=jitter_u,
            init_seed=init_seed,
            ell=gp.ℓ,
            sigma_f=gp.σf,
            c=gp.c,
        )
    end

    fixed_seeds = [
        714078, 849665, 670733, 400294, 909858, 473966, 981559, 318670, 225142, 359405,
        250215, 558664, 438880,   5937, 615903, 150574, 963284, 473867, 150918, 955377,
        902578,  61646, 197255, 462583, 184672, 831702, 720308,  16729, 387749, 215846,
        312561, 749598, 631837, 746550, 709734, 181983, 279125, 965652, 419030, 888571,
    ]
    requested_seeds = env_seed_list("TRACE_SEEDS")
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
    println("score_mode=$SCORE_MODE  kappa=$KAPPA  init_design=$INIT_DESIGN")
    println("bound_scale=$BOUND_SCALE  freq_span_mode=$freq_span_mode  freq_span_raw_kHz=$freq_span_raw_kHz")
    println("seeds=$(join(seeds, ","))")

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
        "seed", "initial_seed", "N_shots", "total_shots", "n_train",
        "q_det", "q_rec",
        "center_jitter_u1", "center_jitter_u2", "center_jitter_u3",
        "true_opt_u1", "true_opt_u2", "true_opt_u3",
        "ell1", "ell2", "ell3", "sigma_f", "c", "elapsed_seconds",
    ]
    seed_progress_path = joinpath(TRACE_DIR, "seed_progress.csv")
    if STREAM_SEED_OUTPUTS
        open(seed_progress_path, "w") do io
            println(io, join(seed_progress_header, ","))
        end
    end
    @everywhere const _TRACE_STREAM_SEED_OUTPUTS = $STREAM_SEED_OUTPUTS
    @everywhere const _TRACE_OUTPUT_DIR = $TRACE_DIR
    @everywhere const _TRACE_PROGRESS_CSV = $seed_progress_path
    @everywhere const _TRACE_TRACE_HEADER = $trace_header
    @everywhere const _TRACE_SEED_PROGRESS_HEADER = $seed_progress_header
    @everywhere function trace_csv_field(value)
        if value isa AbstractFloat
            isnan(value) && return "NaN"
            isinf(value) && return value > 0 ? "Inf" : "-Inf"
            return @sprintf("%.17g", value)
        elseif value isa Bool
            return value ? "true" : "false"
        end
        return string(value)
    end
    @everywhere function trace_write_seed_trace_csv(result)
        mkpath(_TRACE_OUTPUT_DIR)
        path = joinpath(_TRACE_OUTPUT_DIR, "trace_seed$(result.seed)_ucb.csv")
        open(path, "w") do io
            println(io, join(_TRACE_TRACE_HEADER, ","))
            for row in result.trace_rows
                println(io, join((trace_csv_field(getproperty(row, Symbol(name))) for name in _TRACE_TRACE_HEADER), ","))
            end
        end
    end
    @everywhere function trace_with_progress_lock(f::Function)
        lockdir = _TRACE_PROGRESS_CSV * ".lockdir"
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
    @everywhere function trace_append_seed_progress_csv(result, seed_elapsed::Float64)
        row = [
            result.seed, result.init_seed, _TRACE_N_SHOTS,
            result.total_shots, result.n_train, result.q_det, result.y_rec,
            result.jitter_u[1], result.jitter_u[2], result.jitter_u[3],
            -result.jitter_u[1], -result.jitter_u[2], -result.jitter_u[3],
            result.ell[1], result.ell[2], result.ell[3], result.sigma_f, result.c,
            seed_elapsed,
        ]
        trace_with_progress_lock() do
            open(_TRACE_PROGRESS_CSV, "a") do io
                println(io, join(trace_csv_field.(row), ","))
            end
        end
    end
    @everywhere function trace_write_seed_outputs(result, seed_elapsed::Float64)
        !_TRACE_STREAM_SEED_OUTPUTS && return nothing
        trace_write_seed_trace_csv(result)
        trace_append_seed_progress_csv(result, seed_elapsed)
    end

    start_time = time()

    results = pmap(1:NUM_SIMS; batch_size=1) do idx
        seed = seeds[idx]
        println("Starting trace seed $seed ($idx/$NUM_SIMS)")
        flush(stdout)
        seed_start = time()
        result = trace_run_seed(seed)
        seed_elapsed = time() - seed_start
        trace_write_seed_outputs(result, seed_elapsed)
        result = merge(result, (seed_elapsed=seed_elapsed,))
        println("Finished seed $seed: Q_det=$(round(result.q_det, digits=5)), elapsed_s=$(round(seed_elapsed, digits=2))")
        flush(stdout)
        return result
    end

    if !STREAM_SEED_OUTPUTS
        for result in results
            path = joinpath(TRACE_DIR, "trace_seed$(result.seed)_ucb.csv")
            open(path, "w") do io
                println(io, join(trace_header, ","))
                for row in result.trace_rows
                    println(io, join((csv_field(getproperty(row, Symbol(name))) for name in trace_header), ","))
                end
            end
        end
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
        println(io, "score_mode = $SCORE_MODE")
        println(io, "kappa = $KAPPA")
        println(io, "init_design = $INIT_DESIGN")
        println(io, "bound_scale = $BOUND_SCALE")
        println(io, "freq_span_mode = $freq_span_mode")
        println(io, "freq_span_raw_kHz = $freq_span_raw_kHz")
        println(io, "freq_span_2pi_kHz = $freq_span_2pi_kHz")
        println(io, "center_jitter_u_max = $CENTER_JITTER_U_MAX")
        println(io, "fixed_init_seed = $FIXED_INIT_SEED")
        println(io, "M_acq = $M_ACQ")
        println(io, "M_rec = $M_REC")
        println(io, "stream_seed_outputs = $STREAM_SEED_OUTPUTS")
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
        println(io, "Sim\tSeed\tInitialSeed\tTotalShots\tNTrain\tQ_det\tQ_rec\tcenter_jitter_u1\tcenter_jitter_u2\tcenter_jitter_u3\ttrue_opt_u1\ttrue_opt_u2\ttrue_opt_u3\tell1\tell2\tell3\tsigma_f\tc\tElapsedSeconds")
        for (idx, r) in enumerate(results)
            println(io, join(csv_field.([
                idx, r.seed, r.init_seed,
                r.total_shots, r.n_train, r.q_det, r.y_rec,
                r.jitter_u[1], r.jitter_u[2], r.jitter_u[3],
                -r.jitter_u[1], -r.jitter_u[2], -r.jitter_u[3],
                r.ell[1], r.ell[2], r.ell[3], r.sigma_f, r.c, r.seed_elapsed,
            ]), "\t"))
        end
    end

    println("Trace files written to: $TRACE_DIR")
    println("Summary written to: $output_path")
finally
    rmprocs(workers())
end
