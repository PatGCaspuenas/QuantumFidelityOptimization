# Data generator for paper figures figures/paper/figure_ab_vertical.png and
# figures/paper/figure_fcl_amp_n100_gpzoom.png: writes the BO trace dirs
# data/traces_freqspan10_* (scale sweep + selected-N full_l1) those figures read.
# Driven by run_scale05_selectedN_score_sweep.sh (and direct env-var invocations
# for the bound010/050/100 NInf scale sweeps).
#
# Fixed-N 2MS UCB benchmark with per-iteration trace export for Python plots.
#
# Outputs:
#   data/traces/trace_seed<seed>_ucb.csv
#   data/slices/slice_u1u2.csv, slice_u1u3.csv, slice_u2u3.csv
#   scripts/data/benchmark_trace_N<N>.txt
#
# Stopping rule:
#   1. Recommend x_rec from the GP posterior mean.
#   2. TRACE_STOPPING_RULE=two_noisy:
#        use the normal noisy acquisition measurement as the first check. If it
#        clears the threshold, take one more noisy measurement at that same
#        acquisition point and stop only if that second measurement also clears
#        the threshold.
#   3. TRACE_STOPPING_RULE=surrogate_then_noisy:
#        first require clamp(m_rec, 0, 1) - TRACE_CONFIDENCE_Z * s_rec >=
#        TRACE_THRESH_Q. If the surrogate gate passes, take one noisy
#        measurement at x_rec and stop if that measurement clears the threshold.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf
using Random
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO_ROOT, "data")
const TRACE_DIR = get(ENV, "TRACE_OUTPUT_DIR", joinpath(DATA_DIR, "traces"))
const SLICE_DIR = joinpath(DATA_DIR, "slices")
const GP_DIAG_DIR = get(ENV, "TRACE_GP_DIAG_DIR", joinpath(DATA_DIR, "gp_diagnostics"))
const SCRIPT_DATA_DIR = get(ENV, "TRACE_SCRIPT_DATA_DIR", joinpath(@__DIR__, "data"))
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

function env_stopping_rule(require_surrogate_confidence::Bool)
    default = require_surrogate_confidence ? "surrogate_then_noisy" : "two_noisy"
    rule = lowercase(get(ENV, "TRACE_STOPPING_RULE", default))
    aliases = Dict(
        "measurement_only" => "two_noisy",
        "noisy_twostep" => "two_noisy",
        "two_step_noisy" => "two_noisy",
        "surrogate_confidence" => "surrogate_then_noisy",
        "surrogate_then_measurement" => "surrogate_then_noisy",
        "surrogate" => "surrogate_then_noisy",
    )
    rule = get(aliases, rule, rule)
    rule in ("two_noisy", "surrogate_then_noisy") ||
        throw(ArgumentError("TRACE_STOPPING_RULE must be two_noisy or surrogate_then_noisy, got $rule"))
    return rule
end

function env_seed_list(name::String)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int[]
    return [parse(Int, strip(part)) for part in split(raw, ",") if !isempty(strip(part))]
end

function env_int_list(name::String, default::String)
    raw = strip(get(ENV, name, default))
    isempty(raw) && return Int[]
    values = [parse(Int, strip(part)) for part in split(raw, ",") if !isempty(strip(part))]
    any(<(1), values) && throw(ArgumentError("$name entries must be >= 1"))
    return sort(unique(values))
end

function env_axis_list(name::String, default::String)
    raw = strip(get(ENV, name, default))
    isempty(raw) && return Symbol[]
    allowed = Set([:fcl, :fsb, :amp])
    axes = Symbol[]
    for part in split(raw, ",")
        token = lowercase(strip(part))
        isempty(token) && continue
        axis = token in ("fcl", "u1", "cl") ? :fcl :
               token in ("fsb", "u2", "sb") ? :fsb :
               token in ("amp", "a", "u3") ? :amp :
               throw(ArgumentError("Unsupported axis '$token' in $name"))
        axis in allowed || throw(ArgumentError("Unsupported axis '$token' in $name"))
        axis in axes || push!(axes, axis)
    end
    return axes
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
    ALPHA = env_float("TRACE_ALPHA", 1.5)
    THRESHOLD = env_float("TRACE_THRESH_Q", 0.99)
    DISABLE_STOPPING = env_bool("TRACE_DISABLE_STOPPING", false)
    CONFIDENCE_Z = env_float("TRACE_CONFIDENCE_Z", 1.0)
    REQUIRE_SURROGATE_CONFIDENCE = env_bool("TRACE_REQUIRE_SURROGATE_CONFIDENCE", true)
    STOPPING_RULE = env_stopping_rule(REQUIRE_SURROGATE_CONFIDENCE)
    SCORE_MODE = Symbol(lowercase(get(ENV, "TRACE_SCORE_MODE", "standard")))
    SCORE_MODE in (:standard, :odd_penalty, :full_l1) ||
        throw(ArgumentError("TRACE_SCORE_MODE must be standard, odd_penalty, or full_l1, got $SCORE_MODE"))
    PATH_GUARD = env_bool("TRACE_PATH_GUARD", false)
    PATH_INIT_DESIGN = Symbol(lowercase(get(ENV, "TRACE_INIT_DESIGN", "random")))
    PATH_INIT_CENTER_EXCLUSION = env_float("TRACE_INIT_CENTER_EXCLUSION", 0.0)
    PATH_INIT_SOBOL_SKIP_MAX = env_int("TRACE_INIT_SOBOL_SKIP_MAX", 2048)
    DECISION_Q_MAX = env_float("TRACE_DECISION_Q_MAX", 1.0)
    PATH_GLOBAL_SCOUT_PERIOD = env_int("TRACE_GLOBAL_SCOUT_PERIOD", 7)
    PATH_GLOBAL_SCOUT_FRAC = env_float("TRACE_GLOBAL_SCOUT_FRAC", 0.15)
    PATH_STAGNATION_WINDOW = env_int("TRACE_STAGNATION_WINDOW", 12)
    PATH_BOUNDARY_MARGIN = env_float("TRACE_BOUNDARY_MARGIN", 0.03)
    PATH_ELL_HI = env_float("TRACE_HEALTH_ELL_HI", 1.45)
    PATH_SIGMAF_LO = env_float("TRACE_HEALTH_SIGMAF_LO", 0.35)
    PATH_C_HI = env_float("TRACE_HEALTH_C_HI", 2.5)
    PATH_TRUST_START_Q = env_float("TRACE_TRUST_START_Q", 0.70)
    PATH_STRONG_TRUST_Q = env_float("TRACE_STRONG_TRUST_Q", 0.90)
    PATH_TRUST_RADIUS = env_float("TRACE_TRUST_RADIUS", 0.25)
    PATH_TRUST_RADIUS_MIN = env_float("TRACE_TRUST_RADIUS_MIN", 0.12)
    PATH_MID_LOCAL_FRAC = env_float("TRACE_MID_LOCAL_FRAC", 0.35)
    PATH_STRONG_LOCAL_FRAC = env_float("TRACE_STRONG_LOCAL_FRAC", 0.70)
    PATH_SUPPORT_RADIUS = env_float("TRACE_SUPPORT_RADIUS", 0.25)
    PATH_SCOUT_BATCH = env_int("TRACE_SCOUT_BATCH", 2)
    PATH_PRETRUST_KAPPA = env_float("TRACE_PRETRUST_KAPPA", 2.25)
    PATH_MID_KAPPA = env_float("TRACE_MID_KAPPA", 1.9)
    PATH_EXPLOIT_KAPPA = env_float("TRACE_EXPLOIT_KAPPA", 0.5)
    RESTART_ON_STAGNATION = env_bool("TRACE_RESTART_ON_STAGNATION", false)
    RESTART_FAILURE_WINDOW = env_int("TRACE_RESTART_FAILURE_WINDOW", 14)
    RESTART_MIN_ACTIVE_ITER = env_int("TRACE_RESTART_MIN_ACTIVE_ITER", 20)
    RESTART_MIN_DELTA = env_float("TRACE_RESTART_MIN_DELTA", 0.02)
    RESTART_DISABLE_LCB = env_float("TRACE_RESTART_DISABLE_LCB", 0.85)
    RESTART_MAX_RESTARTS = env_int("TRACE_RESTART_MAX_RESTARTS", 3)
    COUNT_RESTART_INIT_IN_ITER_BUDGET = env_bool("TRACE_COUNT_RESTART_INIT_IN_ITER_BUDGET", false)
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
    SLICE_N = env_int("TRACE_SLICE_N", 61)
    RUN_SLICES = env_bool("TRACE_RUN_SLICES", true)
    GP_DIAG = env_bool("TRACE_GP_DIAG", false)
    GP_DIAG_GRID_N = env_int("TRACE_GP_DIAG_GRID_N", 101)
    GP_DIAG_SNAPSHOT_ITERS = filter(<=(N_ITER), env_int_list("TRACE_GP_SNAPSHOT_ITERS", "10,50,100"))
    GP_DIAG_SNAPSHOT_AXES = env_axis_list("TRACE_GP_SNAPSHOT_AXES", "fcl,fsb,amp")
    GP_DIAG_EVERY_ITER_AXES = env_axis_list("TRACE_GP_EVERY_ITER_AXES", "fcl")
    STREAM_SEED_OUTPUTS = env_bool("TRACE_STREAM_SEED_OUTPUTS", false)

    mkpath(TRACE_DIR)
    mkpath(SLICE_DIR)
    mkpath(SCRIPT_DATA_DIR)
    GP_DIAG && mkpath(GP_DIAG_DIR)

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
    path_guard_config = CalibrationCode.PathGuardConfig(
        init_design=PATH_INIT_DESIGN,
        init_center_exclusion_radius=PATH_INIT_CENTER_EXCLUSION,
        init_sobol_skip_max=PATH_INIT_SOBOL_SKIP_MAX,
        decision_q_max=DECISION_Q_MAX,
        global_scout_period=PATH_GLOBAL_SCOUT_PERIOD,
        global_scout_frac=PATH_GLOBAL_SCOUT_FRAC,
        stagnation_window=PATH_STAGNATION_WINDOW,
        boundary_margin=PATH_BOUNDARY_MARGIN,
        ell_hi=PATH_ELL_HI,
        sigmaf_lo=PATH_SIGMAF_LO,
        c_hi=PATH_C_HI,
        trust_start_q=PATH_TRUST_START_Q,
        strong_trust_q=PATH_STRONG_TRUST_Q,
        trust_radius=PATH_TRUST_RADIUS,
        trust_radius_min=PATH_TRUST_RADIUS_MIN,
        mid_local_frac=PATH_MID_LOCAL_FRAC,
        strong_local_frac=PATH_STRONG_LOCAL_FRAC,
        support_radius=PATH_SUPPORT_RADIUS,
        scout_batch=PATH_SCOUT_BATCH,
        pretrust_kappa=PATH_PRETRUST_KAPPA,
        mid_kappa=PATH_MID_KAPPA,
        exploit_kappa=PATH_EXPLOIT_KAPPA,
    )

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
    @everywhere const _TRACE_THRESHOLD = $THRESHOLD
    @everywhere const _TRACE_DISABLE_STOPPING = $DISABLE_STOPPING
    @everywhere const _TRACE_CONFIDENCE_Z = $CONFIDENCE_Z
    @everywhere const _TRACE_REQUIRE_SURROGATE_CONFIDENCE = $REQUIRE_SURROGATE_CONFIDENCE
    @everywhere const _TRACE_STOPPING_RULE = $STOPPING_RULE
    @everywhere const _TRACE_SCORE_MODE = $(QuoteNode(SCORE_MODE))
    @everywhere const _TRACE_PATH_GUARD = $PATH_GUARD
    @everywhere const _TRACE_PATH_CONFIG = $path_guard_config
    @everywhere const _TRACE_RESTART_ON_STAGNATION = $RESTART_ON_STAGNATION
    @everywhere const _TRACE_RESTART_FAILURE_WINDOW = $RESTART_FAILURE_WINDOW
    @everywhere const _TRACE_RESTART_MIN_ACTIVE_ITER = $RESTART_MIN_ACTIVE_ITER
    @everywhere const _TRACE_RESTART_MIN_DELTA = $RESTART_MIN_DELTA
    @everywhere const _TRACE_RESTART_DISABLE_LCB = $RESTART_DISABLE_LCB
    @everywhere const _TRACE_RESTART_MAX_RESTARTS = $RESTART_MAX_RESTARTS
    @everywhere const _TRACE_COUNT_RESTART_INIT_IN_ITER_BUDGET = $COUNT_RESTART_INIT_IN_ITER_BUDGET
    @everywhere const _TRACE_CENTER_JITTER_U_MAX = $CENTER_JITTER_U_MAX
    @everywhere const _TRACE_BOUND_SCALE = $BOUND_SCALE
    @everywhere const _TRACE_FIXED_INIT_SEED = $FIXED_INIT_SEED
    @everywhere const _TRACE_CENTER_JITTER_U = Ref([0.0, 0.0, 0.0])
    @everywhere const _TRACE_GP_DIAG = $GP_DIAG
    @everywhere const _TRACE_GP_DIAG_GRID_N = $GP_DIAG_GRID_N
    @everywhere const _TRACE_GP_SNAPSHOT_ITERS = $GP_DIAG_SNAPSHOT_ITERS
    @everywhere const _TRACE_GP_SNAPSHOT_AXES = $GP_DIAG_SNAPSHOT_AXES
    @everywhere const _TRACE_GP_EVERY_ITER_AXES = $GP_DIAG_EVERY_ITER_AXES

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

    @everywhere function trace_q_det(u)
        return trace_score_from_weights(trace_varms_weights(u), Inf)
    end

    @everywhere function trace_gp_diag_axes(iter::Int)
        axes = Symbol[]
        if _TRACE_GP_DIAG
            if iter in _TRACE_GP_SNAPSHOT_ITERS
                append!(axes, _TRACE_GP_SNAPSHOT_AXES)
            end
            append!(axes, _TRACE_GP_EVERY_ITER_AXES)
        end
        out = Symbol[]
        for axis in axes
            axis in out || push!(out, axis)
        end
        return out
    end

    @everywhere function trace_axis_point(axis::Symbol, x::Float64)
        axis === :fcl && return [x, 0.0, 0.0]
        axis === :fsb && return [0.0, x, 0.0]
        axis === :amp && return [0.0, 0.0, x]
        throw(ArgumentError("Unsupported GP diagnostic axis $axis"))
    end

    @everywhere function trace_axis_physical(axis::Symbol, x::Float64)
        if axis === :fcl
            return _TRACE_SPAN_FCL * x / 1e3, "kHz"
        elseif axis === :fsb
            return _TRACE_SPAN_FSB * x / 1e3, "kHz"
        elseif axis === :amp
            return _TRACE_SPAN_A * x / _TRACE_A0, "DeltaA_over_Aopt"
        end
        throw(ArgumentError("Unsupported GP diagnostic axis $axis"))
    end

    @everywhere function trace_collect_gp_diag_rows(gp, seed::Int, iter::Int,
                                                    restart_id::Int,
                                                    x_rec::Vector{Float64})
        axes = trace_gp_diag_axes(iter)
        isempty(axes) && return NamedTuple[]
        rows = NamedTuple[]
        grid = range(-_TRACE_BOUND_SCALE, _TRACE_BOUND_SCALE; length=_TRACE_GP_DIAG_GRID_N)
        for axis in axes
            for x in grid
                u = trace_axis_point(axis, Float64(x))
                μ, s2 = CalibrationCode.predict_latent(gp, u)
                x_physical, unit = trace_axis_physical(axis, Float64(x))
                push!(rows, (
                    seed=seed,
                    N_shots=_TRACE_N_SHOTS,
                    iter=iter,
                    axis=String(axis),
                    x_u=Float64(x),
                    x_physical=x_physical,
                    physical_unit=unit,
                    mu=μ,
                    sigma=sqrt(max(s2, 0.0)),
                    q_det=trace_q_det(u),
                    x_rec_u1=x_rec[1],
                    x_rec_u2=x_rec[2],
                    x_rec_u3=x_rec[3],
                    restart_id=restart_id,
                ))
            end
        end
        return rows
    end

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
        return (_TRACE_PATH_GUARD || _TRACE_PATH_CONFIG.init_design != :random) ?
            CalibrationCode._path_guard_initial_points(rng, lb, ub, _TRACE_N_INIT, _TRACE_PATH_CONFIG) :
            [trace_rand_in_box(rng, lb, ub) for _ in 1:_TRACE_N_INIT]
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

    @everywhere function trace_ucb_score(μ::Float64, s2::Float64, κ::Float64)
        return μ + κ * sqrt(max(s2, 0.0))
    end

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
        d = length(bounds)
        n_cap = _TRACE_N_INIT + 3 * _TRACE_N_ITER + 4
        X = Matrix{Float64}(undef, d, n_cap)
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
        gp_diag_rows = NamedTuple[]
        final_x = copy(X[:, 1])
        final_y_rec = 0.0
        final_y_last = 0.0
        n_iter_actual = _TRACE_N_ITER
        stopped_early = false
        ell = [NaN, NaN, NaN]
        sigma_f = NaN
        c = NaN
        scout_remaining = 0
        last_supported_q = -Inf
        stagnation_count = 0
        boundary_streak = 0
        restart_count = 0
        restart_seeds = Int[]
        restart_start_iter = 1
        restart_failure_count = 0
        restart_best_signal = -Inf
        restart_init_iter_cost = 0

        for it in 1:_TRACE_N_ITER
            budget_it = it + restart_init_iter_cost
            budget_it > _TRACE_N_ITER && break
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
            ell = gp.ℓ
            sigma_f = gp.σf
            c = gp.c

            supported_x = Vector{Float64}()
            supported_q = -Inf
            supported_count = 0
            health_bad = false
            if _TRACE_PATH_GUARD
                supported_x, supported_q, supported_count =
                    CalibrationCode._best_supported_observation(X, y, write_idx, _TRACE_PATH_CONFIG)
                if supported_q > last_supported_q + 1e-6
                    last_supported_q = supported_q
                    stagnation_count = 0
                else
                    stagnation_count += 1
                end
                health_bad = CalibrationCode._gp_health_bad(gp, _TRACE_PATH_CONFIG)
                if health_bad ||
                   stagnation_count >= _TRACE_PATH_CONFIG.stagnation_window ||
                   boundary_streak >= 3
                    scout_remaining = max(scout_remaining, _TRACE_PATH_CONFIG.scout_batch)
                end
            end

            κ_eff = _TRACE_PATH_GUARD ?
                CalibrationCode._path_guard_kappa(supported_q, _TRACE_PATH_CONFIG) :
                _TRACE_KAPPA
            local_frac = _TRACE_PATH_GUARD ?
                CalibrationCode._path_guard_local_frac(supported_q, _TRACE_PATH_CONFIG) :
                0.0
            force_scout = _TRACE_PATH_GUARD &&
                (scout_remaining > 0 ||
                 (_TRACE_PATH_CONFIG.global_scout_period > 0 &&
                  it % _TRACE_PATH_CONFIG.global_scout_period == 0) ||
                 rand(rng) < _TRACE_PATH_CONFIG.global_scout_frac)

            x_acq = trace_rand_in_box(rng, lb, ub)
            best_a = -Inf
            m_acq = NaN
            s_acq = NaN
            acq_source = force_scout ? "scout" : "ucb"
            if force_scout
                x_acq = CalibrationCode._maximin_candidate(rng, lb, ub, X, write_idx; M=_TRACE_M_ACQ)
                μ, s2 = CalibrationCode.predict_latent(gp, x_acq)
                m_acq = μ
                s_acq = sqrt(max(s2, 0.0))
                μ_dec = _TRACE_PATH_GUARD ? CalibrationCode._decision_mean(μ, _TRACE_PATH_CONFIG) : μ
                best_a = trace_ucb_score(μ_dec, s2, κ_eff)
                scout_remaining = max(scout_remaining - 1, 0)
            else
                for _ in 1:_TRACE_M_ACQ
                    use_local = _TRACE_PATH_GUARD &&
                                supported_count >= _TRACE_PATH_CONFIG.min_support_count &&
                                rand(rng) < local_frac
                    x = use_local ?
                        CalibrationCode._rand_near_box(
                            rng, supported_x, lb, ub,
                            max(_TRACE_PATH_CONFIG.trust_radius_min,
                                _TRACE_PATH_CONFIG.trust_radius)) :
                        trace_rand_in_box(rng, lb, ub)
                    μ, s2 = CalibrationCode.predict_latent(gp, x)
                    μ_dec = _TRACE_PATH_GUARD ? CalibrationCode._decision_mean(μ, _TRACE_PATH_CONFIG) : μ
                    a = trace_ucb_score(μ_dec, s2, κ_eff)
                    if a > best_a
                        best_a = a
                        x_acq = x
                        m_acq = μ
                        s_acq = sqrt(max(s2, 0.0))
                    end
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
            final_y_last = y_acq
            y_acq_second = NaN

            if !_TRACE_DISABLE_STOPPING &&
               _TRACE_STOPPING_RULE == "two_noisy" &&
               y_acq >= _TRACE_THRESHOLD
                y2, _ = trace_q_fun(x_acq, _TRACE_N_SHOTS)
                total_shots += _TRACE_N_SHOTS
                y_acq_second = y2
                if y2 >= _TRACE_THRESHOLD
                    stopped_early = true
                    n_iter_actual = budget_it
                    final_x = x_acq
                    final_y_rec = 0.5 * (y_acq + y2)
                end
            end

            rec_center = _TRACE_PATH_GUARD && supported_count >= _TRACE_PATH_CONFIG.min_support_count ?
                supported_x : nothing
            x_rec, m_rec_raw, s_rec = _TRACE_PATH_GUARD ?
                CalibrationCode._recommend_mean_path_guard(
                    gp, bounds, _TRACE_PATH_CONFIG;
                    M=_TRACE_M_REC, rng=rng, center=rec_center,
                    local_frac=local_frac) :
                CalibrationCode.recommend_mean(gp, bounds; M=_TRACE_M_REC, rng=rng)
            if _TRACE_GP_DIAG
                append!(gp_diag_rows, trace_collect_gp_diag_rows(
                    gp, seed, budget_it, restart_count, x_rec))
            end
            # The GP posterior mean is mathematically unbounded. Keep the raw
            # value for diagnostics, but use physical Q bounds for stopping and
            # the default plotting column.
            m_rec = _TRACE_PATH_GUARD ?
                CalibrationCode._decision_mean(m_rec_raw, _TRACE_PATH_CONFIG) :
                clamp(m_rec_raw, 0.0, 1.0)
            m_rec_lcb = m_rec - _TRACE_CONFIDENCE_Z * s_rec
            surrogate_confident = m_rec_lcb >= _TRACE_THRESHOLD
            take_rec_check = !_TRACE_DISABLE_STOPPING &&
                             _TRACE_STOPPING_RULE == "surrogate_then_noisy" &&
                             surrogate_confident
            y_rec_check = NaN
            y_rec_second = NaN
            stop_supported = true
            if take_rec_check
                y_rec_check, σ_rec_check = trace_q_fun(x_rec, _TRACE_N_SHOTS)
                total_shots += _TRACE_N_SHOTS
                if trace_far_enough(x_rec, X, write_idx)
                    write_idx += 1
                    X[:, write_idx] = x_rec
                    y[write_idx] = y_rec_check
                    σy[write_idx] = σ_rec_check
                end

                if _TRACE_PATH_GUARD
                    support_n, support_avg = CalibrationCode._local_support_stats(
                        X, y, write_idx, x_rec, _TRACE_PATH_CONFIG.support_radius)
                    stop_supported = support_n >= _TRACE_PATH_CONFIG.min_support_count &&
                                     support_avg >= _TRACE_PATH_CONFIG.trust_start_q &&
                                     !health_bad
                    boundary_streak = CalibrationCode._near_boundary(
                        x_rec, lb, ub, _TRACE_PATH_CONFIG.boundary_margin) ?
                        boundary_streak + 1 : 0
                end

                if y_rec_check >= _TRACE_THRESHOLD && stop_supported
                    stopped_early = true
                    n_iter_actual = budget_it
                    final_x = x_rec
                    final_y_rec = y_rec_check
                end
            end

            q_det_rec = trace_q_det(x_rec)
            q_det_acq = trace_q_det(x_acq)
            restart_signal = m_rec_lcb
            restart_triggered = false
            restart_seed = 0
            active_iter = it - restart_start_iter + 1
            if _TRACE_RESTART_ON_STAGNATION && !stopped_early &&
               restart_count < _TRACE_RESTART_MAX_RESTARTS &&
               active_iter >= _TRACE_RESTART_MIN_ACTIVE_ITER
                if restart_signal >= _TRACE_RESTART_DISABLE_LCB
                    restart_failure_count = 0
                elseif restart_signal > restart_best_signal + _TRACE_RESTART_MIN_DELTA
                    restart_best_signal = restart_signal
                    restart_failure_count = 0
                else
                    restart_failure_count += 1
                    if restart_failure_count >= _TRACE_RESTART_FAILURE_WINDOW
                        restart_count += 1
                        restart_triggered = true
                        restart_seed = rand(rng, 1:typemax(Int32))
                    end
                end
            end
            push!(trace_rows, (
                iter=budget_it,
                total_shots=total_shots,
                n_train=write_idx,
                m_rec=m_rec,
                m_rec_raw=m_rec_raw,
                s_rec=s_rec,
                m_rec_lcb=m_rec_lcb,
                surrogate_confident=surrogate_confident,
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
                y_acq_second=y_acq_second,
                y_rec_check=y_rec_check,
                y_rec_second=y_rec_second,
                q_det_rec=q_det_rec,
                q_det_acq=q_det_acq,
                acq_source=acq_source,
                kappa_eff=κ_eff,
                supported_q=supported_q,
                supported_count=supported_count,
                health_bad=health_bad,
                stagnation_count=stagnation_count,
                restart_id=restart_count,
                restart_failure_count=restart_failure_count,
                restart_signal=restart_signal,
                restart_best_signal=restart_best_signal,
                restart_triggered=restart_triggered,
                restart_seed=restart_seed,
                boundary_streak=boundary_streak,
                scout_remaining=scout_remaining,
                stop_supported=stop_supported,
                center_jitter_u1=jitter_u[1],
                center_jitter_u2=jitter_u[2],
                center_jitter_u3=jitter_u[3],
                true_opt_u1=-jitter_u[1],
                true_opt_u2=-jitter_u[2],
                true_opt_u3=-jitter_u[3],
                stopped_early=stopped_early,
            ))

            if stopped_early
                return (
                    seed=seed,
                    trace_rows=trace_rows,
                    gp_diag_rows=gp_diag_rows,
                    x_rec=final_x,
                    y_rec=final_y_rec,
                    y_last=final_y_last,
                    n_iter_actual=n_iter_actual,
                    stopped_early=true,
                    total_shots=total_shots,
                    n_train=write_idx,
                    q_det=trace_q_det(final_x),
                    jitter_u=jitter_u,
                    init_seed=init_seed,
                    ell=ell,
                    sigma_f=sigma_f,
                    c=c,
                    restarts=restart_count,
                    restart_seeds=copy(restart_seeds),
                )
            end

            if restart_triggered
                push!(restart_seeds, restart_seed)
                Random.seed!(restart_seed)
                rng = MersenneTwister(restart_seed)
                write_idx, restart_shots =
                    trace_fill_initial_design!(X, y, σy, rng, lb, ub)
                total_shots += restart_shots
                if _TRACE_COUNT_RESTART_INIT_IN_ITER_BUDGET
                    restart_init_iter_cost += _TRACE_N_INIT
                end
                θ_prev = nothing
                scout_remaining = 0
                last_supported_q = -Inf
                stagnation_count = 0
                boundary_streak = 0
                restart_start_iter = it + 1
                restart_failure_count = 0
                restart_best_signal = -Inf
            end
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
        if _TRACE_PATH_GUARD
            supported_x, supported_q, supported_count =
                CalibrationCode._best_supported_observation(X, y, write_idx, _TRACE_PATH_CONFIG)
            rec_center = supported_count >= _TRACE_PATH_CONFIG.min_support_count ? supported_x : nothing
            final_x, _, _ = CalibrationCode._recommend_mean_path_guard(
                gp, bounds, _TRACE_PATH_CONFIG;
                M=_TRACE_M_REC, rng=rng, center=rec_center,
                local_frac=CalibrationCode._path_guard_local_frac(supported_q, _TRACE_PATH_CONFIG))
        else
            final_x, _, _ = CalibrationCode.recommend_mean(gp, bounds; M=_TRACE_M_REC, rng=rng)
        end
        final_y_rec, _ = trace_q_fun(final_x, _TRACE_N_SHOTS)
        total_shots += _TRACE_N_SHOTS
        return (
            seed=seed,
            trace_rows=trace_rows,
            gp_diag_rows=gp_diag_rows,
            x_rec=final_x,
            y_rec=final_y_rec,
            y_last=final_y_last,
            n_iter_actual=_TRACE_N_ITER,
            stopped_early=false,
            total_shots=total_shots,
            n_train=write_idx,
            q_det=trace_q_det(final_x),
            jitter_u=jitter_u,
            init_seed=init_seed,
            ell=gp.ℓ,
            sigma_f=gp.σf,
            c=gp.c,
            restarts=restart_count,
            restart_seeds=copy(restart_seeds),
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
    println("N=$N_SHOTS, seeds=$NUM_SIMS, workers=$(nworkers()), n_init=$N_INIT, n_iter=$N_ITER")
    println("threshold=$THRESHOLD, kappa=$KAPPA, M_acq=$M_ACQ, M_rec=$M_REC")
    println("disable_stopping=$DISABLE_STOPPING")
    println("surrogate_confidence_required=$REQUIRE_SURROGATE_CONFIDENCE, confidence_z=$CONFIDENCE_Z")
    println("stopping_rule=$STOPPING_RULE")
    println("score_mode=$SCORE_MODE")
    println("restart_on_stagnation=$RESTART_ON_STAGNATION, restart_failure_window=$RESTART_FAILURE_WINDOW, restart_disable_lcb=$RESTART_DISABLE_LCB")
    seed_label = join(seeds, ",")
    println("path_guard=$PATH_GUARD, init_design=$PATH_INIT_DESIGN, decision_q_max=$DECISION_Q_MAX, bound_scale=$BOUND_SCALE, freq_span_mode=$freq_span_mode, freq_span_raw_kHz=$freq_span_raw_kHz, fixed_init_seed=$FIXED_INIT_SEED, seeds=$seed_label")

    trace_header = [
        "iter", "total_shots", "n_train", "m_rec", "m_rec_raw", "s_rec",
        "m_rec_lcb", "surrogate_confident",
        "x_rec_u1", "x_rec_u2", "x_rec_u3",
        "x_acq_u1", "x_acq_u2", "x_acq_u3",
        "m_acq", "s_acq", "acq_score", "y_acq", "y_acq_second",
        "y_rec_check", "y_rec_second", "q_det_rec", "q_det_acq",
        "acq_source", "kappa_eff", "supported_q", "supported_count",
        "health_bad", "stagnation_count", "restart_id",
        "restart_failure_count", "restart_signal", "restart_best_signal",
        "restart_triggered", "restart_seed", "boundary_streak",
        "scout_remaining", "stop_supported",
        "center_jitter_u1", "center_jitter_u2", "center_jitter_u3",
        "true_opt_u1", "true_opt_u2", "true_opt_u3",
        "stopped_early",
    ]
    gp_diag_header = [
        "seed", "N_shots", "iter", "axis", "x_u", "x_physical", "physical_unit",
        "mu", "sigma", "q_det",
        "x_rec_u1", "x_rec_u2", "x_rec_u3", "restart_id",
    ]
    seed_progress_header = [
        "seed", "initial_seed", "N_shots", "iterations", "stopped_early",
        "restarts", "restart_seeds", "total_shots", "n_train", "q_det",
        "q_rec", "q_last", "center_jitter_u1", "center_jitter_u2",
        "center_jitter_u3", "true_opt_u1", "true_opt_u2", "true_opt_u3",
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
    @everywhere const _TRACE_GP_DIAG_DIR = $GP_DIAG_DIR
    @everywhere const _TRACE_PROGRESS_CSV = $seed_progress_path
    @everywhere const _TRACE_TRACE_HEADER = $trace_header
    @everywhere const _TRACE_GP_DIAG_HEADER = $gp_diag_header
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
    @everywhere function trace_write_seed_gpdiag_csv(result)
        !_TRACE_GP_DIAG && return nothing
        mkpath(_TRACE_GP_DIAG_DIR)
        path = joinpath(_TRACE_GP_DIAG_DIR, "gpdiag_seed$(result.seed).csv")
        open(path, "w") do io
            println(io, join(_TRACE_GP_DIAG_HEADER, ","))
            for row in result.gp_diag_rows
                println(io, join((trace_csv_field(getproperty(row, Symbol(name))) for name in _TRACE_GP_DIAG_HEADER), ","))
            end
        end
        return nothing
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
                    try
                        rm(lockdir; recursive=true, force=true)
                    catch
                    end
                end
                sleep(0.05)
            end
        end
        try
            return f()
        finally
            try
                rm(lockdir; recursive=true, force=true)
            catch
            end
        end
    end
    @everywhere function trace_append_seed_progress_csv(result, seed_elapsed::Float64)
        row = [
            result.seed, result.init_seed, _TRACE_N_SHOTS, result.n_iter_actual,
            result.stopped_early, result.restarts, join(result.restart_seeds, ";"),
            result.total_shots, result.n_train, result.q_det, result.y_rec, result.y_last,
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
        return nothing
    end
    @everywhere function trace_write_seed_outputs(result, seed_elapsed::Float64)
        !_TRACE_STREAM_SEED_OUTPUTS && return nothing
        trace_write_seed_trace_csv(result)
        trace_write_seed_gpdiag_csv(result)
        trace_append_seed_progress_csv(result, seed_elapsed)
        return nothing
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
        println("Finished seed $seed: Q_det=$(round(result.q_det, digits=5)), iter=$(result.n_iter_actual), stopped=$(result.stopped_early), restarts=$(result.restarts), elapsed_s=$(round(seed_elapsed, digits=2))")
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

    if GP_DIAG && !STREAM_SEED_OUTPUTS
        for result in results
            path = joinpath(GP_DIAG_DIR, "gpdiag_seed$(result.seed).csv")
            open(path, "w") do io
                println(io, join(gp_diag_header, ","))
                for row in result.gp_diag_rows
                    println(io, join((csv_field(getproperty(row, Symbol(name))) for name in gp_diag_header), ","))
                end
            end
        end
    end

    if RUN_SLICES
        @everywhere _TRACE_CENTER_JITTER_U[] = [0.0, 0.0, 0.0]
        grid = collect(range(-1.0, 1.0; length=SLICE_N))
        slice_specs = [
            ("slice_u1u2.csv", "u1", "u2", (x, y) -> [x, y, 0.0]),
            ("slice_u1u3.csv", "u1", "u3", (x, y) -> [x, 0.0, y]),
            ("slice_u2u3.csv", "u2", "u3", (x, y) -> [0.0, x, y]),
        ]
        for (filename, x_col, y_col, builder) in slice_specs
            jobs = [(x=x, y=y) for y in grid for x in grid]
            qvals = pmap(jobs; batch_size=max(1, length(jobs) ÷ (10 * max(nworkers(), 1)))) do job
                trace_q_det(builder(job.x, job.y))
            end
            open(joinpath(SLICE_DIR, filename), "w") do io
                println(io, "$x_col,$y_col,q")
                for (job, q) in zip(jobs, qvals)
                    println(io, join(csv_field.([job.x, job.y, q]), ","))
                end
            end
            println("Wrote $(joinpath(SLICE_DIR, filename))")
        end
    end

    elapsed = time() - start_time
    q_values = [r.q_det for r in results]
    iter_values = [r.n_iter_actual for r in results]
    shot_values = [r.total_shots for r in results]
    restart_values = [r.restarts for r in results]
    seed_elapsed_values = [r.seed_elapsed for r in results]
    early = count(r -> r.stopped_early, results)
    output_stem = isempty(OUTPUT_TAG) ? "benchmark_trace_N$(N_SHOTS)" : "benchmark_trace_N$(N_SHOTS)_$(OUTPUT_TAG)"
    output_path = joinpath(SCRIPT_DATA_DIR, "$(output_stem).txt")
    open(output_path, "w") do io
        println(io, "=== TRACE BENCHMARK RESULTS ===")
        println(io, "N_shots = $N_SHOTS")
        println(io, "n_init = $N_INIT")
        println(io, "n_iter = $N_ITER")
        println(io, "num_sims = $NUM_SIMS")
        println(io, "workers = $(nworkers())")
        println(io, "threshold = $THRESHOLD")
        println(io, "disable_stopping = $DISABLE_STOPPING")
        println(io, "require_surrogate_confidence = $REQUIRE_SURROGATE_CONFIDENCE")
        println(io, "stopping_rule = $STOPPING_RULE")
        println(io, "score_mode = $SCORE_MODE")
        println(io, "confidence_z = $CONFIDENCE_Z")
        println(io, "restart_on_stagnation = $RESTART_ON_STAGNATION")
        println(io, "restart_failure_window = $RESTART_FAILURE_WINDOW")
        println(io, "restart_min_active_iter = $RESTART_MIN_ACTIVE_ITER")
        println(io, "restart_min_delta = $RESTART_MIN_DELTA")
        println(io, "restart_disable_lcb = $RESTART_DISABLE_LCB")
        println(io, "restart_max_restarts = $RESTART_MAX_RESTARTS")
        println(io, "restart_seed_policy = reset_local_bo_rng_and_worker_global_rng")
        println(io, "kappa = $KAPPA")
        println(io, "path_guard = $PATH_GUARD")
        println(io, "path_guard_init_design = $PATH_INIT_DESIGN")
        println(io, "path_guard_init_center_exclusion = $PATH_INIT_CENTER_EXCLUSION")
        println(io, "path_guard_init_sobol_skip_max = $PATH_INIT_SOBOL_SKIP_MAX")
        println(io, "decision_q_max = $DECISION_Q_MAX")
        println(io, "path_guard_trust_start_q = $PATH_TRUST_START_Q")
        println(io, "path_guard_strong_trust_q = $PATH_STRONG_TRUST_Q")
        println(io, "path_guard_global_scout_period = $PATH_GLOBAL_SCOUT_PERIOD")
        println(io, "path_guard_global_scout_frac = $PATH_GLOBAL_SCOUT_FRAC")
        println(io, "center_jitter_u_max = $CENTER_JITTER_U_MAX")
        println(io, "bound_scale = $BOUND_SCALE")
        println(io, "freq_span_mode = $freq_span_mode")
        println(io, "freq_span_raw_kHz = $freq_span_raw_kHz")
        println(io, "freq_span_2pi_kHz = $freq_span_2pi_kHz")
        println(io, "fixed_init_seed = $FIXED_INIT_SEED")
        println(io, "gp_diag = $GP_DIAG")
        println(io, "gp_diag_dir = $GP_DIAG_DIR")
        println(io, "gp_diag_grid_n = $GP_DIAG_GRID_N")
        println(io, "gp_diag_snapshot_iters = $(join(GP_DIAG_SNAPSHOT_ITERS, ","))")
        println(io, "gp_diag_snapshot_axes = $(join(string.(GP_DIAG_SNAPSHOT_AXES), ","))")
        println(io, "gp_diag_every_iter_axes = $(join(string.(GP_DIAG_EVERY_ITER_AXES), ","))")
        println(io, "stream_seed_outputs = $STREAM_SEED_OUTPUTS")
        println(io, "seed_progress_csv = $seed_progress_path")
        println(io, "M_acq = $M_ACQ")
        println(io, "M_rec = $M_REC")
        println(io, "elapsed_seconds = $elapsed")
        println(io, "")
        println(io, "=== RESULTS SUMMARY ===")
        println(io, "Early stops = $early / $NUM_SIMS")
        println(io, "Average Q_det = $(mean(q_values)) +/- $(sample_std(q_values))")
        println(io, "Median Q_det = $(median(q_values))")
        println(io, "Min Q_det = $(minimum(q_values))")
        println(io, "Max Q_det = $(maximum(q_values))")
        println(io, "Average log10(1-Q_det) = $(mean(log_infid.(q_values))) +/- $(sample_std(log_infid.(q_values)))")
        println(io, "Average iterations = $(mean(iter_values)) +/- $(sample_std(iter_values))")
        println(io, "Average total shots = $(mean(shot_values))")
        println(io, "Total restarts = $(sum(restart_values))")
        println(io, "Average restarts = $(mean(restart_values)) +/- $(sample_std(restart_values))")
        println(io, "Average seed elapsed seconds = $(mean(seed_elapsed_values)) +/- $(sample_std(seed_elapsed_values))")
        println(io, "")
        println(io, "=== All Results ===")
        println(io, "Sim\tSeed\tInitialSeed\tIterations\tStoppedEarly\tRestarts\tRestartSeeds\tTotalShots\tNTrain\tQ_det\tQ_rec\tQ_last\tcenter_jitter_u1\tcenter_jitter_u2\tcenter_jitter_u3\ttrue_opt_u1\ttrue_opt_u2\ttrue_opt_u3\tell1\tell2\tell3\tsigma_f\tc\tElapsedSeconds")
        for (idx, r) in enumerate(results)
            println(io, join(csv_field.([
                idx, r.seed, r.init_seed, r.n_iter_actual, r.stopped_early, r.restarts,
                join(r.restart_seeds, ";"), r.total_shots,
                r.n_train, r.q_det, r.y_rec, r.y_last,
                r.jitter_u[1], r.jitter_u[2], r.jitter_u[3],
                -r.jitter_u[1], -r.jitter_u[2], -r.jitter_u[3],
                r.ell[1], r.ell[2], r.ell[3], r.sigma_f, r.c, r.seed_elapsed,
            ]), "\t"))
        end
    end

    println("Trace files written to: $TRACE_DIR")
    println("Slice files written to: $SLICE_DIR")
    println("Summary written to: $output_path")
finally
    rmprocs(workers())
end
