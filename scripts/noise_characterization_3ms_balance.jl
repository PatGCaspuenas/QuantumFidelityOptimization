import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using .CalibrationCode
using Random
using Statistics
using Printf
using StatsBase
using Base.Threads

# ============================================================
# CONFIG
# ============================================================
const t          = 100.0
const N_values   = [50, 400, 1000]
const M_reps     = 500
const n_near     = 25
const n_far      = 25
const rng        = MersenneTwister(42)
const PHASE_GRID = 0.0:0.1:π

base            = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A
span_fcl        = 2.0 * 1e3 * 2π
span_fsb        = 2.0 * 1e3 * 2π
span_A          = 1.2 * A0 - A0

# 3D parameter space only — no phase shift
u_to_params(u) = (f_cl0 + span_fcl * u[1],
                  f_sb0 + span_fsb * u[2],
                  A0    + span_A   * u[3])

# ── σ models ──────────────────────────────────────────────────────────────────
sigma_simple(N)      = 1.0 / sqrt(N)
sigma_binomial(p, N) = sqrt(max(p, 0.0) * max(1.0 - p, 0.0) / N)

# Delta method for Q = 1 - |r1-P1| - |r2-P2| with multinomial (P1,P2).
# Cov(P1,P2) = -p1·p2/N. Degrades at the kink (pi == ri).
function sigma_delta(p1, p2, r1, r2, N)
    var_Q = (p1*(1-p1) + p2*(1-p2) - 2*sign(p1-r1)*sign(p2-r2)*p1*p2) / N
    return sqrt(max(var_Q, 0.0))
end

# ============================================================
# Sample points: 25 near origin + 25 over full space
# ============================================================
pts_near      = [0.3 .* (2 .* rand(rng, 3) .- 1) for _ in 1:n_near]
pts_far       = [2 .* rand(rng, 3) .- 1           for _ in 1:n_far]
all_points    = vcat(pts_near, pts_far)
region_labels = vcat(fill("near", n_near), fill("far", n_far))
n_points      = length(all_points)

# ============================================================
# Q_ms_sequence ideal-point reference (fixed for all spatial points)
# ============================================================
c_subgates  = CalibrationCode.sequence_C_subgates()
ideal_probs = CalibrationCode.Q_ms_sequence_probs(t, f_cl0, f_sb0, A0, c_subgates)
exp_gg      = ideal_probs[1]
exp_ee      = ideal_probs[2]
@printf("Q_ms_sequence reference: exp_gg=%.5f  exp_ee=%.5f\n", exp_gg, exp_ee)

# ============================================================
# Pre-compute per-point quantum states (parallelized, once each).
# Each function is the deterministic ODE part of the corresponding
# stochastic model — sampling is done separately in the M_reps loops.
# ============================================================
Q_det_vals    = Vector{Float64}(undef, n_points)
varMS_probs   = Vector{NTuple{4,Float64}}(undef, n_points)   # numMS=2
balance_probs = Vector{NTuple{4,Float64}}(undef, n_points)   # numMS=3
ms_seq_probs  = Vector{NTuple{4,Float64}}(undef, n_points)   # sequence_C
noisy_states  = Vector{Any}(undef, n_points)                 # Q_noisy precomp

Threads.@threads for i in 1:n_points
    u = all_points[i]
    fcl, fsb, A = u_to_params(u)
    Q_det_vals[i]    = CalibrationCode.Q_det(t, fcl, fsb, A)
    varMS_probs[i]   = CalibrationCode.Q_varMS_balance_probs(t, fcl, fsb, A; numMS=2)
    balance_probs[i] = CalibrationCode.Q_varMS_balance_probs(t, fcl, fsb, A; numMS=3)
    ms_seq_probs[i]  = CalibrationCode.Q_ms_sequence_probs(t, fcl, fsb, A, c_subgates)
    noisy_states[i]  = CalibrationCode.Q_noisy_precompute(t, fcl, fsb, A;
                           phase_grid=PHASE_GRID)
end

# True Q for each model in the N→∞ limit (from precomputed probs)
varMS_true   = [varMS_probs[i][2] for i in 1:n_points]   # p_DD (DD-only success, numMS=2)
balance_true = [clamp(1.0 - (abs(0.5    - balance_probs[i][1]) +
                               abs(0.5    - balance_probs[i][2])), 0.0, 1.0)
                for i in 1:n_points]
mseq_true    = [clamp(1.0 - (abs(exp_gg - ms_seq_probs[i][1])  +
                               abs(exp_ee - ms_seq_probs[i][2])),  0.0, 1.0)
                for i in 1:n_points]
noisy_true   = Q_det_vals   # Q_noisy estimates Bell fidelity

# ============================================================
# Helper: per-N breakdown
# ============================================================
function per_n_summary(io, results, sigma_keys)
    @printf(io, "  Per-N: %-6s", "N")
    for k in sigma_keys; @printf(io, "  %-14s", "err_$k"); end
    println(io)
    println(io, "  " * repeat("-", 8 + 16 * length(sigma_keys)))
    for N in N_values
        sub = filter(r -> r.N == N, results)
        @printf(io, "  %-6d", N)
        for k in sigma_keys
            e = mean(abs(r.sigmas[k] - r.σ_emp) / max(r.σ_emp, 1e-15) for r in sub)
            @printf(io, "  %-14.4f", e)
        end
        println(io)
    end
end

# ============================================================
# Output
# ============================================================
out_file = joinpath(@__DIR__, "data", "noise_characterization_all_models.txt")
mkpath(dirname(out_file))

open(out_file, "w") do io
    println(io, "=== Noise characterization — 4 estimator models (3D, no phase shift) ===")
    println(io, "Points: $n_points  ($n_near near |u|≤0.3, $n_far over [-1,1]³)")
    println(io, "M_reps=$M_reps independent N-shot experiments; σ_emp = std across reps")
    @printf(io, "Q_ms_sequence reference: exp_gg=%.5f  exp_ee=%.5f\n", exp_gg, exp_ee)

    summary = Dict{String, Dict{String, Float64}}()

    # ════════════════════════════════════════════════════════════════════════
    # Model 1: Q_varMS (numMS=2) — estimator k_DD/N (Bernoulli mean)
    # σ_binomial is exact; σ_delta reduces to σ_binomial for a single count
    # ════════════════════════════════════════════════════════════════════════
    println(io, "\n" * "="^90)
    println(io, "Model 1: Q_varMS (numMS=2)   estimator: k_DD / N   (Bernoulli — σ_binomial exact)")
    println(io)
    @printf(io, "%-6s  %-6s  %-8s  %-8s  %-12s  %-12s  %-12s  %-10s  %-10s\n",
        "N", "region", "Q_true", "Q_mean", "σ_emp", "σ_simple", "σ_binomial",
        "err_simple", "err_binom")
    println(io, repeat("-", 92))

    m1_all = NamedTuple[]
    for N in N_values
        local_res = Vector{NamedTuple}(undef, n_points)
        Threads.@threads for i in 1:n_points
            SS, DD, SD, DS = varMS_probs[i]
            wts    = Weights(Float64[SS, DD, SD, DS])
            Q_true = varMS_true[i]
            region = region_labels[i]
            q_reps = [count(==(2), sample(1:4, wts, N)) / N for _ in 1:M_reps]
            Q_mean = mean(q_reps)
            σ_emp  = std(q_reps; corrected=true)
            sigmas = Dict("simple"   => sigma_simple(N),
                          "binomial" => sigma_binomial(Q_true, N))
            local_res[i] = (; N, region, Q_true, Q_mean, σ_emp, sigmas)
        end
        for r in local_res
            push!(m1_all, r)
            @printf(io, "%-6d  %-6s  %-8.4f  %-8.4f  %-12.5f  %-12.5f  %-12.5f  %-10.4f  %-10.4f\n",
                r.N, r.region, r.Q_true, r.Q_mean, r.σ_emp,
                r.sigmas["simple"], r.sigmas["binomial"],
                abs(r.sigmas["simple"]   - r.σ_emp) / max(r.σ_emp, 1e-15),
                abs(r.sigmas["binomial"] - r.σ_emp) / max(r.σ_emp, 1e-15))
        end
        println(io)
    end
    per_n_summary(io, m1_all, ["simple", "binomial"])
    summary["Q_varMS(2ms)"] = Dict(
        k => mean(abs(r.sigmas[k] - r.σ_emp) / max(r.σ_emp, 1e-15) for r in m1_all)
        for k in ["simple", "binomial"])

    # ════════════════════════════════════════════════════════════════════════
    # Model 2: Q_varMS_balance (numMS=3)
    # estimator: 1 - |0.5-P_SS| - |0.5-P_DD|   (nonlinear in two counts)
    # ════════════════════════════════════════════════════════════════════════
    println(io, "\n" * "="^90)
    println(io, "Model 2: Q_varMS_balance (numMS=3)   estimator: 1 - |0.5-P_SS| - |0.5-P_DD|")
    println(io)
    @printf(io, "%-6s  %-6s  %-8s  %-8s  %-12s  %-12s  %-12s  %-12s  %-10s  %-10s  %-10s\n",
        "N", "region", "Q_true", "Q_mean", "σ_emp", "σ_simple", "σ_binomial", "σ_delta",
        "err_simple", "err_binom", "err_delta")
    println(io, repeat("-", 112))

    m2_all = NamedTuple[]
    for N in N_values
        local_res = Vector{NamedTuple}(undef, n_points)
        Threads.@threads for i in 1:n_points
            SS, DD, SD, DS = balance_probs[i]
            wts    = Weights(Float64[SS, DD, SD, DS])
            Q_true = balance_true[i]
            region = region_labels[i]
            q_reps = map(1:M_reps) do _
                bulk = sample(1:4, wts, N)
                P_SS = count(==(1), bulk) / N
                P_DD = count(==(2), bulk) / N
                clamp(1.0 - (abs(0.5 - P_SS) + abs(0.5 - P_DD)), 0.0, 1.0)
            end
            Q_mean = mean(q_reps)
            σ_emp  = std(q_reps; corrected=true)
            sigmas = Dict("simple"   => sigma_simple(N),
                          "binomial" => sigma_binomial(Q_true, N),
                          "delta"    => sigma_delta(SS, DD, 0.5, 0.5, N))
            local_res[i] = (; N, region, Q_true, Q_mean, σ_emp, sigmas)
        end
        for r in local_res
            push!(m2_all, r)
            @printf(io, "%-6d  %-6s  %-8.4f  %-8.4f  %-12.5f  %-12.5f  %-12.5f  %-12.5f  %-10.4f  %-10.4f  %-10.4f\n",
                r.N, r.region, r.Q_true, r.Q_mean, r.σ_emp,
                r.sigmas["simple"], r.sigmas["binomial"], r.sigmas["delta"],
                abs(r.sigmas["simple"]   - r.σ_emp) / max(r.σ_emp, 1e-15),
                abs(r.sigmas["binomial"] - r.σ_emp) / max(r.σ_emp, 1e-15),
                abs(r.sigmas["delta"]    - r.σ_emp) / max(r.σ_emp, 1e-15))
        end
        println(io)
    end
    per_n_summary(io, m2_all, ["simple", "binomial", "delta"])
    summary["Q_varMS_balance"] = Dict(
        k => mean(abs(r.sigmas[k] - r.σ_emp) / max(r.σ_emp, 1e-15) for r in m2_all)
        for k in ["simple", "binomial", "delta"])

    # ════════════════════════════════════════════════════════════════════════
    # Model 3: Q_ms_sequence (sequence_C, 3 subgates)
    # estimator: 1 - |exp_gg-P_gg| - |exp_ee-P_ee|  (same form, shifted refs)
    # ════════════════════════════════════════════════════════════════════════
    println(io, "\n" * "="^90)
    println(io, "Model 3: Q_ms_sequence (sequence_C, 3 subgates)")
    @printf(io, "         estimator: 1 - |%.4f-P_gg| - |%.4f-P_ee|\n", exp_gg, exp_ee)
    println(io)
    @printf(io, "%-6s  %-6s  %-8s  %-8s  %-12s  %-12s  %-12s  %-12s  %-10s  %-10s  %-10s\n",
        "N", "region", "Q_true", "Q_mean", "σ_emp", "σ_simple", "σ_binomial", "σ_delta",
        "err_simple", "err_binom", "err_delta")
    println(io, repeat("-", 112))

    m3_all = NamedTuple[]
    for N in N_values
        local_res = Vector{NamedTuple}(undef, n_points)
        Threads.@threads for i in 1:n_points
            p_gg, p_ee, p_eg, p_ge = ms_seq_probs[i]
            wts    = Weights(Float64[p_gg, p_ee, p_eg, p_ge])
            Q_true = mseq_true[i]
            region = region_labels[i]
            q_reps = map(1:M_reps) do _
                bulk = sample(1:4, wts, N)
                P_gg = count(==(1), bulk) / N
                P_ee = count(==(2), bulk) / N
                clamp(1.0 - (abs(exp_gg - P_gg) + abs(exp_ee - P_ee)), 0.0, 1.0)
            end
            Q_mean = mean(q_reps)
            σ_emp  = std(q_reps; corrected=true)
            sigmas = Dict("simple"   => sigma_simple(N),
                          "binomial" => sigma_binomial(Q_true, N),
                          "delta"    => sigma_delta(p_gg, p_ee, exp_gg, exp_ee, N))
            local_res[i] = (; N, region, Q_true, Q_mean, σ_emp, sigmas)
        end
        for r in local_res
            push!(m3_all, r)
            @printf(io, "%-6d  %-6s  %-8.4f  %-8.4f  %-12.5f  %-12.5f  %-12.5f  %-12.5f  %-10.4f  %-10.4f  %-10.4f\n",
                r.N, r.region, r.Q_true, r.Q_mean, r.σ_emp,
                r.sigmas["simple"], r.sigmas["binomial"], r.sigmas["delta"],
                abs(r.sigmas["simple"]   - r.σ_emp) / max(r.σ_emp, 1e-15),
                abs(r.sigmas["binomial"] - r.σ_emp) / max(r.σ_emp, 1e-15),
                abs(r.sigmas["delta"]    - r.σ_emp) / max(r.σ_emp, 1e-15))
        end
        println(io)
    end
    per_n_summary(io, m3_all, ["simple", "binomial", "delta"])
    summary["Q_ms_sequence"] = Dict(
        k => mean(abs(r.sigmas[k] - r.σ_emp) / max(r.σ_emp, 1e-15) for r in m3_all)
        for k in ["simple", "binomial", "delta"])

    # ════════════════════════════════════════════════════════════════════════
    # Model 4: Q_noisy (single-gate parity scan + cosine fit)
    # estimator: (1 - P_odd + |C|) / 2
    # σ_delta not applicable (estimator is output of a curve fit, not a count mean)
    # ════════════════════════════════════════════════════════════════════════
    println(io, "\n" * "="^90)
    println(io, "Model 4: Q_noisy (single-gate parity scan + cosine fit)")
    println(io, "         estimator: (1 - P_odd + |C|) / 2   (σ_delta not applicable)")
    println(io)
    @printf(io, "%-6s  %-6s  %-8s  %-8s  %-12s  %-12s  %-12s  %-10s  %-10s\n",
        "N", "region", "Q_true", "Q_mean", "σ_emp", "σ_simple", "σ_binomial",
        "err_simple", "err_binom")
    println(io, repeat("-", 92))

    m4_all = NamedTuple[]
    for N in N_values
        local_res = Vector{NamedTuple}(undef, n_points)
        Threads.@threads for i in 1:n_points
            precomp = noisy_states[i]
            Q_true  = noisy_true[i]
            region  = region_labels[i]
            q_reps  = [CalibrationCode.Q_noisy_rep(precomp; N=N) for _ in 1:M_reps]
            filter!(!isnan, q_reps)
            Q_mean = isempty(q_reps) ? NaN : mean(q_reps)
            σ_emp  = isempty(q_reps) ? NaN : std(q_reps; corrected=true)
            sigmas = Dict("simple"   => sigma_simple(N),
                          "binomial" => sigma_binomial(Q_true, N))
            local_res[i] = (; N, region, Q_true, Q_mean, σ_emp, sigmas)
        end
        for r in local_res
            push!(m4_all, r)
            @printf(io, "%-6d  %-6s  %-8.4f  %-8.4f  %-12.5f  %-12.5f  %-12.5f  %-10.4f  %-10.4f\n",
                r.N, r.region, r.Q_true, r.Q_mean, r.σ_emp,
                r.sigmas["simple"], r.sigmas["binomial"],
                abs(r.sigmas["simple"]   - r.σ_emp) / max(r.σ_emp, 1e-15),
                abs(r.sigmas["binomial"] - r.σ_emp) / max(r.σ_emp, 1e-15))
        end
        println(io)
    end
    valid_m4 = filter(r -> !isnan(r.σ_emp), m4_all)
    per_n_summary(io, valid_m4, ["simple", "binomial"])
    summary["Q_noisy"] = Dict(
        k => mean(abs(r.sigmas[k] - r.σ_emp) / max(r.σ_emp, 1e-15) for r in valid_m4)
        for k in ["simple", "binomial"])

    # ════════════════════════════════════════════════════════════════════════
    # Cross-model summary
    # ════════════════════════════════════════════════════════════════════════
    println(io, "\n" * "="^90)
    println(io, "Cross-model summary — mean relative error |σ_model - σ_emp| / σ_emp")
    println(io, "(lower = better; N/A = not applicable to that estimator)")
    println(io)
    @printf(io, "%-22s  %-14s  %-14s  %-14s\n",
        "Model", "err_simple", "err_binomial", "err_delta")
    println(io, repeat("-", 68))
    for (mname, has_delta) in [
            ("Q_varMS(2ms)",    false),
            ("Q_varMS_balance", true),
            ("Q_ms_sequence",   true),
            ("Q_noisy",         false)]
        errs = summary[mname]
        @printf(io, "%-22s  %-14.4f  %-14.4f  %-14s\n",
            mname,
            errs["simple"],
            errs["binomial"],
            has_delta ? @sprintf("%.4f", errs["delta"]) : "N/A")
    end
end

println("Written → $out_file")
