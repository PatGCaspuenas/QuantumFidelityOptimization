import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

using .CalibrationCode
using Random
using Statistics
using Printf

# ============================================================
# CONFIG
# ============================================================
const t        = 100.0
const N_values = [50, 100, 250, 500, 1000]   # shot counts to test
const n_points = 20      # number of spatial sample points
const log_fid  = false    # match pretrain_gp_3d.jl setting
const rng      = MersenneTwister(42)

# same spans as pretrain_gp_3d.jl
base      = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A
span_kHz  = 2.0
span_fcl  = span_kHz * 1e3 * 2π
span_fsb  = span_kHz * 1e3 * 2π
span_A    = 1.2 * A0 - A0

u_to_params(u) = (f_cl0 + span_fcl*u[1],
                  f_sb0 + span_fsb*u[2],
                  A0    + span_A  *u[3])

apply_log(Q) = log_fid ? log10(max(1.0 - Q, 1e-15)) : Q

# ── predicted σy from each model ─────────────────────────────────────────────
sigma_simple(N)         = 1.0/sqrt(N)
sigma_correct(Q_raw, N) = begin
    one_minus_Q = max(1.0 - Q_raw, 1e-15)
    σ_Q = sqrt(max(Q_raw, 0.0) * one_minus_Q / N)
    log_fid ? σ_Q / (one_minus_Q * log(10)) : σ_Q
end

# ============================================================
# Sample random points in [-1,1]^3
# ============================================================
points = [rand(rng, 3) .* 2 .- 1 for _ in 1:n_points]
Q_det_vals = [CalibrationCode.Q_det(t, u_to_params(u)...) for u in points]

out_file = joinpath(@__DIR__, "..", "data", "compare_sigma_models.txt")

open(out_file, "w") do io
    results = [(; N=0, Q=0.0, Q_mean=0.0, σ_emp=0.0, σ_simple=0.0, σ_correct=0.0) for _ in 1:0]

    println(io, "=== Sigma model comparison — $(n_points) points, N single-shot reps each ===")
    println(io, "log_fid = $log_fid")
    @printf(io, "\n%-6s  %-8s  %-8s  %-12s  %-12s  %-12s  %-10s  %-10s\n",
        "N", "Q_det", "Q_mean", "σ_empirical", "σ_simple", "σ_correct", "err_simple", "err_correct")
    println(io, repeat("-", 84))

    for N in N_values
        for (u, Q_true) in zip(points, Q_det_vals)
            fcl, fsb, A = u_to_params(u)

            single_shots = [CalibrationCode.Q_varMS(t, fcl, fsb, A; N=1, numMS=2)
                            for _ in 1:N]
            σ_emp = std(single_shots) / sqrt(N)
            Q_measured = CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2)
            σ_s = sigma_simple(N)
            σ_c = sigma_correct(Q_measured, N)

            push!(results, (; N, Q=Q_true, Q_mean=Q_measured, σ_emp, σ_simple=σ_s, σ_correct=σ_c))

            @printf(io, "%-6d  %-8.4f  %-8.4f  %-12.5f  %-12.5f  %-12.5f  %-10.4f  %-10.4f\n",
                N, Q_true, Q_measured, σ_emp, σ_s, σ_c,
                abs(σ_s - σ_emp) / max(σ_emp, 1e-15),
                abs(σ_c - σ_emp) / max(σ_emp, 1e-15))
        end
        println(io)
    end

    rel_err_simple  = mean(abs(r.σ_simple  - r.σ_emp) / max(r.σ_emp, 1e-15) for r in results)
    rel_err_correct = mean(abs(r.σ_correct - r.σ_emp) / max(r.σ_emp, 1e-15) for r in results)
    @printf(io, "\nMean relative error — simple: %.4f    correct: %.4f\n",
        rel_err_simple, rel_err_correct)

    # Per-N breakdown
    println(io, "\n=== Per-N summary ===")
    @printf(io, "%-6s  %-14s  %-14s\n", "N", "err_simple", "err_correct")
    println(io, repeat("-", 38))
    for N in N_values
        sub = filter(r -> r.N == N, results)
        e_s = mean(abs(r.σ_simple  - r.σ_emp) / max(r.σ_emp, 1e-15) for r in sub)
        e_c = mean(abs(r.σ_correct - r.σ_emp) / max(r.σ_emp, 1e-15) for r in sub)
        @printf(io, "%-6d  %-14.4f  %-14.4f\n", N, e_s, e_c)
    end

    # Per-point breakdown (ratio σ_simple/σ_empirical and σ_correct/σ_empirical)
    println(io, "\n=== Ratio predicted/empirical per point (N=400) ===")
    @printf(io, "%-8s  %-10s  %-10s\n", "Q_det", "simple/emp", "correct/emp")
    println(io, repeat("-", 32))
    sub400 = filter(r -> r.N == (400 in N_values ? 400 : N_values[end]), results)
    for r in sub400
        @printf(io, "%-8.4f  %-10.4f  %-10.4f\n",
            r.Q,
            r.σ_simple  / max(r.σ_emp, 1e-15),
            r.σ_correct / max(r.σ_emp, 1e-15))
    end
end

println("Written → $out_file")
