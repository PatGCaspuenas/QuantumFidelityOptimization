import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Statistics
using Printf

if nprocs() == 1
    addprocs(max(1, Sys.CPU_THREADS - 1))
end

@everywhere begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
    include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
    using LinearAlgebra
    using StatsBase
    using Statistics
    BLAS.set_num_threads(1)
end

# Probe settings (overridable via ENV)
const N_SHOTS_PROBE = parse(Int, get(ENV, "PROBE_N_SHOTS",   "10000"))
const N_REPEATS     = parse(Int, get(ENV, "PROBE_N_REPEATS", "100"))

t = 100.0
base  = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

span_kHz = 2.0
span_fcl = span_kHz * 1e3 * 2π
span_fsb = span_kHz * 1e3 * 2π
span_A   = 1.2 * A0 - A0

# Load jacobian subgate sequence (same path as opt_benchmark.jl)
_jac_path = joinpath(@__DIR__, "..", "data", "ms_sequence_search_result.jl")
isfile(_jac_path) || error("Jacobian probe requires data/ms_sequence_search_result.jl")
_jac_raw       = include(_jac_path)
_jac_subgates  = [CalibrationCode.MSSubgate(sg.theta, sg.phi)
                  for sg in _jac_raw.best_overall.subgates]
_I_center      = Float64(_jac_raw.I_center)

# Reference populations at nominal parameters
_ref_pulses   = CalibrationCode.build_closed_loop_ms_sequence(
                    t, f_cl0, f_sb0, _I_center, _jac_subgates)
_ref_pops     = CalibrationCode.populations_ms_sequence(_ref_pulses)
_expected_gg  = Float64(_ref_pops.gg)
_expected_ee  = Float64(_ref_pops.ee)
println("Reference pops: gg=$(round(_expected_gg, digits=5))  ee=$(round(_expected_ee, digits=5))")
println("N_shots_probe=$N_SHOTS_PROBE  n_repeats=$N_REPEATS")

@everywhere const _t           = $t
@everywhere const _f_cl0       = $f_cl0
@everywhere const _f_sb0       = $f_sb0
@everywhere const _A0          = $A0
@everywhere const _span_fcl    = $span_fcl
@everywhere const _span_fsb    = $span_fsb
@everywhere const _span_A      = $span_A
@everywhere const _I_center_w  = $_I_center
@everywhere const _subgates    = $_jac_subgates
@everywhere const _exp_gg      = $_expected_gg
@everywhere const _exp_ee      = $_expected_ee

# ── Deterministic eval (no shots) ────────────────────────────────────────────
@everywhere function eval_point(u1, u2, u3)
    fcl = _f_cl0 + _span_fcl * u1
    fsb = _f_sb0 + _span_fsb * u2
    A   = _A0    + _span_A   * u3
    pulses = CalibrationCode.build_closed_loop_ms_sequence(
                 _t, fcl, fsb, A, _subgates)
    pops   = CalibrationCode.populations_ms_sequence(pulses)
    Q_jac  = clamp(1.0 - abs(_exp_gg - pops.gg) - abs(_exp_ee - pops.ee), 0.0, 1.0)
    Q_det  = clamp(CalibrationCode.Q_det(_t, fcl, fsb, A), 0.0, 1.0)
    return (u1=u1, u2=u2, u3=u3, Q_jac=Q_jac, Q_det=Q_det, gg=Float64(pops.gg), ee=Float64(pops.ee))
end

# ── Noisy eval: deterministic + shot-noise Monte Carlo ───────────────────────
# For each point: run ODE once, sample N_shots n_repeats times.
# Returns: Q_jac_det, Q_det, mean/std of Q_shot and Q_debiased,
#          theoretical folded bias, and residual (empirical - theory).
@everywhere function eval_point_noisy(u1, u2, u3, N_shots::Int, n_repeats::Int)
    fcl = _f_cl0 + _span_fcl * u1
    fsb = _f_sb0 + _span_fsb * u2
    A   = _A0    + _span_A   * u3

    pulses = CalibrationCode.build_closed_loop_ms_sequence(_t, fcl, fsb, A, _subgates)
    pops   = CalibrationCode.populations_ms_sequence(pulses)
    p_gg   = Float64(pops.gg)
    p_ee   = Float64(pops.ee)

    Q_jac_det = clamp(1.0 - abs(_exp_gg - p_gg) - abs(_exp_ee - p_ee), 0.0, 1.0)
    Q_det_val = clamp(CalibrationCode.Q_det(_t, fcl, fsb, A), 0.0, 1.0)

    # Theoretical folded bias using exact populations
    d_gg_true  = abs(p_gg - _exp_gg)
    d_ee_true  = abs(p_ee - _exp_ee)
    σ_gg_true  = sqrt(max(p_gg * (1.0 - p_gg), 0.0) / N_shots)
    σ_ee_true  = sqrt(max(p_ee * (1.0 - p_ee), 0.0) / N_shots)
    bias_theory = CalibrationCode._folded_bias(d_gg_true, σ_gg_true) +
                  CalibrationCode._folded_bias(d_ee_true, σ_ee_true)

    # Monte Carlo: sample N_shots from exact population probabilities n_repeats times.
    # Only the multinomial sampling differs each rep — no ODE re-evaluation.
    weights = Float64[max(p_gg, 0.0), max(p_ee, 0.0),
                      max(Float64(pops.eg), 0.0), max(Float64(pops.ge), 0.0)]
    q_shot_vals = Vector{Float64}(undef, n_repeats)
    q_db_vals   = Vector{Float64}(undef, n_repeats)

    for rep in 1:n_repeats
        samples = StatsBase.sample(1:4, StatsBase.Weights(weights), N_shots)
        P_SS = count(==(1), samples) / N_shots
        P_DD = count(==(2), samples) / N_shots

        d_gg_obs = abs(P_SS - _exp_gg)
        d_ee_obs = abs(P_DD - _exp_ee)
        σ_gg_obs = sqrt(max(P_SS * (1.0 - P_SS), 0.0) / N_shots)
        σ_ee_obs = sqrt(max(P_DD * (1.0 - P_DD), 0.0) / N_shots)

        q_shot = clamp(1.0 - d_gg_obs - d_ee_obs, 0.0, 1.0)
        q_db   = clamp(q_shot +
                       CalibrationCode._folded_bias(d_gg_obs, σ_gg_obs) +
                       CalibrationCode._folded_bias(d_ee_obs, σ_ee_obs), 0.0, 1.0)

        q_shot_vals[rep] = q_shot
        q_db_vals[rep]   = q_db
    end

    q_shot_mean = mean(q_shot_vals)
    q_db_mean   = mean(q_db_vals)

    return (
        u1           = u1,
        u2           = u2,
        u3           = u3,
        Q_jac_det    = Q_jac_det,
        Q_det        = Q_det_val,
        Q_shot_mean  = q_shot_mean,
        Q_shot_std   = std(q_shot_vals),
        Q_db_mean    = q_db_mean,
        Q_db_std     = std(q_db_vals),
        bias_empirical = Q_jac_det - q_shot_mean,     # how much shots underestimate det
        bias_theory    = bias_theory,                  # E[|X|]+E[|Y|] analytical formula
        bias_residual  = (Q_jac_det - q_shot_mean) - bias_theory,  # should be ≈0
        db_error       = Q_jac_det - q_db_mean,        # should be ≈0 if debias is good
        gg             = p_gg,
        ee             = p_ee,
    )
end

# ── Grid 1: coarse wide scan (deterministic only) ────────────────────────────
u1_coarse = LinRange(-0.07,  0.07, 29)
u2_coarse = LinRange(-0.07,  0.07, 29)
u3_coarse = LinRange(-0.40,  0.40, 41)
coarse_pts = [(u1, u2, u3) for u1 in u1_coarse, u2 in u2_coarse, u3 in u3_coarse][:]
println("Coarse grid: $(length(coarse_pts)) points … evaluating in parallel")
flush(stdout)

coarse = pmap(coarse_pts; batch_size=4) do (u1, u2, u3)
    eval_point(u1, u2, u3)
end

best_c  = coarse[argmax([r.Q_jac for r in coarse])]
@printf("Coarse max  Q_jac = %.6f  Q_det = %.6f  at (u1=%.4f, u2=%.4f, u3=%.4f)\n",
        best_c.Q_jac, best_c.Q_det, best_c.u1, best_c.u2, best_c.u3)
flush(stdout)

# ── Grid 2: fine zoom around the coarse peak (deterministic only) ─────────────
zoom_r   = 0.03
n_zoom   = 30
u1_fine  = LinRange(best_c.u1 - zoom_r, best_c.u1 + zoom_r, n_zoom)
u2_fine  = LinRange(best_c.u2 - zoom_r, best_c.u2 + zoom_r, n_zoom)
u3_fine  = LinRange(best_c.u3 - zoom_r, best_c.u3 + zoom_r, n_zoom)
fine_pts = [(u1, u2, u3) for u1 in u1_fine, u2 in u2_fine, u3 in u3_fine][:]
println("Fine  grid: $(length(fine_pts)) points around coarse peak … evaluating")
flush(stdout)

fine = pmap(fine_pts; batch_size=4) do (u1, u2, u3)
    eval_point(u1, u2, u3)
end

best_f = fine[argmax([r.Q_jac for r in fine])]
@printf("Fine  max  Q_jac = %.6f  Q_det = %.6f  at (u1=%.5f, u2=%.5f, u3=%.5f)\n",
        best_f.Q_jac, best_f.Q_det, best_f.u1, best_f.u2, best_f.u3)
flush(stdout)

# ── Line scans: deterministic + noisy (N_shots_probe shots, N_REPEATS reps) ──
n_line = 200
u1_scan = LinRange(-0.07,  0.07, n_line)
u2_scan = LinRange(-0.07,  0.07, n_line)
u3_scan = LinRange(-0.40,  0.40, n_line)

println("Line scans: $n_line pts × 3 axes, N_shots=$N_SHOTS_PROBE, n_repeats=$N_REPEATS … evaluating")
flush(stdout)

_N = N_SHOTS_PROBE; _R = N_REPEATS
line_u1 = pmap(u1_scan; batch_size=2) do u1
    eval_point_noisy(u1, best_f.u2, best_f.u3, _N, _R)
end
line_u2 = pmap(u2_scan; batch_size=2) do u2
    eval_point_noisy(best_f.u1, u2, best_f.u3, _N, _R)
end
line_u3 = pmap(u3_scan; batch_size=2) do u3
    eval_point_noisy(best_f.u1, best_f.u2, u3, _N, _R)
end

# ── Summary ───────────────────────────────────────────────────────────────────
println()
println("=== JACOBIAN PEAK PROBE SUMMARY ===")
@printf("Coarse global max  Q_jac = %.6f  Q_det = %.6f\n", best_c.Q_jac, best_c.Q_det)
@printf("  location: u1=%+.5f  u2=%+.5f  u3=%+.5f\n", best_c.u1, best_c.u2, best_c.u3)
println()
@printf("Fine-zoom max      Q_jac = %.6f  Q_det = %.6f\n", best_f.Q_jac, best_f.Q_det)
@printf("  location: u1=%+.5f  u2=%+.5f  u3=%+.5f\n", best_f.u1, best_f.u2, best_f.u3)
println()

fcl_rec = f_cl0 + span_fcl * best_f.u1
fsb_rec = f_sb0 + span_fsb * best_f.u2
A_rec   = A0    + span_A   * best_f.u3
@printf("Physical: f_cl=%.6e  f_sb=%.6e  A=%.6e\n", fcl_rec, fsb_rec, A_rec)
println()

# Folded bias at the peak
peak_u1 = pmap([best_f.u1]) do u; eval_point_noisy(u, best_f.u2, best_f.u3, N_SHOTS_PROBE, 500); end[1]
@printf("=== FOLDED BIAS AT PEAK (n_repeats=500) ===\n")
@printf("  Q_jac_det        = %.6f\n", peak_u1.Q_jac_det)
@printf("  Q_shot_mean      = %.6f  (expected ≈ Q_jac_det − bias_theory)\n", peak_u1.Q_shot_mean)
@printf("  Q_debiased_mean  = %.6f  (expected ≈ Q_jac_det)\n", peak_u1.Q_db_mean)
@printf("  bias_empirical   = %.6f\n", peak_u1.bias_empirical)
@printf("  bias_theory      = %.6f\n", peak_u1.bias_theory)
@printf("  bias_residual    = %.6f  (empirical − theory, should be ≈0)\n", peak_u1.bias_residual)
@printf("  db_error         = %.6f  (Q_jac_det − Q_db_mean, should be ≈0)\n", peak_u1.db_error)
@printf("  Q_shot_std       = %.6f\n", peak_u1.Q_shot_std)
@printf("  Q_debiased_std   = %.6f\n", peak_u1.Q_db_std)
println()

# Half-width stats (still using deterministic Q_jac_det from noisy eval)
thresh_hw = 0.99 * best_f.Q_jac
hw_u1 = [r.u1 for r in line_u1 if r.Q_jac_det >= thresh_hw]
hw_u2 = [r.u2 for r in line_u2 if r.Q_jac_det >= thresh_hw]
hw_u3 = [r.u3 for r in line_u3 if r.Q_jac_det >= thresh_hw]
isempty(hw_u1) || @printf("Peak width (Q≥99%%max) u1: [%+.4f, %+.4f]  Δu1=%.4f\n",
                           minimum(hw_u1), maximum(hw_u1), maximum(hw_u1)-minimum(hw_u1))
isempty(hw_u2) || @printf("Peak width (Q≥99%%max) u2: [%+.4f, %+.4f]  Δu2=%.4f\n",
                           minimum(hw_u2), maximum(hw_u2), maximum(hw_u2)-minimum(hw_u2))
isempty(hw_u3) || @printf("Peak width (Q≥99%%max) u3: [%+.4f, %+.4f]  Δu3=%.4f\n",
                           minimum(hw_u3), maximum(hw_u3), maximum(hw_u3)-minimum(hw_u3))

# ── Save CSV ──────────────────────────────────────────────────────────────────
outdir  = joinpath(@__DIR__, "data")
mkpath(outdir)

# Coarse grid CSV (deterministic only, unchanged)
open(joinpath(outdir, "jacobian_probe_coarse.csv"), "w") do io
    println(io, "u1,u2,u3,Q_jac,Q_det,gg,ee")
    for r in coarse
        @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u1, r.u2, r.u3, r.Q_jac, r.Q_det, r.gg, r.ee)
    end
end

# Line scan CSV: deterministic + noisy (proves folded bias)
# Columns:
#   Q_jac_det    — deterministic Q_jac (no shots): the "true" objective value
#   Q_shot_mean  — empirical mean of Q_jac with N_shots shots (averaged over n_repeats)
#   Q_shot_std   — empirical std of Q_jac_shot
#   Q_db_mean    — empirical mean of debiased Q_jac (should ≈ Q_jac_det)
#   Q_db_std     — empirical std of debiased Q_jac
#   bias_empirical — Q_jac_det - Q_shot_mean (observed downshift from shot noise)
#   bias_theory    — analytical folded-normal E[|X|]+E[|Y|] prediction
#   bias_residual  — empirical - theory (should be ≈0)
#   db_error       — Q_jac_det - Q_db_mean (should be ≈0)
open(joinpath(outdir, "jacobian_probe_lines.csv"), "w") do io
    println(io, "axis,u_scan,u1,u2,u3,Q_jac_det,Q_det,Q_shot_mean,Q_shot_std,Q_db_mean,Q_db_std,bias_empirical,bias_theory,bias_residual,db_error,gg,ee")
    for r in line_u1
        @printf(io, "u1,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u1, r.u1, r.u2, r.u3,
                r.Q_jac_det, r.Q_det,
                r.Q_shot_mean, r.Q_shot_std,
                r.Q_db_mean, r.Q_db_std,
                r.bias_empirical, r.bias_theory, r.bias_residual, r.db_error,
                r.gg, r.ee)
    end
    for r in line_u2
        @printf(io, "u2,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u2, r.u1, r.u2, r.u3,
                r.Q_jac_det, r.Q_det,
                r.Q_shot_mean, r.Q_shot_std,
                r.Q_db_mean, r.Q_db_std,
                r.bias_empirical, r.bias_theory, r.bias_residual, r.db_error,
                r.gg, r.ee)
    end
    for r in line_u3
        @printf(io, "u3,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u3, r.u1, r.u2, r.u3,
                r.Q_jac_det, r.Q_det,
                r.Q_shot_mean, r.Q_shot_std,
                r.Q_db_mean, r.Q_db_std,
                r.bias_empirical, r.bias_theory, r.bias_residual, r.db_error,
                r.gg, r.ee)
    end
end

println()
println("Saved:")
println("  scripts/data/jacobian_probe_coarse.csv  ($(length(coarse)) points, deterministic)")
println("  scripts/data/jacobian_probe_lines.csv   (3 × $n_line pts, det + noisy N=$N_SHOTS_PROBE × $N_REPEATS reps)")

rmprocs(workers())
