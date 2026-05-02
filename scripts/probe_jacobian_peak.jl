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
    BLAS.set_num_threads(1)
end

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

# Deterministic eval: no shots, direct from exact populations.
# Q_jac matches the BO objective: 1 - |expected_gg - P_gg| - |expected_ee - P_ee|
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

# ── Grid 1: coarse wide scan ─────────────────────────────────────────────────
# Covers the spread seen across 40 BO simulations (u3 has ±0.15 std in benchmark)
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

# ── Grid 2: fine zoom around the coarse peak ─────────────────────────────────
zoom_r   = 0.03   # ±0.03 around coarse peak in each dimension
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

# ── Line scans through fine peak (1D slices for landscape shape) ──────────────
n_line = 200
u1_scan = LinRange(-0.07,  0.07, n_line)
u2_scan = LinRange(-0.07,  0.07, n_line)
u3_scan = LinRange(-0.40,  0.40, n_line)

line_u1 = pmap(u1_scan; batch_size=4) do u1; eval_point(u1,     best_f.u2, best_f.u3); end
line_u2 = pmap(u2_scan; batch_size=4) do u2; eval_point(best_f.u1, u2,     best_f.u3); end
line_u3 = pmap(u3_scan; batch_size=4) do u3; eval_point(best_f.u1, best_f.u2, u3    ); end

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

# Half-width in each dimension (where Q_jac > 0.99 * max)
thresh_hw = 0.99 * best_f.Q_jac
hw_u1 = [r.u1 for r in line_u1 if r.Q_jac >= thresh_hw]
hw_u2 = [r.u2 for r in line_u2 if r.Q_jac >= thresh_hw]
hw_u3 = [r.u3 for r in line_u3 if r.Q_jac >= thresh_hw]
isempty(hw_u1) || @printf("Peak width (Q≥99%%max) u1: [%+.4f, %+.4f]  Δu1=%.4f\n",
                           minimum(hw_u1), maximum(hw_u1), maximum(hw_u1)-minimum(hw_u1))
isempty(hw_u2) || @printf("Peak width (Q≥99%%max) u2: [%+.4f, %+.4f]  Δu2=%.4f\n",
                           minimum(hw_u2), maximum(hw_u2), maximum(hw_u2)-minimum(hw_u2))
isempty(hw_u3) || @printf("Peak width (Q≥99%%max) u3: [%+.4f, %+.4f]  Δu3=%.4f\n",
                           minimum(hw_u3), maximum(hw_u3), maximum(hw_u3)-minimum(hw_u3))

# ── Save CSV ──────────────────────────────────────────────────────────────────
outdir  = joinpath(@__DIR__, "data")
mkpath(outdir)

# Coarse grid CSV
open(joinpath(outdir, "jacobian_probe_coarse.csv"), "w") do io
    println(io, "u1,u2,u3,Q_jac,Q_det,gg,ee")
    for r in coarse
        @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u1, r.u2, r.u3, r.Q_jac, r.Q_det, r.gg, r.ee)
    end
end

# Line scan CSV
open(joinpath(outdir, "jacobian_probe_lines.csv"), "w") do io
    println(io, "axis,u_scan,u1,u2,u3,Q_jac,Q_det")
    for r in line_u1
        @printf(io, "u1,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u1, r.u1, r.u2, r.u3, r.Q_jac, r.Q_det)
    end
    for r in line_u2
        @printf(io, "u2,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u2, r.u1, r.u2, r.u3, r.Q_jac, r.Q_det)
    end
    for r in line_u3
        @printf(io, "u3,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u3, r.u1, r.u2, r.u3, r.Q_jac, r.Q_det)
    end
end

println()
println("Saved:")
println("  scripts/data/jacobian_probe_coarse.csv  ($(length(coarse)) points)")
println("  scripts/data/jacobian_probe_lines.csv   (3 × $n_line line scans)")

rmprocs(workers())
