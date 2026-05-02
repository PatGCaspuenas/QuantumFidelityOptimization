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
base = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

span_kHz = 2.0
span_fcl = span_kHz * 1e3 * 2π
span_fsb = span_kHz * 1e3 * 2π
span_A   = 1.2 * A0 - A0

@everywhere const _t        = $t
@everywhere const _f_cl0    = $f_cl0
@everywhere const _f_sb0    = $f_sb0
@everywhere const _A0       = $A0
@everywhere const _span_fcl = $span_fcl
@everywhere const _span_fsb = $span_fsb
@everywhere const _span_A   = $span_A

@everywhere function eval_point(u1, u2, u3; N::Int=20_000)
    fcl = _f_cl0 + _span_fcl * u1
    fsb = _f_sb0 + _span_fsb * u2
    A   = _A0    + _span_A   * u3
    Q_bal, σ = CalibrationCode.Q_varMS_balance_σ(_t, fcl, fsb, A; N=N, numMS=3)
    Q_det    = clamp(CalibrationCode.Q_det(_t, fcl, fsb, A), 0.0, 1.0)
    return (u1=u1, u2=u2, u3=u3, Q_bal=Q_bal, σ=σ, Q_det=Q_det)
end

N_probe = 20_000

# ── FALSE PEAK basin: fix u1=0, scan u2 × u3 finely ─────────────────────────
fp_u2_vals = collect(range(-0.84, -0.76; step=0.01))   #  9 values
fp_u3_vals = collect(range( 0.68,  0.82; step=0.01))   # 15 values
fp_points  = [(0.0, u2, u3) for u2 in fp_u2_vals, u3 in fp_u3_vals][:]

# ── TRUE PEAK basin: fix u1=0, scan u2 × u3 finely ──────────────────────────
# asymmetric window: u2 from the scan looked very different on + vs - side
tp_u2_vals = collect(range(-0.04, 0.02; step=0.005))   # 13 values
tp_u3_vals = collect(range(-0.02, 0.10; step=0.01))    # 13 values
tp_points  = [(0.0, u2, u3) for u2 in tp_u2_vals, u3 in tp_u3_vals][:]

all_points = vcat(fp_points, tp_points)
n_fp = length(fp_points)

println("False-peak grid : $(length(fp_points)) points  ($(length(fp_u2_vals))×$(length(fp_u3_vals)))")
println("True-peak  grid : $(length(tp_points)) points  ($(length(tp_u2_vals))×$(length(tp_u3_vals)))")
println("Total: $(length(all_points)) points, N=$N_probe shots each")
flush(stdout)

results = pmap(all_points; batch_size=1) do (u1, u2, u3)
    eval_point(u1, u2, u3; N=N_probe)
end

fp_results = results[1:n_fp]
tp_results = results[n_fp+1:end]

# ── write CSV ─────────────────────────────────────────────────────────────────
output_dir = joinpath(@__DIR__, "data")
mkpath(output_dir)
csv_file = joinpath(output_dir, "basin_probe.csv")

open(csv_file, "w") do io
    println(io, "basin,u1,u2,u3,Q_bal,sigma,Q_det")
    for r in fp_results
        @printf(io, "false,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u1, r.u2, r.u3, r.Q_bal, r.σ, r.Q_det)
    end
    for r in tp_results
        @printf(io, "true,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.u1, r.u2, r.u3, r.Q_bal, r.σ, r.Q_det)
    end
end
println("CSV written → $csv_file")

# ── summary ───────────────────────────────────────────────────────────────────
fp_best_idx = argmax([r.Q_bal for r in fp_results])
tp_best_idx = argmax([r.Q_bal for r in tp_results])
fp_best = fp_results[fp_best_idx]
tp_best = tp_results[tp_best_idx]

println("\n=== FALSE PEAK basin best ===")
@printf("u=(%+.4f, %+.4f, %+.4f)  Q_balance=%.6f  Q_det=%.6f\n",
        fp_best.u1, fp_best.u2, fp_best.u3, fp_best.Q_bal, fp_best.Q_det)

println("\n=== TRUE PEAK basin best ===")
@printf("u=(%+.4f, %+.4f, %+.4f)  Q_balance=%.6f  Q_det=%.6f\n",
        tp_best.u1, tp_best.u2, tp_best.u3, tp_best.Q_bal, tp_best.Q_det)

println("\n=== COMPARISON ===")
@printf("False peak max Q_balance : %.6f  (Q_det=%.4f)\n", fp_best.Q_bal, fp_best.Q_det)
@printf("True  peak max Q_balance : %.6f  (Q_det=%.4f)\n", tp_best.Q_bal, tp_best.Q_det)
@printf("Difference (true-false)  : %+.6f\n", tp_best.Q_bal - fp_best.Q_bal)

rmprocs(workers())
