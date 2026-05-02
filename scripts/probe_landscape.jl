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

@everywhere const _t     = $t
@everywhere const _f_cl0 = $f_cl0
@everywhere const _f_sb0 = $f_sb0
@everywhere const _A0    = $A0
@everywhere const _span_fcl = $span_fcl
@everywhere const _span_fsb = $span_fsb
@everywhere const _span_A   = $span_A

@everywhere function eval_point(u1, u2, u3; N::Int=50_000)
    fcl = _f_cl0 + _span_fcl * u1
    fsb = _f_sb0 + _span_fsb * u2
    A   = _A0    + _span_A   * u3
    Q_bal, σ_bal = CalibrationCode.Q_varMS_balance_σ(_t, fcl, fsb, A; N=N, numMS=3)
    Q_det        = clamp(CalibrationCode.Q_det(_t, fcl, fsb, A), 0.0, 1.0)
    return (u1=u1, u2=u2, u3=u3, Q_bal=Q_bal, σ_bal=σ_bal, Q_det=Q_det)
end

N_probe = 50_000

# -----------------------------------------------------------------------
# Region 1: false peak — fix u2≈-0.8, u3≈+1.0, scan u1; then scan u2/u3
# -----------------------------------------------------------------------
false_peak_points = [
    # (u1, u2, u3)
    # scan u1 at the false peak core
    (u1, -0.80, 1.00) for u1 in -0.20:0.05:0.20
]
append!(false_peak_points, [
    # scan u2 around -0.8 at fixed u1=0, u3=1
    (0.0, u2, 1.00) for u2 in -1.00:0.05:-0.50
])
append!(false_peak_points, [
    # scan u3 around 1.0 at fixed u1=0, u2=-0.8
    (0.0, -0.80, u3) for u3 in 0.50:0.05:1.00
])

# -----------------------------------------------------------------------
# Region 2: true peak — scan around (0, 0, -0.3) in all three dims
# -----------------------------------------------------------------------
true_peak_points = [
    (u1, 0.00, -0.30) for u1 in -0.10:0.02:0.10
]
append!(true_peak_points, [
    (0.00, u2, -0.30) for u2 in -0.10:0.02:0.10
])
append!(true_peak_points, [
    (0.00, 0.00, u3)  for u3 in -0.50:0.05:0.10
])
# also sprinkle the exact candidate from best UCB seed
push!(true_peak_points, (-0.023, -0.024, -0.300))

all_points = vcat(false_peak_points, true_peak_points)
n_false    = length(false_peak_points)

println("Probing $(length(all_points)) points with N=$N_probe shots each …")
flush(stdout)

results = pmap(all_points; batch_size=1) do (u1, u2, u3)
    eval_point(u1, u2, u3; N=N_probe)
end

false_results = results[1:n_false]
true_results  = results[n_false+1:end]

header = @sprintf("%-8s  %-8s  %-8s  %-10s  %-8s  %-8s",
                  "u_1", "u_2", "u_3", "Q_balance", "σ_bal", "Q_det")
sep    = "-"^60

println()
println("=== FALSE PEAK REGION (u2≈-0.8, u3≈+1.0) ===")
println(header); println(sep)
for r in false_results
    @printf("%-8.4f  %-8.4f  %-8.4f  %-10.4f  %-8.4f  %-8.4f\n",
            r.u1, r.u2, r.u3, r.Q_bal, r.σ_bal, r.Q_det)
end
max_false = maximum(r.Q_bal for r in false_results)
@printf("\nMax Q_balance in false region : %.6f\n", max_false)
@printf("Q_det at false-peak max      : %.6f\n",
        false_results[argmax([r.Q_bal for r in false_results])].Q_det)

println()
println("=== TRUE PEAK REGION (u1≈0, u2≈0, u3≈-0.3) ===")
println(header); println(sep)
for r in true_results
    @printf("%-8.4f  %-8.4f  %-8.4f  %-10.4f  %-8.4f  %-8.4f\n",
            r.u1, r.u2, r.u3, r.Q_bal, r.σ_bal, r.Q_det)
end
max_true = maximum(r.Q_bal for r in true_results)
@printf("\nMax Q_balance in true  region : %.6f\n", max_true)
@printf("Q_det at true-peak max       : %.6f\n",
        true_results[argmax([r.Q_bal for r in true_results])].Q_det)

println()
println("=== COMPARISON ===")
@printf("False peak max Q_balance : %.6f\n", max_false)
@printf("True  peak max Q_balance : %.6f\n", max_true)
@printf("Difference (true-false)  : %+.6f\n", max_true - max_false)
if max_true > max_false
    println("→ True peak IS higher in Q_balance: EI should find it given enough budget/exploration.")
else
    println("→ False peak is equally or more optimal in Q_balance: objective is fundamentally degenerate.")
end

rmprocs(workers())
