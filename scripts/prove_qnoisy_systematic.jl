import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random, Statistics, Printf

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

# ── Reference point ───────────────────────────────────────────────────────────
t    = 100.0
base = CalibrationCode.ideal(t)
f_cl, f_sb, A = base.f_cl, base.f_sb, base.A

Q_det_ref = CalibrationCode.Q_det(t, f_cl, f_sb, A)
@printf("Q_det at reference point: %.8f\n\n", Q_det_ref)

# ── Sweep N ───────────────────────────────────────────────────────────────────
# Q_noisy ODE solve happens once per call regardless of N,
# so N=100000 is cheap (just more samples from the same density matrix).
N_vals = [400, 1_000, 5_000, 20_000, 100_000]
n_reps = 60

println("N_shots   | Q_noisy_fixed mean ± std  | Q_noisy_amp mean ± std   | gap_fixed   gap_amp")
println("-"^95)

results = []

for N in N_vals
    fixed_vals = Float64[]
    amp_vals   = Float64[]
    for rep in 1:n_reps
        Random.seed!(rep)
        qf = CalibrationCode.Q_noisy(t, f_cl, f_sb, A; N=N, use_amplitude=false)
        qa = CalibrationCode.Q_noisy(t, f_cl, f_sb, A; N=N, use_amplitude=true)
        push!(fixed_vals, qf)
        push!(amp_vals,   qa)
    end
    mf, sf = mean(fixed_vals), std(fixed_vals)
    ma, sa = mean(amp_vals),   std(amp_vals)
    gap_f  = mf - Q_det_ref
    gap_a  = ma - Q_det_ref
    @printf("N=%7d | %.6f ± %.6f          | %.6f ± %.6f          | %+.6f   %+.6f\n",
            N, mf, sf, ma, sa, gap_f, gap_a)
    push!(results, (N=N, mean_fixed=mf, std_fixed=sf, mean_amp=ma, std_amp=sa,
                    gap_fixed=gap_f, gap_amp=gap_a))
end

println()
@printf("Q_det (reference, exact):  %.8f\n", Q_det_ref)
println()
println("Interpretation:")
println("  gap_fixed  → systematic offset of Q_noisy (fixed phase) vs Q_det at N→∞")
println("  gap_amp    → should shrink toward 0 as N increases (amplitude is unbiased)")

# ── CSV ───────────────────────────────────────────────────────────────────────
outpath = joinpath(@__DIR__, "data", "qnoisy_systematic.csv")
open(outpath, "w") do io
    println(io, "N,Q_det,mean_fixed,std_fixed,mean_amp,std_amp,gap_fixed,gap_amp")
    for r in results
        @printf(io, "%d,%.8f,%.8f,%.8f,%.8f,%.8f,%+.8f,%+.8f\n",
                r.N, Q_det_ref, r.mean_fixed, r.std_fixed,
                r.mean_amp, r.std_amp, r.gap_fixed, r.gap_amp)
    end
end
println("Saved: $outpath")
