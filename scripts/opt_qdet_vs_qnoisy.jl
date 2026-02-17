# scripts/optimize_qdet_vs_qnoisy.jl
#
# Compare:
# 1) Standard BayesOpt (EI) on deterministic Q_det
# 2) Heteroscedastic BO (UCB + σ-threshold) on noisy Q_noisy
#    and evaluate its recommendation with Q_det
#
# Output:
#   compare_recommendations.png
#
# Run:
#   julia --project=. scripts/optimize_qdet_vs_qnoisy.jl

using Random
using Plots
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

redirect_stderr(devnull)

# -------------------------
# Config
# -------------------------
const t = 100.0

# work in scaled u-space: u ∈ [-1,1]^2 mapped to frequency neighborhood
const span_hz = 1e4  # ± span around baseline (set 2e4 for ±20 kHz)
const bounds_u = [(-1.0, 1.0), (-1.0, 1.0)]

# Map "noise knob" σ (algorithmic) -> number of shots N used in Q_noisy
# Rule: smaller σ => larger N. Tune these to match your simulator runtime.
function N_from_sigma(σ::Float64)
    σ ≤ 1e-4  && return 2000
    σ ≤ 5e-4  && return 1500
    σ ≤ 1e-3  && return 1000
    σ ≤ 5e-3  && return 500
    σ ≤ 1e-2  && return 200
    σ ≤ 2e-2  && return 150
    σ ≤ 5e-2  && return 80
    return 40
end

# -------------------------
# Baseline
# -------------------------
base = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

println("Baseline (t=$(t) μs):")
println(" f_cl0 = $(round(f_cl0, digits=3))")
println(" f_sb0 = $(round(f_sb0, digits=3))")
println(" A0    = $(round(A0,    digits=3))")
println(" fid0  = $(round(base.fid, digits=6))")

# u -> (f_cl, f_sb)
u_to_freq(u) = (f_cl0 + span_hz*u[1],  f_sb0 + span_hz*u[2])

# deterministic objective in u-space
function Qdet_u(u)
    fcl, fsb = u_to_freq(u)
    return CalibrationCode.Q_det(t, fcl, fsb, A0)
end

# noisy objective in u-space; signature required by bayesopt_ucb_threshold: f(u, σ)
function Qnoisy_u(u, σ)
    fcl, fsb = u_to_freq(u)
    return CalibrationCode.Q_noisy(t, fcl, fsb, A0; N=N_from_sigma(σ))
end

# -------------------------
# 1) Standard BayesOpt (EI) on Q_det
# -------------------------
Random.seed!(1)

res_det, u_rec_det, y_rec_det_gpmean = CalibrationCode.bayesopt(Qdet_u;
    bounds=bounds_u,
    n_init=20,
    n_iter=60,
    M=4000,
    xi=0.01,
    maximize=true,
    seed=1,
    obs_noise=1e-4
)

fcl_det, fsb_det = u_to_freq(u_rec_det)
y_det_true = CalibrationCode.Q_det(t, fcl_det, fsb_det, A0)

println("\n=== BayesOpt (EI) on Q_det ===")
println("u_rec = ", u_rec_det)
println("f_cl = ", fcl_det)
println("f_sb = ", fsb_det)
println("GP-mean y_rec ≈ ", y_rec_det_gpmean)
println("Q_det(f_rec)  = ", y_det_true)

# -------------------------
# 2) Heteroscedastic BO on Q_noisy, evaluated with Q_det
# -------------------------
σ_levels = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]

Random.seed!(2)

res_het = CalibrationCode.bayesopt_ucb_threshold(Qnoisy_u;
    bounds=bounds_u,
    σ_levels=σ_levels,
    n_init=20,
    n_iter=80,
    κ=2.0,
    α=0.5,
    seed=2
)

fcl_het, fsb_het = u_to_freq(res_het.x_rec)
y_het_true = CalibrationCode.Q_det(t, fcl_het, fsb_het, A0)

println("\n=== BayesHeteroOpt (UCB+threshold) on Q_noisy ===")
println("u_rec = ", res_het.x_rec)
println("f_cl = ", fcl_het)
println("f_sb = ", fsb_het)
println("GP-mean y_rec ≈ ", res_het.y_rec, " (model space)")
println("Q_det(f_rec)  = ", y_het_true, " (true deterministic)")

counts = CalibrationCode.count_noise_levels(res_het)
println("\nNoise usage:")
for σ in res_het.σ_levels
    println("σ=$(σ): ", counts[σ], "   (N=$(N_from_sigma(σ)))")
end

# -------------------------
# 3) Visual comparison on deterministic landscape (u-space)
# -------------------------
nx, ny = 41, 41
us1 = range(-1.0, 1.0, length=nx)
us2 = range(-1.0, 1.0, length=ny)

Z = Matrix{Float64}(undef, ny, nx)   # rows u2, cols u1
for (iy, u2) in enumerate(us2), (ix, u1) in enumerate(us1)
    Z[iy, ix] = Qdet_u([u1, u2])
end

p = contourf(us1, us2, Z;
    xlabel="u₁ (scaled f_cl)", ylabel="u₂ (scaled f_sb)",
    title="Q_det landscape with recommendations",
    colorbar_title="Q_det")

scatter!(p, [u_rec_det[1]], [u_rec_det[2]]; ms=6, label="BayesOpt EI (det)", color=:white)
scatter!(p, [res_het.x_rec[1]], [res_het.x_rec[2]]; ms=6, label="Hetero UCB (noisy)", color=:black)

savefig(p, "compare_recommendations.png")
println("\nSaved: compare_recommendations.png")
