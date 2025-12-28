include("bayes_opt.jl");           using .BayesOpt
include("bayes_hetero_opt.jl");    using .BayesHeteroOpt
include("bayes_opt_plots.jl");     using .BayesPlotting

using Random
using Distributions
using Plots

include("calibration.jl")

# --- baseline from ideal ---
t = 100.0
res0 = ideal(t)                   # [fid, f_cl, f_sb, A, delta_phi]
f_cl0, f_sb0, A0 = res0[2], res0[3], res0[4]

# --- search range: ±20 kHz around baseline for BOTH f_cl and f_sb ---
span_hz = 0.1e4
bounds = [(-1.0, 1.0), (-1.0, 1.0)]  # BO works in u-space

# map u ∈ [-1,1]^2  ->  physical Hz
u_to_freq(u) = (f_cl0 + span_hz*u[1],  f_sb0 + span_hz*u[2])

# map "noise setting" σ -> number of shots N (integer)
# simplest: N ≈ 1/σ^2  (scale as you prefer)
N_from_sigma(σ) = max(1, round(Int, 1 / (σ^2)))

# Objective compatible with bayesopt_ucb_threshold: f(u, σ) -> noisy fidelity
function Q_fun_2d(u, σ)
    fcl, fsb = u_to_freq(u)
    N = N_from_sigma(σ)
    return (Q_noisy(t, fcl, fsb, A0; N=N))
end

function Q_true_2d(u)
    fcl, fsb = u_to_freq(u)
    return (Q_det(t, fcl, fsb, A0))
end

σ_levels = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]   # "noise knobs" (smaller -> more shots)

resH = BayesHeteroOpt.bayesopt_ucb_threshold(Q_fun_2d;
    bounds=bounds,
    σ_levels=σ_levels,
    n_init=6,
    n_iter=100,
    κ=2.0,
    α=0.5,
    seed=1
)

println("Recommended u = ", resH.x_rec, "   GP-mean y ≈ ", resH.y_rec)

# recommended physical frequencies
fcl_rec, fsb_rec = u_to_freq(resH.x_rec)
println("Recommended f_cl = ", fcl_rec)
println("Recommended f_sb = ", fsb_rec)
println("Trye f_cl = ", f_cl0)
println("Recommended f_sb = ", f_sb0)

counts = BayesHeteroOpt.count_noise_levels(resH)
for σ in resH.σ_levels
    println("σ=$(σ): ", counts[σ])
end

# Plot without f_true unless you define a deterministic reference
display(BayesPlotting.plot2d(resH))

anim, fps = BayesPlotting.animate2d(resH; f_true=Q_true_2d, fps=5, nx=10, ny=10)
gif(anim, "hetero.gif", fps=fps)
