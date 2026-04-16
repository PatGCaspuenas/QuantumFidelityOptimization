import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Distributed
using Statistics
using LaTeXStrings

# ── Configuration ────────────────────────────────────────────────────────────

# Well-performing seed from benchmark_fixedN250 (Q_det=0.9999)
const INIT_SEED  = 849665    # seeds the BO initial design points
const N_SHOTS    = 100
const N_INIT     = 12
const N_ITER     = 50
const NUM_MS     = 2
const κ_VAL      = 2.0
const α_VAL      = 1.5

# Noise parameters (from evaluate_ionsim.jl)
const δ_RMS_HZ    = 300.0    # detuning noise RMS in Hz
const Ω_RMS_FRAC  = 0.007    # fractional Rabi frequency noise

# ── Workers for parallel MC shots ────────────────────────────────────────────
if nprocs() == 1
    n_workers_cfg = get(ENV, "MC_N_WORKERS", "")
    n_workers_add = isempty(n_workers_cfg) ? max(1, Sys.CPU_THREADS - 1) : parse(Int, n_workers_cfg)
    addprocs(n_workers_add)
end

try

@everywhere begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
    include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
    using Random
end

# ── Baseline parameters ──────────────────────────────────────────────────────

t = 100.0
base = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

# 3D parameter spans (same as benchmark script)
span_kHz = 1.0
span_fcl = span_kHz * 1e3 * 2π
span_fsb = span_kHz * 1e3 * 2π
span_A   = 1.2 * A0 - A0

# Broadcast constants to workers
@everywhere _t       = $t
@everywhere _f_cl0   = $f_cl0
@everywhere _f_sb0   = $f_sb0
@everywhere _A0      = $A0
@everywhere _span_fcl = $span_fcl
@everywhere _span_fsb = $span_fsb
@everywhere _span_A   = $span_A
@everywhere _NUM_MS  = $NUM_MS
@everywhere _δ_RMS   = $(δ_RMS_HZ * 2π)
@everywhere _Ω_RMS   = $Ω_RMS_FRAC

@everywhere u_to_params(u) = (_f_cl0 + _span_fcl * u[1],
                              _f_sb0 + _span_fsb * u[2],
                              _A0    + _span_A   * u[3])

# ── Parallel MC helper ───────────────────────────────────────────────────────

@everywhere function _mc_single_shot(f_cl::Float64, Δ::Float64, I_val::Float64,
                                     δ_noise::Float64, ε_noise::Float64)::Float64
    Δ_noisy = Δ + δ_noise
    I_noisy = I_val * (1.0 + ε_noise)^2
    return CalibrationCode.Q_varMS(_t, f_cl, Δ_noisy, I_noisy; N=1, numMS=_NUM_MS)
end

function run_mc_parallel(f_cl::Float64, Δ::Float64, I_val::Float64,
                         N::Int, rng::AbstractRNG)::Float64
    δs = randn(rng, N) .* _δ_RMS
    εs = randn(rng, N) .* _Ω_RMS
    results = pmap(i -> _mc_single_shot(f_cl, Δ, I_val, δs[i], εs[i]), 1:N)
    return mean(results)
end

# ── 3D Objective functions ───────────────────────────────────────────────────

function Q_fun_proj(u, N::Int)
    fcl, fsb, A = u_to_params(u)
    y = CalibrationCode.Q_varMS(_t, fcl, fsb, A; N=N, numMS=_NUM_MS)
    σy = sqrt(max(y * (1.0 - y), 0.0) / N)
    return (clamp(y, 0.0, 1.0), σy)
end

rng_mc = MersenneTwister(42)
function Q_fun_mc(u, N::Int)
    fcl, fsb, A = u_to_params(u)
    y = run_mc_parallel(fcl, fsb, A, N, rng_mc)
    σy = sqrt(max(y * (1.0 - y), 0.0) / N)
    return (clamp(y, 0.0, 1.0), σy)
end

# ── Run 3D BO and reconstruct GP ─────────────────────────────────────────────

function run_bo_3d(f, seed::Int)
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    res = CalibrationCode.bayesopt_ucb_threshold(f;
        bounds=bounds,
        n_shots=N_SHOTS,
        n_init=N_INIT,
        n_iter=N_ITER,
        κ=κ_VAL,
        α=α_VAL,
        seed=seed,
        maximize=true,
        hyper_every=10,
        learn_noise_scale=true,
        n_restarts=6,
    )

    θ_final = Float64[log.(res.ℓ_final)..., log(res.σf_final), log(res.c_final)]
    gp = CalibrationCode.fit_heterogp(res.X, res.y, res.σy;
        learn_hypers=false, θ_init=θ_final)

    return res, gp
end

# ── Slice 3D GP along one dimension ──────────────────────────────────────────
# Sweep dimension `dim` while holding others at 0.0 (ideal)

function slice_gp(gp, dim::Int; n_grid::Int=200)
    xs = range(-1.0, 1.0, length=n_grid)
    μs = zeros(n_grid)
    σs = zeros(n_grid)
    for (i, xv) in enumerate(xs)
        u = [0.0, 0.0, 0.0]
        u[dim] = xv
        μ, s2 = CalibrationCode.predict_latent(gp, u)
        μs[i] = μ
        σs[i] = sqrt(max(s2, 0.0))
    end
    return collect(xs), μs, σs
end

# ── Plotting ─────────────────────────────────────────────────────────────────

using PyPlot

function make_plot(xs_phys, y_true, μ_proj, σ_proj, x_rec_proj_phys,
                   μ_mc, σ_mc, x_rec_mc_phys;
                   xlabel_str::AbstractString, out_path::AbstractString)

    fig, ax = subplots(figsize=(10, 6))

    ax.plot(xs_phys, y_true, color="black", linewidth=2.5, label="True Q_det", zorder=4)

    ax.plot(xs_phys, μ_proj, color="royalblue", linewidth=2.0, label="GP mean (no drift)", zorder=3)
    ax.fill_between(xs_phys, μ_proj .- σ_proj, μ_proj .+ σ_proj,
                    color="lightskyblue", alpha=0.4, zorder=2)
    ax.axvline(x_rec_proj_phys, color="royalblue", linestyle="--", linewidth=1.5,
               label="Rec. point (baseline)", zorder=3)

    ax.plot(xs_phys, μ_mc, color="firebrick", linewidth=2.0, label="GP mean (drifted)", zorder=3)
    ax.fill_between(xs_phys, μ_mc .- σ_mc, μ_mc .+ σ_mc,
                    color="lightcoral", alpha=0.35, zorder=2)
    ax.axvline(x_rec_mc_phys, color="firebrick", linestyle="--", linewidth=1.5,
               label="Rec. point (drifted)", zorder=3)

    ax.set_xlabel(xlabel_str, fontsize=14)
    ax.set_ylabel(L"\mathcal{F}", fontsize=16)
    ax.legend(fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(true, linestyle=":", alpha=0.4)
    ax.tick_params(labelsize=12)

    tight_layout()
    savefig(out_path, dpi=200)
    println("Saved: $out_path")
    close(fig)
end

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — run TWO 3D BO sequences, then slice GPs for plots
# ══════════════════════════════════════════════════════════════════════════════

mkpath(joinpath(@__DIR__, "plots"))

println("\n=== Running 3D BO (projection noise only) ===")
res_proj, gp_proj = run_bo_3d(Q_fun_proj, INIT_SEED)
fcl_rec_p, fsb_rec_p, A_rec_p = u_to_params(res_proj.x_rec)
Q_det_proj = CalibrationCode.Q_det(_t, fcl_rec_p, fsb_rec_p, A_rec_p)
println("  → Rec Q_det = $(round(Q_det_proj, digits=5))")

println("\n=== Running 3D BO (full MC noise) ===")
res_mc, gp_mc = run_bo_3d(Q_fun_mc, INIT_SEED)
fcl_rec_m, fsb_rec_m, A_rec_m = u_to_params(res_mc.x_rec)
Q_det_mc = CalibrationCode.Q_det(_t, fcl_rec_m, fsb_rec_m, A_rec_m)
println("  → Rec Q_det = $(round(Q_det_mc, digits=5))")

# ── Slice along f_sb (dim 2) ────────────────────────────────────────────────
println("\nGenerating f_sb slice plots...")

xs_fsb, μ_proj_fsb, σ_proj_fsb = slice_gp(gp_proj, 2)
_, μ_mc_fsb, σ_mc_fsb = slice_gp(gp_mc, 2)

# True Q_det landscape along f_sb (f_cl, A at ideal)
y_true_fsb = [clamp(CalibrationCode.Q_det(_t, _f_cl0, _f_sb0 + span_fsb * x, _A0), 0.0, 1.0)
              for x in xs_fsb]

xs_phys_fsb = xs_fsb .* (span_fsb / (2π * 1e3))   # kHz
x_rec_proj_phys_fsb = res_proj.x_rec[2] * (span_fsb / (2π * 1e3))
x_rec_mc_phys_fsb   = res_mc.x_rec[2]   * (span_fsb / (2π * 1e3))

make_plot(xs_phys_fsb, y_true_fsb, μ_proj_fsb, σ_proj_fsb, x_rec_proj_phys_fsb,
          μ_mc_fsb, σ_mc_fsb, x_rec_mc_phys_fsb;
          xlabel_str=L"f_\mathrm{sb}\;\mathrm{offset}\;/\;(2\pi)\;\;(\mathrm{kHz})",
          out_path=joinpath(@__DIR__, "plots", "gp_noise_comparison_fsb.png"))

# ── Slice along A (dim 3) ───────────────────────────────────────────────────
println("Generating Omega slice plots...")

xs_A, μ_proj_A, σ_proj_A = slice_gp(gp_proj, 3)
_, μ_mc_A, σ_mc_A = slice_gp(gp_mc, 3)

# True Q_det landscape along A (f_cl, f_sb at ideal)
y_true_A = [clamp(CalibrationCode.Q_det(_t, _f_cl0, _f_sb0, _A0 + span_A * x), 0.0, 1.0)
            for x in xs_A]

xs_phys_A = xs_A .* (span_A / _A0) .* 100   # percent of A0
x_rec_proj_phys_A = res_proj.x_rec[3] * (span_A / _A0) * 100
x_rec_mc_phys_A   = res_mc.x_rec[3]   * (span_A / _A0) * 100

make_plot(xs_phys_A, y_true_A, μ_proj_A, σ_proj_A, x_rec_proj_phys_A,
          μ_mc_A, σ_mc_A, x_rec_mc_phys_A;
          xlabel_str=L"\Omega\;\mathrm{offset}\;\;(\%\;\mathrm{of}\;\Omega_0)",
          out_path=joinpath(@__DIR__, "plots", "gp_noise_comparison_A.png"))

println("\nDone.")

finally
    rmprocs(workers())
end
