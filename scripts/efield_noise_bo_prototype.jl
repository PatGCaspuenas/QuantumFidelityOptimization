# scripts/efield_noise_bo_prototype.jl
#
# Prototype: 1/f^α electric field noise effects on BO calibration.
#
# Physics model:
#   Electric field noise causes slow motional mode frequency drift Δω_m(i)
#   on the calibration timescale (seconds–minutes), NOT within a single gate
#   (100 µs — drift is negligible and averaged out over one gate).
#   Each BO evaluation i receives a fidelity measurement at the drifted
#   effective sideband detuning:
#       f_sb_eff[i] = f_sb_programmed[i] + Δω_m[i]
#   The BO only sees f_sb_programmed; the GP is unaware of the drift.
#   This mismatch between programmed and actual parameters is the core
#   degradation mechanism studied here.
#
# No changes to any existing source file — fully self-contained.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

using Random, Statistics, LinearAlgebra, FFTW

ENV["MPLBACKEND"] = "Agg"
using PyPlot, LaTeXStrings
using Printf

# ── Physical / BO constants (match benchmark script) ────────────────────
const t_gate = 100.0
base = CalibrationCode.ideal(t_gate)
const f_cl0 = base.f_cl
const f_sb0 = base.f_sb
const A0 = base.A
@printf "Baseline: f_cl0=%.2f Hz  f_sb0=%.2f Hz  A0=%.4f\n" f_cl0 f_sb0 A0
@printf "Baseline fidelity: %.6f\n\n" base.fid

const span_kHz = 2.0
const span_fcl = span_kHz * 1e3 * 2π
const span_fsb = span_kHz * 1e3 * 2π
const span_A = 1.2 * A0 - A0

u_to_params(u) = (f_cl0 + span_fcl * u[1],
    f_sb0 + span_fsb * u[2],
    A0 + span_A * u[3])

N_from_sigma(σ::Float64) = clamp(round(Int, 1 / σ^2), 20, 10000)

const σ_levels = [0.1412, 0.1, 0.06, 0.04472]
const bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
const κ_bo = 1.9
const α_bo = 1.5   # BO threshold param (distinct from noise exponent below)

# ── Noise prototype config ───────────────────────────────────────────────
const noise_α = 1.0     # exponent: 0=white, 1=pink (1/f), 2=Brownian
const noise_amp_Hz = 500.0   # RMS drift amplitude in Hz (~0.5 kHz)
const n_init = 12
const n_iter = 100      # increase to 60–120 for fuller study
const num_seeds = 5
const N_CALLS = n_init + n_iter

# ── 1. Colored noise generator — Timmer & Koenig (1995) ─────────────────
"""
    generate_colored_noise(n, α, amplitude_Hz; rng) → Vector{Float64}

1/f^α noise via spectral filtering. Zero-mean; RMS amplitude = amplitude_Hz.
DC component is zeroed to prevent unbounded mean drift.
"""
function generate_colored_noise(n::Int, α::Float64, amplitude_Hz::Float64;
    rng::Random.AbstractRNG=Random.default_rng())
    n >= 4 || throw(ArgumentError("n must be ≥ 4"))
    freqs = collect(rfftfreq(n)) # mutable copy; Frequencies is read-only
    freqs[1] = 1.0               # avoid DC ÷ 0
    filter_amp = freqs .^ (-α / 2.0)
    white = randn(rng, length(freqs)) .+ im .* randn(rng, length(freqs))
    colored = white .* filter_amp
    colored[1] = 0.0 + 0.0im    # zero DC → zero mean drift
    signal = real(irfft(colored, n))
    s = std(signal)
    return s < 1e-12 ? zeros(n) : (signal ./ s) .* amplitude_Hz
end

# ── 2. Q_fun factory with injected drift ────────────────────────────────
"""
    make_Q_fun(drift) → (Q_fun, call_count)

Returns a closure Q_fun(u, σ) that evaluates Q_varMS at
    f_sb_eff = f_sb_programmed + drift[call_index]
on each successive call. Pass drift=zeros(N_CALLS) for the no-drift baseline.
The BO sees only f_sb_programmed; the gate runs with f_sb_eff.
"""
function make_Q_fun(drift::Vector{Float64})
    call_count = Ref(0)
    function Q_fun(u, σ)
        call_count[] += 1
        i = min(call_count[], length(drift))
        fcl, fsb, A = u_to_params(u)
        # Suppress IonSim's per-call stderr chatter while preserving error propagation
        old_stderr = redirect_stderr(devnull)
        result = try
            CalibrationCode.Q_varMS(t_gate, fcl, fsb + drift[i], A;
                N=N_from_sigma(σ), numMS=2)
        finally
            redirect_stderr(old_stderr)
        end
        return result
    end
    return Q_fun, call_count
end

# ── 3. True 1D landscape (f_sb slice at f_cl0, A0) ──────────────────────
println("Precomputing Q_det landscape (41 points along f_sb)…")
const n_ls = 41
const fsb_offsets_Hz = collect(range(-2000.0, 2000.0; length=n_ls))
const fsb_offsets_kHz = fsb_offsets_Hz ./ 1e3
function _q_det_quiet(args...)
    old_se = redirect_stderr(devnull)
    val = try
        CalibrationCode.Q_det(args...)
    finally
        redirect_stderr(old_se)
    end
    return val
end

landscape_F = Vector{Float64}(undef, n_ls)
for j in 1:n_ls
    landscape_F[j] = _q_det_quiet(t_gate, f_cl0, f_sb0 + fsb_offsets_Hz[j], A0)
    (j % 10 == 0 || j == n_ls) && (@printf "  %2d/%d\n" j n_ls; flush(stdout))
end
const Q_opt = maximum(landscape_F)
println("  Peak Q_det = $(round(Q_opt; digits=5))\n")

# ── 4. BO runs ──────────────────────────────────────────────────────────
println("Running BO: $num_seeds seeds × 2 conditions (drift / no-drift)…")
println("  (first call triggers JIT — may be slow for ~1 min before output resumes)")
flush(stdout)
rng_master = MersenneTwister(2024)
bo_seeds = rand(rng_master, 1:99999, num_seeds)

results_drift = Vector{HeteroBOResult}(undef, num_seeds)
results_baseline = Vector{HeteroBOResult}(undef, num_seeds)
drifts_all = Vector{Vector{Float64}}(undef, num_seeds)

for k in 1:num_seeds
    seed = bo_seeds[k]
    rng = MersenneTwister(seed)
    drift = generate_colored_noise(N_CALLS, noise_α, noise_amp_Hz; rng=rng)
    drifts_all[k] = drift

    Q_d, _ = make_Q_fun(drift)
    Q_b, _ = make_Q_fun(zeros(N_CALLS))

    @printf "  [%d/%d] seed=%d  (drifted)…" k num_seeds seed
    flush(stdout)
    results_drift[k] = try
        CalibrationCode.bayesopt_ucb_threshold(Q_d;
            bounds=bounds, σ_levels=σ_levels,
            n_init=n_init, n_iter=n_iter,
            κ=κ_bo, α=α_bo, seed=seed)
    catch e
        println("\nERROR in drifted BO (seed=$seed): $e")
        Base.showerror(stdout, e, catch_backtrace())
        rethrow()
    end

    @printf " (baseline)…"
    flush(stdout)
    results_baseline[k] = try
        CalibrationCode.bayesopt_ucb_threshold(Q_b;
            bounds=bounds, σ_levels=σ_levels,
            n_init=n_init, n_iter=n_iter,
            κ=κ_bo, α=α_bo, seed=seed)
    catch e
        println("\nERROR in baseline BO (seed=$seed): $e")
        Base.showerror(stdout, e, catch_backtrace())
        rethrow()
    end

    qd = _q_det_quiet(t_gate, u_to_params(results_drift[k].x_rec)...)
    qb = _q_det_quiet(t_gate, u_to_params(results_baseline[k].x_rec)...)
    @printf " Q_det: drift=%.4f  baseline=%.4f\n" qd qb
    flush(stdout)
end

# ── 5. Convergence traces (running best of noisy y) ─────────────────────
running_max(v) = accumulate(max, v)

traces_drift = [running_max(results_drift[k].y[1:N_CALLS]) for k in 1:num_seeds]
traces_baseline = [running_max(results_baseline[k].y[1:N_CALLS]) for k in 1:num_seeds]

Q_det_drift = [_q_det_quiet(t_gate, u_to_params(results_drift[k].x_rec)...)
               for k in 1:num_seeds]
Q_det_baseline = [_q_det_quiet(t_gate, u_to_params(results_baseline[k].x_rec)...)
                  for k in 1:num_seeds]

println("\n=== Summary ===")
@printf "Baseline Q_det: %.4f ± %.4f\n" mean(Q_det_baseline) std(Q_det_baseline)
@printf "Drifted  Q_det: %.4f ± %.4f\n" mean(Q_det_drift) std(Q_det_drift)

# ── 6. GP posteriors on 1D f_sb slice ───────────────────────────────────
# Grid: u = [0, u₂, 0] sweeping f_sb at fixed f_cl=f_cl0, A=A0
println("\nFitting GP posteriors for landscape panel…")
u_grid = [Float64[0.0, ε/span_fsb, 0.0] for ε in fsb_offsets_Hz]

function gp_posterior_1d(res::HeteroBOResult)
    n = res.n_init + res.n_iter_actual
    gp = fit_heterogp(res.X[:, 1:n], res.y[1:n], res.σy[1:n];
        learn_hypers=true, n_restarts=3)
    μs = Vector{Float64}(undef, length(u_grid))
    σs = Vector{Float64}(undef, length(u_grid))
    for (j, u) in enumerate(u_grid)
        μj, s2j = predict_latent(gp, u)
        μs[j] = μj
        σs[j] = sqrt(max(s2j, 0.0))
    end
    return μs, σs
end

# Use seed 1 as the representative case for panels (a)–(c)
μ_d, σ_d = gp_posterior_1d(results_drift[1])
μ_b, σ_b = gp_posterior_1d(results_baseline[1])
println("  Done.")

# ── 7. Figure ────────────────────────────────────────────────────────────
println("\nGenerating figure…")

rcP = PyPlot.matplotlib.rcParams
rcP["font.family"] = "serif"
rcP["mathtext.fontset"] = "cm"
rcP["font.size"] = 9.0
rcP["axes.labelsize"] = 9.0
rcP["xtick.labelsize"] = 8.0
rcP["ytick.labelsize"] = 8.0
rcP["axes.linewidth"] = 0.8
rcP["xtick.direction"] = "in"
rcP["ytick.direction"] = "in"
rcP["xtick.top"] = true
rcP["ytick.right"] = true

C_d = "#c0392b"   # red   — drifted
C_b = "#2980b9"   # blue  — baseline
C_t = "#2c3e50"   # dark  — truth

fig = figure(figsize=(11, 8))
fig.subplots_adjust(hspace=0.42, wspace=0.30,
    left=0.09, right=0.97, top=0.94, bottom=0.09)
ax_a = subplot(2, 2, 1)
ax_b = subplot(2, 2, 2)
ax_c = subplot(2, 2, 3)
ax_d = subplot(2, 2, 4)

# ─── Panel (a): Mode frequency drift trajectory ──────────────────────────
dk_kHz = drifts_all[1] ./ 1e3
ev_idx = 1:N_CALLS

ax_a.axvspan(1, n_init + 0.5, alpha=0.07, color="gray")
ax_a.axvline(n_init + 0.5, color="gray", lw=0.8, ls=":", alpha=0.7)
ax_a.axhline(0.0, color="gray", lw=0.6, ls="--", alpha=0.5)
ax_a.plot(ev_idx, dk_kHz, color=C_d, lw=1.3, zorder=3)

ylim_mag = max(abs(minimum(dk_kHz)), abs(maximum(dk_kHz))) * 1.2
ax_a.set_ylim(-ylim_mag, ylim_mag)
ax_a.text(n_init / 2.0, ylim_mag * 0.82, "init",
    ha="center", va="center", fontsize=7, color="gray")
ax_a.text(n_init + n_iter / 2.0, ylim_mag * 0.82, "BO iters",
    ha="center", va="center", fontsize=7, color="gray")

ax_a.set_xlabel("Evaluation index")
ax_a.set_ylabel(L"\Delta\omega_m\,/(2\pi)\;\mathrm{(kHz)}")
ax_a.set_title(
    L"(a)\;\mathrm{Mode\;freq.\;drift\;}(1/f,\;500\,\mathrm{Hz\;RMS})",
    fontsize=9)
ax_a.set_xlim(1, N_CALLS)

# ─── Panel (b): Power spectral density ──────────────────────────────────
# Use the BO phase only (n_iter points — avoids boundary effects from init)
sig_bo = drifts_all[1][n_init+1:end]
pfreqs = rfftfreq(length(sig_bo))
psd = abs2.(rfft(sig_bo)) ./ length(sig_bo)
f_psd = collect(pfreqs[2:end])   # skip DC; plain Vector for broadcasting
p_psd = collect(psd[2:end])
ref_line = p_psd[1] .* (f_psd[1] ./ f_psd) .^ noise_α

ax_b.loglog(f_psd, p_psd, color=C_d, lw=1.4, label="Measured PSD")
ax_b.loglog(f_psd, ref_line, color="gray", lw=1.0, ls="--",
    label=L"1/f^\alpha\;\mathrm{ref.}")
ax_b.set_xlabel("Frequency (cycles / iteration)")
ax_b.set_ylabel("PSD (arb.)")
ax_b.set_title(L"(b)\;\mathrm{Drift\;power\;spectrum}", fontsize=9)
ax_b.legend(fontsize=7, loc="lower left")

# ─── Panel (c): True landscape + GP posteriors (1D f_sb slice) ──────────
ax_c.plot(fsb_offsets_kHz, landscape_F,
    color=C_t, lw=2.0, zorder=5, label="True Q_det")

ax_c.plot(fsb_offsets_kHz, μ_b, color=C_b, lw=1.4, label="GP mean (no drift)")
ax_c.fill_between(fsb_offsets_kHz, μ_b .- σ_b, μ_b .+ σ_b,
    color=C_b, alpha=0.15)

ax_c.plot(fsb_offsets_kHz, μ_d, color=C_d, lw=1.4, label="GP mean (drifted)")
ax_c.fill_between(fsb_offsets_kHz, μ_d .- σ_d, μ_d .+ σ_d,
    color=C_d, alpha=0.15)

# Mark recommended f_sb for seed 1 (projected onto f_sb axis)
rec_b_kHz = (u_to_params(results_baseline[1].x_rec)[2] - f_sb0) / 1e3
rec_d_kHz = (u_to_params(results_drift[1].x_rec)[2] - f_sb0) / 1e3
ax_c.axvline(rec_b_kHz, color=C_b, lw=1.0, ls="--", alpha=0.75,
    label="Rec. point (baseline)")
ax_c.axvline(rec_d_kHz, color=C_d, lw=1.0, ls="--", alpha=0.75,
    label="Rec. point (drifted)")

ax_c.set_xlabel(L"f_\mathrm{sb}\;\mathrm{offset}\,/(2\pi)\;\mathrm{(kHz)}")
ax_c.set_ylabel(L"\mathcal{F}")
ax_c.set_title(
    L"(c)\;\mathrm{GP\;posterior\;vs.\;true\;landscape}\;(f_\mathrm{sb}\;\mathrm{slice})",
    fontsize=9)
ax_c.legend(fontsize=6.5, loc="lower center", ncol=2)
ax_c.set_xlim(extrema(fsb_offsets_kHz))
ax_c.set_ylim(0.70, 1.02)
ax_c.grid(true, alpha=0.10, lw=0.5)

# ─── Panel (d): BO convergence (running best noisy y, multi-seed) ────────
mat_d = hcat(traces_drift...)       # N_CALLS × num_seeds
mat_b = hcat(traces_baseline...)
μ_td = vec(mean(mat_d, dims=2));
σ_td = vec(std(mat_d, dims=2));
μ_tb = vec(mean(mat_b, dims=2));
σ_tb = vec(std(mat_b, dims=2));

ax_d.axvline(n_init + 0.5, color="gray", lw=0.8, ls=":", alpha=0.6)
ax_d.axhline(Q_opt, color=C_t, lw=0.8, ls="--", alpha=0.5,
    label="True Q_det peak")

ax_d.plot(ev_idx, μ_tb, color=C_b, lw=1.4, label="No drift")
ax_d.fill_between(ev_idx, μ_tb .- σ_tb, μ_tb .+ σ_tb,
    color=C_b, alpha=0.15)

ax_d.plot(ev_idx, μ_td, color=C_d, lw=1.4, label="Drifted (500 Hz RMS)")
ax_d.fill_between(ev_idx, μ_td .- σ_td, μ_td .+ σ_td,
    color=C_d, alpha=0.15)

# Dotted lines at mean final Q_det (true, not noisy running-best)
ax_d.axhline(mean(Q_det_baseline), color=C_b, lw=0.8, ls=":", alpha=0.65)
ax_d.axhline(mean(Q_det_drift), color=C_d, lw=0.8, ls=":", alpha=0.65)

ax_d.set_xlabel("Evaluation index (init + BO)")
ax_d.set_ylabel(L"\mathrm{Running\;best}\;Q_\mathrm{noisy}")
ax_d.set_title("(d) BO convergence ($num_seeds seeds)", fontsize=9)
ax_d.legend(fontsize=7)
ax_d.set_xlim(1, N_CALLS)
ax_d.grid(true, alpha=0.10, lw=0.5)

# ── Save ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "plots")
mkpath(outdir)
outfile = joinpath(outdir, "efield_noise_prototype.png")
savefig(outfile, dpi=200, bbox_inches="tight")
println("Saved → $outfile")

# ── Final summary ────────────────────────────────────────────────────────
println("\n=== Final Results ===")
println("Noise: α=$(noise_α), amplitude=$(noise_amp_Hz) Hz RMS")
println("       n_init=$n_init, n_iter=$n_iter, seeds=$num_seeds")
@printf "Baseline Q_det: %.4f ± %.4f (mean ± std)\n" mean(Q_det_baseline) std(Q_det_baseline)
@printf "Drifted  Q_det: %.4f ± %.4f\n" mean(Q_det_drift) std(Q_det_drift)
@printf "True peak Q_det:                 %.4f\n" Q_opt
@printf "Degradation (mean):              %.4f\n" mean(Q_det_baseline) - mean(Q_det_drift)
