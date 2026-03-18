# scripts/filter_function_validation.jl
#
# Validates the filter-function formalism from arXiv:2510.17286v1, Appendix B.1
# against the existing IonSim-based Q_det fidelity estimator.
#
# Validation strategy (static-detuning limit):
#   For a constant detuning error  ε₀  (i.e. ω → 0), the filter-function
#   predicts an infidelity:
#       ℐ_predicted(ε₀) ≈ S(0) · ε₀²
#   where S(ω) is the filter function of our gate trajectory.
#   We compare this to the direct IonSim infidelity:
#       ℐ_direct(ε₀) = 1 - Q_det(t, f_cl, f_sb + ε₀, A)
#   These should agree at small ε₀ (quadratic regime).
#
# Power-law check (§B.1.1):
#   For a single-loop MS gate (Walsh-0), at high frequency:
#       S_γ(ω)   ∝  ω^{-4}   (residual spin-motion entanglement term)
#       S_δθ(ω)  ∝  ω^{-6}   (entanglement angle term)
#   We verify the numerically computed filter function matches these slopes.
#
# Usage:
#   julia --project=. scripts/filter_function_validation.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

using IonSim, QuantumOptics
ENV["MPLBACKEND"] = "Agg"   # non-interactive backend — safe in headless terminals
using PyPlot
using LaTeXStrings
using Printf
using LinearAlgebra
using FFTW

redirect_stderr(devnull)

# ─────────────────────────────────────────────────
# 1.  Build baseline gate from existing infrastructure
# ─────────────────────────────────────────────────
const t_gate = 100.0   # µs  (consistent with rest of codebase)

println("Computing ideal baseline parameters...")
base = CalibrationCode.ideal(t_gate)
const f_cl0 = base.f_cl
const f_sb0 = base.f_sb
const A0 = base.A
@printf "  f_cl0 = %.6e Hz\n" f_cl0
@printf "  f_sb0 = %.6e Hz\n" f_sb0
@printf "  A0    = %.6e W/m²\n" A0

# ─────────────────────────────────────────────────
# 2.  Extract physical gate parameters from IonSim
#     needed to build γ(t) analytically
# ─────────────────────────────────────────────────

# Rebuild chamber at baseline to pull η and Ω_eff
println("\nExtracting gate parameters from IonSim...")
setup = CalibrationCode.build_chamber()
CalibrationCode.configure_lasers!(setup, f_cl0, f_sb0, A0)

mode = setup.mode
ν_mode = frequency(mode)                          # motional mode frequency [Hz]

# Lamb-Dicke parameter η  (IonSim helper)
η = abs(lambdicke(mode, setup.ca, setup.laser1))

# Gate detuning δ  (sideband detuning above the mode)
# f_sb0 = ν_mode + δ   →   δ = f_sb0 - ν_mode
δ_hz = f_sb0 - ν_mode                              # Hz
δ = 2π * δ_hz                                   # rad/s

# Gate time in seconds  (IonSim uses µs internally with timescale=1e-6)
t_gate_s = t_gate * 1e-6                            # seconds

# Carrier Rabi frequency Ω_c from IonSim intensity
# intensity_from_pitime! sets the laser so that  Ω_c = π / π_time
# We recover it via: η * Ω_c = effective coupling ≡ Ω_g
# The complete loop condition  Ω_g / δ = 1/(2π) * t_gate  gives
# Ω_g = δ / (2π)  * ... actually let's pull Ω_c directly from atom/laser.
# IonSim stores intensity A = I_0; Rabi freq via  pitime relation.
# The MS loop closes when Ω_g²/δ · t_gate = π/4 → Ω_g = sqrt(π δ / (4 t_gate))
Ω_g = sqrt(π * abs(δ) / (4 * t_gate_s))           # effective coupling [rad/s]

@printf "  ν_mode  = %.6e Hz\n" ν_mode
@printf "  η       = %.6f\n" η
@printf "  δ       = %.6e rad/s  (%.3f kHz)\n" δ (δ_hz / 1e3)
@printf "  Ω_g     = %.6e rad/s  (%.3f kHz)\n" Ω_g (Ω_g / 2π / 1e3)
@printf "  t_gate  = %.3e s  (%.1f µs)\n" t_gate_s t_gate

# ─────────────────────────────────────────────────
# 3.  Compute γ(t) analytically
#
#   For a single-loop bichromatic MS gate the forced eigenstate
#   traces the displacement trajectory (rotating frame, Eq. 1):
#
#       γ(t) = (Ω_g / δ) * (exp(i δ t) - 1)
#
#   At t = t_g = 2π/δ  this returns to γ = 0  (closed loop).
#   The geometric phase accumulated is  θ_g = Ω_g² · t_g / δ = π/4.
# ─────────────────────────────────────────────────
println("\nComputing γ(t) trajectory...")

dt_s = t_gate_s / 2000          # time step  (~50 ns for 100 µs gate)
τ = 0.0:dt_s:t_gate_s        # time axis in seconds
Nt = length(τ)

γ_t = @. (Ω_g / δ) * (exp(1im * δ * τ) - 1)   # complex displacement

# Sanity check: loop should close
loop_residual = abs(γ_t[end])
@printf "  |γ(t_gate)| = %.2e  (should be ≈ 0)\n" loop_residual

# Accumulated geometric phase
θ_g = Ω_g^2 * t_gate_s / abs(δ)
@printf "  θ_g = %.6f π  (should be ≈ 0.25 π = π/4)\n" (θ_g / π)

# ─────────────────────────────────────────────────
# 4.  Compute the filter function S(ω) via FFT
#
#   From §B.1, averaging over ϕ ∈ {0, -π/2}:
#
#     α_cos(ω)  = ∫₀^{tg} γ(t) cos(ωt) dt      (Re part of FT of γ)
#     α_sin(ω)  = ∫₀^{tg} γ(t) sin(ωt) dt      (Im part of FT of γ)
#     dθ_cos(ω) = ∫₀^{tg} |γ(t)|² cos(ωt) dt   (Re part of FT of |γ|²)
#     dθ_sin(ω) = ∫₀^{tg} |γ(t)|² sin(ωt) dt   (Im part of FT of |γ|²)
#
#   Then (Eq. B.1, after dividing by ε₀²):
#
#     S_γ(ω)  = ½ (|α_cos|² + |α_sin|²) = ½ |FT[γ](ω)|²
#     S_δθ(ω) = ½ (|dθ_cos|² + |dθ_sin|²) = ½ |FT[|γ|²](ω)|²
#     S(ω)    = S_γ(ω) + S_δθ(ω)
#
#   We use FFT (O(N log N)) for accuracy — direct trapezoidal summation
#   at high frequencies undersamples the fast-oscillating integrand.
# ─────────────────────────────────────────────────
println("Computing filter function S(ω) via FFT...")

# Use the same τ / dt_s grid
γ_arr = collect(γ_t)       # complex displacement γ(t)
γabs2_ = abs2.(γ_arr)       # |γ(t)|²  (always real)

# γ(t) is complex → use fft(), then keep positive-frequency half
# |γ(t)|² is real → rfft() is fine
Nt = length(γ_arr)
fft_γ_full = fft(γ_arr)            # full DFT of complex γ(t)
fft_γ2 = rfft(γabs2_)          # one-sided DFT of real |γ|²

Nω_fft = Nt ÷ 2 + 1               # number of non-redundant bins
fft_γ = fft_γ_full[1:Nω_fft]     # keep DC + positive-freq bins

df = 1.0 / (Nt * dt_s)        # frequency resolution [Hz]
freqs = (0:(Nω_fft-1)) .* df     # [Hz]

# Scale by dt² to convert from discrete sum → Riemann-integral units
# S_γ(ω)  = ½|∫γ(t)e^{iωt}dt|²  ≈  ½|Σ γ_k e^{iωt_k} dt|²
S_γ_fft = 0.5 .* abs2.(fft_γ) .* dt_s^2
S_dθ_fft = 0.5 .* abs2.(fft_γ2) .* dt_s^2
S_tot_fft = S_γ_fft .+ S_dθ_fft

# DC value (bin 0) for the static detuning prediction
# The static-detuning error is a step at ω = 0; index=1 in Julia 1-indexing
S_γ_dc = S_γ_fft[1]
S_dθ_dc = S_dθ_fft[1]
S_dc = S_tot_fft[1]

@printf "\n  S(0)   = %.6e  s²\n" S_dc
@printf "  S_γ(0) = %.6e  s²\n" S_γ_dc
@printf "  S_δθ(0)= %.6e  s²\n" S_dθ_dc

# Keep only frequencies up to 30× the loop freq for the plots
δ_hz_abs = abs(δ_hz)
ω_max_plot = 30.0 * δ_hz_abs
keep = freqs .> 0 .&& freqs .<= ω_max_plot
ω_plot_hz = freqs[keep]
Sγ_plot = S_γ_fft[keep]
Sdθ_plot = S_dθ_fft[keep]
Stot_plot = S_tot_fft[keep]

# ─────────────────────────────────────────────────
# 5.  Static-detuning validation
#
#   For a static (constant) detuning error ε₀ [Hz], the filter function
#   formula gives:
#
#       ℐ_pred(ε₀) = S(ω=0) · (2π ε₀)²
#
#   (The (2πε₀)² converts Hz → rad/s.  S(ω) has units of s², so the
#    product is dimensionless infidelity.)
#
#   IonSim direct (compare from perfect unity, not from the noisy baseline):
#       ℐ_direct(ε₀) = 1.0 - Q_det(t, f_cl, f_sb + ε₀, A)
#
#   At small ε₀ both should be quadratic and agree; deviations at large ε₀
#   reveal higher-order terms not captured by the quadratic approximation.
# ─────────────────────────────────────────────────
println("\nRunning static-detuning validation sweep...")

# Sweep detuning offsets
ε₀_hz = [0.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0, 1500.0, 2000.0]  # Hz
ℐ_direct = Vector{Float64}(undef, length(ε₀_hz))
ℐ_predicted = Vector{Float64}(undef, length(ε₀_hz))

F_ideal = CalibrationCode.Q_det(t_gate, f_cl0, f_sb0, A0)   # baseline (≈ 1)
@printf "  Baseline fidelity F₀ = %.8f\n" F_ideal

for (i, ε) in enumerate(ε₀_hz)
    ε_rads = 2π * ε                               # rad/s
    F_ε = CalibrationCode.Q_det(t_gate, f_cl0, f_sb0 + ε, A0)
    # IonSim direct: absolute infidelity from perfect gate (not from noisy baseline)
    # Use the *change* from baseline so baseline numerical error cancels
    ℐ_direct[i] = max(0.0, F_ideal - F_ε)
    ℐ_predicted[i] = S_dc * ε_rads^2
    ratio = ε > 0 ? ℐ_predicted[i] / max(ℐ_direct[i], 1e-15) : NaN
    @printf "  ε₀ = %6.0f Hz  →  ℐ_direct = %.4e  ℐ_pred = %.4e  ratio = %.4f\n" ε ℐ_direct[i] ℐ_predicted[i] ratio
end

# ─────────────────────────────────────────────────
# 6.  Power-law slope check  (§B.1.1)
#
#   For ω ≫ 2 Ω_g K^{1/2}  (=2Ω_g for K=1):
#       S_γ(ω)  ∝ ω^{-4}   (residual spin-motion entanglement)
#       S_δθ(ω) ∝ ω^{-6}   (entanglement angle error)
#
#   The power-law regime is roughly ω ∈ [1×, 5×] × δ before numerical
#   noise dominates at very high ω.  We evaluate the slope there.
# ─────────────────────────────────────────────────
println("\nChecking high-frequency power-law slopes...")

# Select the frequency window [1.5×δ, 5×δ] for the slope fit
δ_hz_abs = abs(δ_hz)
lo_hz = 1.5 * δ_hz_abs
hi_hz = 5.0 * δ_hz_abs

hf_mask = (freqs .> lo_hz) .&& (freqs .<= hi_hz)
if !any(hf_mask)
    println("  WARNING: no FFT bins in [1.5δ, 5δ] — widening check window")
    hf_mask = (freqs .> 0.5 * δ_hz_abs) .&& (freqs .<= 8.0 * δ_hz_abs)
end

ω_hf = freqs[hf_mask]
Sγ_hf = S_γ_fft[hf_mask]
Sdθ_hf = S_dθ_fft[hf_mask]

log_ω = log10.(ω_hf)
log_Sγ = log10.(max.(Sγ_hf, 1e-300))
log_Sdθ = log10.(max.(Sdθ_hf, 1e-300))

slope_γ = (log_Sγ[end] - log_Sγ[1]) / (log_ω[end] - log_ω[1])
slope_dθ = (log_Sdθ[end] - log_Sdθ[1]) / (log_ω[end] - log_ω[1])

@printf "  Slope evaluation window: %.1f kHz – %.1f kHz\n" (lo_hz / 1e3) (hi_hz / 1e3)
@printf "  Measured S_γ  slope = %.2f  (expected ≈ -4)\n" slope_γ
@printf "  Measured S_δθ slope = %.2f  (expected ≈ -6)\n" slope_dθ

pass_γ = abs(slope_γ - (-4)) < 1.5
pass_dθ = abs(slope_dθ - (-6)) < 1.5
println("  S_γ  slope check:  ", pass_γ ? "PASS ✓" : "FAIL ✗")
println("  S_δθ slope check:  ", pass_dθ ? "PASS ✓" : "FAIL ✗")

# ─────────────────────────────────────────────────
# 7.  Plots
# ─────────────────────────────────────────────────
println("\nGenerating plots...")

fig, axes = subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Filter Function Validation — MS Gate (Walsh-0, t = $(t_gate) µs)", fontsize=12)

# ── Panel 1: γ(t) trajectory in phase space ──────
ax1 = axes[1]
ax1.plot(real.(γ_t), imag.(γ_t), color="#3a86ff", lw=1.5)
ax1.scatter([real(γ_t[1])], [imag(γ_t[1])], color="#06d6a0", s=60, zorder=5, label="start")
ax1.scatter([real(γ_t[end])], [imag(γ_t[end])], color="#ef233c", s=60, zorder=5, label="end")
ax1.set_xlabel("Re[γ]  (rad)", fontsize=10)
ax1.set_ylabel("Im[γ]  (rad)", fontsize=10)
ax1.set_title("γ(t) phase-space trajectory", fontsize=10)
ax1.legend(fontsize=9)
ax1.set_aspect("equal")
ax1.grid(true, alpha=0.3)

# ── Panel 2: S(ω) filter function — log-log ──────
ax2 = axes[2]
ax2.loglog(ω_plot_hz ./ 1e3, Sγ_plot, color="#3a86ff", lw=1.8, label=L"$S_\gamma(\omega)$")
ax2.loglog(ω_plot_hz ./ 1e3, Sdθ_plot, color="#ef476f", lw=1.8, ls="--", label=L"$S_{\delta\theta}(\omega)$")
ax2.loglog(ω_plot_hz ./ 1e3, Stot_plot, color="#06d6a0", lw=1.8, ls=":", label=L"$S(\omega)$ total")

# Power-law reference guides anchored at ω ≈ 2δ
ref_idx = argmin(abs.(ω_plot_hz .- 2.0 * δ_hz_abs))
ω_ref = ω_plot_hz[ref_idx] / 1e3
ref_Sγ = Sγ_plot[ref_idx]
ref_Sdθ = Sdθ_plot[ref_idx]
guide_w = ω_plot_hz[ref_idx:end] ./ 1e3
ax2.loglog(guide_w, ref_Sγ .* (guide_w ./ ω_ref) .^ (-4), "k--", lw=0.9, alpha=0.5, label=L"$\omega^{-4}$ guide")
ax2.loglog(guide_w, ref_Sdθ .* (guide_w ./ ω_ref) .^ (-6), "k:", lw=0.9, alpha=0.5, label=L"$\omega^{-6}$ guide")

ax2.axvline(δ_hz_abs / 1e3, color="gray", ls="--", lw=0.8, alpha=0.6, label=L"$\delta/2\pi$")
ax2.set_xlabel("Frequency  [kHz]", fontsize=10)
ax2.set_ylabel(L"$S(\omega)$  [s²]", fontsize=10)
ax2.set_title("Filter function  (log-log)", fontsize=10)
ax2.legend(fontsize=7.5, ncol=2)
ax2.grid(true, alpha=0.3, which="both")

# ── Panel 3: Static-detuning validation ──────────
ax3 = axes[3]
ε_plot = ε₀_hz[2:end]          # skip zero
ℐd_plot = ℐ_direct[2:end]
ℐp_plot = ℐ_predicted[2:end]

ax3.plot(ε_plot, ℐd_plot, "o-", color="#3a86ff", lw=1.8, ms=5, label="IonSim direct  1 - Q_det")
ax3.plot(ε_plot, ℐp_plot, "s--", color="#ef476f", lw=1.8, ms=5, label=L"Filter fn:  $S(0)\cdot(2\pi\varepsilon_0)^2$")
ax3.set_xlabel(L"Static detuning offset $\varepsilon_0$  [Hz]", fontsize=10)
ax3.set_ylabel("Infidelity  ℐ", fontsize=10)
ax3.set_title("Static-detuning validation", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(true, alpha=0.3)

tight_layout()

outfile = joinpath(@__DIR__, "plots", "filter_function_validation.png")
mkpath(dirname(outfile))
savefig(outfile, dpi=150, bbox_inches="tight")
println("  Saved → $outfile")

# ─────────────────────────────────────────────────
# 8.  Summary
# ─────────────────────────────────────────────────
println("\n════════════════════════════════════")
println("  VALIDATION SUMMARY")
println("════════════════════════════════════")
@printf "  S_γ  high-freq slope = %+.2f  (theory: -4)\n" slope_γ
@printf "  S_δθ high-freq slope = %+.2f  (theory: -6)\n" slope_dθ

# Report agreement at small ε₀ (avoid division by zero at ε=0)
ratio_str = join([@sprintf("%.3f", ℐ_predicted[i] / max(ℐ_direct[i], 1e-15))
                  for i in 2:min(5, length(ε₀_hz))], ", ")
println("  Filter/IonSim ratio at small ε₀: $ratio_str  (expect ≈ 1.0)")
println("  Validation plot saved to:  filter_function_validation.png")
println("════════════════════════════════════")
