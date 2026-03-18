# scripts/electric_field_noise_sim.jl
#
# Line plots of fidelity vs. offsets in f_cl, f_sb, and A.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

ENV["MPLBACKEND"] = "Agg"
using PyPlot
using LaTeXStrings
using Printf

redirect_stderr(devnull)

const t_gate = 100.0   # µs

println("Computing baseline...")
base = CalibrationCode.ideal(t_gate)
const f_cl0 = base.f_cl
const f_sb0 = base.f_sb
const A0 = base.A
@printf "  f_cl0 = %.2f Hz\n" f_cl0
@printf "  f_sb0 = %.4f kHz\n" (f_sb0 / 1e3)
@printf "  A0    = %.2f W/m²\n" A0
@printf "  F₀    = %.8f\n" base.fid

# Sweeps
println("Sweeping f_cl...")
# ±2 kHz in the scan = ±2 on the plot axis (2π kHz units)
eps_valsf_Hz = range(-2000.0, 2000.0; length=81)
eps_valsf_kHz = eps_valsf_Hz ./ 1e3

F_fcl = [CalibrationCode.Q_det(t_gate, f_cl0 + eps, f_sb0, A0; dt=0.1) for eps in eps_valsf_Hz]

println("Sweeping f_sb...")
F_fsb = [CalibrationCode.Q_det(t_gate, f_cl0, f_sb0 + eps, A0; dt=0.1) for eps in eps_valsf_Hz]

println("Sweeping A (via relative Rabi frequency Ω/Ω_ideal)...")
# User requested Ω/Ω_ideal from 0.75 to 1.25.
# Since Ω ∝ √A, we have A = A0 * (Ω/Ω_0)²
omega_ratios = range(0.75, 1.25; length=81)
A_vals = [A0 * (r^2) for r in omega_ratios]
F_A = [CalibrationCode.Q_det(t_gate, f_cl0, f_sb0, A_val; dt=0.1) for A_val in A_vals]

println("Plotting...")
rcParams = PyPlot.matplotlib.rcParams
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["font.size"] = 9.0
rcParams["axes.labelsize"] = 9.0
rcParams["xtick.labelsize"] = 8.0
rcParams["ytick.labelsize"] = 8.0
rcParams["axes.linewidth"] = 0.8
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.top"] = true
rcParams["ytick.right"] = true

# Do not sharey so we can see the potentially very different magnitudes of fidelity drop
fig, axes = subplots(1, 3, figsize=(8.5, 3.0))
fig.subplots_adjust(left=0.08, right=0.96, bottom=0.18, top=0.9, wspace=0.3)

C_data = "#2c6fad"

# f_cl plot
ax = axes[1]
ax.plot(eps_valsf_kHz, F_fcl, color=C_data, lw=1.6)
ax.set_xlabel(L"Carrier detuning offset ($2\pi$ kHz)")
ax.set_ylabel(L"Bell-state fidelity $\mathcal{F}$")
ax.set_title("(a) Carrier frequency", fontsize=10)
ax.grid(true, alpha=0.15, lw=0.5)
ax.set_xlim(extrema(eps_valsf_kHz))
ax.set_ylim(0.7, 1.01)

# f_sb plot
ax = axes[2]
ax.plot(eps_valsf_kHz, F_fsb, color=C_data, lw=1.6)
ax.set_xlabel(L"Sideband detuning offset ($2\pi$ kHz)")
ax.set_title("(b) Sideband frequency", fontsize=10)
ax.grid(true, alpha=0.15, lw=0.5)
ax.set_xlim(extrema(eps_valsf_kHz))
ax.set_ylim(0.7, 1.01)

# A plot
ax = axes[3]
ax.plot(omega_ratios, F_A, color=C_data, lw=1.6)
ax.set_xlabel(L"Relative Rabi frequency $\Omega / \Omega_{\mathrm{ideal}}$")
ax.set_title("(c) Laser intensity", fontsize=10)
ax.grid(true, alpha=0.15, lw=0.5)
ax.set_xlim(extrema(omega_ratios))
ax.set_ylim(0.7, 1.01)

outfile = joinpath(@__DIR__, "plots", "parameter_scans.png")
mkpath(dirname(outfile))
savefig(outfile, dpi=200, bbox_inches="tight")
println("Saved → $outfile")
