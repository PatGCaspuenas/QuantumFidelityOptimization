# evaluate_ionsim.jl — IonSim Monte Carlo noise evaluation
#
# Full Hamiltonian propagation + 1/f noise sampling.
# Metric: ΔF = F_ideal − ⟨F_noisy⟩  (lower = more noise-robust)
#
# Run:
#   julia --project=. evaluate_ionsim.jl              # full MC evaluation
#   julia --project=. evaluate_ionsim.jl --ideal-only  # ideal only (fast check)

# ── Project + IonSim (local dev copy; see Project.toml [sources]) ────────────
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using IonSim
using IonSim: timeevolution
using Printf
using LinearAlgebra

# Vern7 solver (transitive dep of IonSim, access via loaded modules)
const _VernMod = Base.loaded_modules[Base.PkgId(
    Base.UUID("79d7bb75-1356-48c1-b8c0-6832512096c2"), "OrdinaryDiffEqVerner")]
const _Vern7 = _VernMod.Vern7

ENV["MPLBACKEND"] = "Agg"
using PyPlot

include(joinpath(@__DIR__, "pulse.jl"))

# ── Physical system ──────────────────────────────────────────────────────────

const N_FOCK = 10
const ω_AX_HZ = 1.0e6
const ω_COM_HZ = sqrt(3.3e6^2 + ω_AX_HZ^2)  # ≈ 3.4482 MHz → rocking mode at 3.3 MHz (Hughes geometry)
const ω_RAD_Y_HZ = ω_COM_HZ
const B_FIELD_T = 4e-4

function build_system()
    ion1 = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])
    ion2 = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])

    chain = LinearChain(
        ions=[ion1, ion2],
        comfrequencies=(x=ω_COM_HZ, y=ω_RAD_Y_HZ, z=ω_AX_HZ),
        selectedmodes=(x=[1, 2],)
    )
    for m in xmodes(chain)
        modecutoff!(m, N_FOCK)
    end

    L_blue = Laser()
    L_red  = Laser()
    s2 = 1 / sqrt(2)
    chamber = Chamber(iontrap=chain, B=B_FIELD_T,
                      Bhat=(x=s2, y=0.0, z=s2), lasers=[L_blue, L_red])

    wavelength_from_transition!(L_blue, ion1, ("S", "D"), chamber)
    wavelength_from_transition!(L_red,  ion1, ("S", "D"), chamber)
    wavevector!(L_blue, x̂)
    wavevector!(L_red,  x̂)
    polarization!(L_blue, ẑ)
    polarization!(L_red,  ẑ)
    pointing!(L_blue, [(1, 1.0), (2, 1.0)])
    pointing!(L_red,  [(1, 1.0), (2, 1.0)])

    return chamber, ion1, ion2, chain, L_blue, L_red
end

# ── Parameter mapping ────────────────────────────────────────────────────────
#
# pulse.jl uses rad/µs for delta and Omega_amp (evaluate.jl convention).
# IonSim uses Hz for detuning and carrier Rabi frequency.
#
# Conversion (corrected):
#   Ω_carrier_Hz = Ω_amp / (4·π·η·1e-6)
#   Derivation: Ω_g = 2η·Ω_carrier (bichromatic MS, 2 ions on COM)
#   evaluate.jl: Ω_amp = Ω_g, so Ω_carrier = Ω_amp/(2η) → in Hz: Ω_amp/(4πη·1e-6)
#
# IonSim limitation: Laser.Δ is Real (static only).
# Time-dependent δ(t) is modeled via the phase function: φ(t) = ∫₀ᵗ δ(τ) dτ.
# Time-dependent Ω(t) is modeled via intensity scaling: I(t) = I_peak·(Ω(t)/Ω_peak)².

"""
    build_detuning_phase(delta_func, t_gate_us; N=4000)

Precompute the accumulated detuning phase η(t) = ∫₀ᵗ δ(τ) dτ [radians].
Returns an interpolating function of t (µs).

IonSim's Laser.Δ only accepts a static Real value. Time-dependent sideband
detuning δ(t) must be encoded via the laser phase: setting φ(t) = ±η(t)
shifts the effective laser frequency by ±δ(t) relative to the sideband.
"""
function build_detuning_phase(delta_func, t_gate_us::Float64; N::Int=4000)
    dt = t_gate_us / N
    acc = zeros(Float64, N + 1)
    @inbounds for i in 1:N
        t = (i - 1) * dt
        acc[i + 1] = acc[i] + delta_func(t) * dt
    end
    return t -> begin
        idx = t / dt + 1
        i1 = clamp(floor(Int, idx), 1, N)
        i2 = min(i1 + 1, N + 1)
        frac = idx - i1
        (1 - frac) * acc[i1] + frac * acc[i2]
    end
end

function configure_lasers!(chamber, p::PulseDefinition, L_blue, L_red, ion1)
    chain = iontrap(chamber)
    modes = xmodes(chain)
    com_mode = modes[1]

    η = lambdicke(com_mode, ion1, L_blue)
    ω_com = frequency(com_mode)

    # ── Detuning: lasers on sidebands (Δ is static in IonSim) ──
    detuning!(L_blue, ω_com)
    detuning!(L_red, -ω_com)

    # ── Precompute phase and amplitude profiles into type-stable lerps ──
    N_pre = 4000
    dt_pre = p.t_gate_us / N_pre

    # Detuning phase: ∫₀ᵗ δ(τ) dτ
    δ_acc = zeros(Float64, N_pre + 1)
    @inbounds for i in 1:N_pre
        t = (i - 1) * dt_pre
        δ_acc[i + 1] = δ_acc[i] + p.delta(t) * dt_pre
    end

    # Combined phase for blue laser; negative detuning phase for red
    phi_vals = Float64[p.phi((i - 1) * dt_pre) for i in 1:(N_pre + 1)]
    phase!(L_blue, make_lerp(δ_acc .+ phi_vals, dt_pre))
    phase!(L_red,  make_lerp(.-δ_acc, dt_pre))

    # ── Amplitude: carrier Rabi from peak Omega_amp ──
    Ω_vals = Float64[abs(p.Omega_amp((i - 1) * dt_pre)) for i in 1:(N_pre + 1)]
    Ω_peak = maximum(Ω_vals)
    Ω_carrier_Hz = Ω_peak / (4 * π * η * 1e-6)

    intensity_from_rabifrequency!(L_blue, Ω_carrier_Hz, ion1, ("S", "D"), chamber)
    intensity_from_rabifrequency!(L_red,  Ω_carrier_Hz, ion1, ("S", "D"), chamber)

    # Time-dependent Ω(t) via intensity scaling: I ∝ Ω²
    I_cal = L_blue.I(0.0)
    I_vals = Float64[I_cal * (Ω_vals[i] / Ω_peak)^2 for i in eachindex(Ω_vals)]
    I_fn = make_lerp(I_vals, dt_pre)
    L_blue.I = I_fn
    L_red.I  = I_fn

    # Diagnostics
    println("  η (COM, ion1)      = $η")
    println("  ω_COM              = $(ω_com / 1e6) MHz")
    println("  ω_mode2            = $(frequency(modes[2]) / 1e6) MHz")
    for (i, m) in enumerate(modes)
        η1 = lambdicke(m, ion1, L_blue)
        η2 = lambdicke(m, ions(chain)[2], L_blue)
        println("  η mode$i: ion1=$η1, ion2=$η2")
    end
    @printf "  Ω_carrier (peak)   = %.1f kHz\n" (Ω_carrier_Hz / 1e3)
    @printf "  δ(0)               = %.1f kHz\n" (p.delta(0.0) / (2π * 1e-6) / 1e3)
    @printf "  δ(T/2)             = %.1f kHz\n" (p.delta(p.t_gate_us/2) / (2π * 1e-6) / 1e3)
end

# ── V2: Independent red/blue laser configuration ────────────────────────────

function configure_lasers!(chamber, p::PulseDefinitionV2, L_blue, L_red, ion1)
    chain = iontrap(chamber)
    modes = xmodes(chain)
    target_idx = p.mode_target === :rocking ? 2 : 1
    target_mode = modes[target_idx]

    η = lambdicke(target_mode, ion1, L_blue)
    ω_mode = frequency(target_mode)

    detuning!(L_blue, ω_mode)
    detuning!(L_red, -ω_mode)

    N_pre = 4000
    dt_pre = p.t_gate_us / N_pre

    # ── Blue laser: independent δ, Ω, φ ──
    δ_acc_b = zeros(Float64, N_pre + 1)
    @inbounds for i in 1:N_pre
        t = (i - 1) * dt_pre
        δ_acc_b[i + 1] = δ_acc_b[i] + p.delta_blue(t) * dt_pre
    end
    phi_b = Float64[p.phi_blue((i - 1) * dt_pre) for i in 1:(N_pre + 1)]
    phase!(L_blue, make_lerp(δ_acc_b .+ phi_b, dt_pre))

    Ω_b = Float64[abs(p.Omega_blue((i - 1) * dt_pre)) for i in 1:(N_pre + 1)]
    Ω_peak_b = maximum(Ω_b)
    if Ω_peak_b > 0
        Ω_car_b = Ω_peak_b / (4π * η * 1e-6)
        intensity_from_rabifrequency!(L_blue, Ω_car_b, ion1, ("S", "D"), chamber)
        I_cal_b = L_blue.I(0.0)
        L_blue.I = make_lerp(Float64[I_cal_b * (Ω_b[i] / Ω_peak_b)^2 for i in eachindex(Ω_b)], dt_pre)
    end

    # ── Red laser: independent δ, Ω, φ ──
    δ_acc_r = zeros(Float64, N_pre + 1)
    @inbounds for i in 1:N_pre
        t = (i - 1) * dt_pre
        δ_acc_r[i + 1] = δ_acc_r[i] + p.delta_red(t) * dt_pre
    end
    phi_r = Float64[p.phi_red((i - 1) * dt_pre) for i in 1:(N_pre + 1)]
    phase!(L_red, make_lerp(.-δ_acc_r .+ phi_r, dt_pre))

    Ω_r = Float64[abs(p.Omega_red((i - 1) * dt_pre)) for i in 1:(N_pre + 1)]
    Ω_peak_r = maximum(Ω_r)
    if Ω_peak_r > 0
        Ω_car_r = Ω_peak_r / (4π * η * 1e-6)
        intensity_from_rabifrequency!(L_red, Ω_car_r, ion1, ("S", "D"), chamber)
        I_cal_r = L_red.I(0.0)
        L_red.I = make_lerp(Float64[I_cal_r * (Ω_r[i] / Ω_peak_r)^2 for i in eachindex(Ω_r)], dt_pre)
    end

    # Diagnostics
    mode_name = p.mode_target === :rocking ? "rocking" : "COM"
    println("  Mode target        = $mode_name ($(round(ω_mode/1e6, digits=3)) MHz)")
    println("  η ($mode_name, ion1) = $η")
    Ω_peak_b > 0 && @printf("  Ω_carrier blue     = %.1f kHz\n", Ω_peak_b / (4π * η * 1e-6) / 1e3)
    Ω_peak_r > 0 && @printf("  Ω_carrier red      = %.1f kHz\n", Ω_peak_r / (4π * η * 1e-6) / 1e3)
    @printf "  δ_blue(0)          = %.1f kHz\n" (p.delta_blue(0.0) / (2π * 1e-6) / 1e3)
    @printf "  δ_red(0)           = %.1f kHz\n" (p.delta_red(0.0) / (2π * 1e-6) / 1e3)
    @printf "  δ_blue(T/2)        = %.1f kHz\n" (p.delta_blue(p.t_gate_us / 2) / (2π * 1e-6) / 1e3)
    @printf "  δ_red(T/2)         = %.1f kHz\n" (p.delta_red(p.t_gate_us / 2) / (2π * 1e-6) / 1e3)
end

function configure_noisy!(chamber, p::PulseDefinitionV2, Lb, Lr, ion1,
                          δn_radus::Vector{Float64}, εn::Vector{Float64})
    chain = iontrap(chamber)
    modes = xmodes(chain)
    target_idx = p.mode_target === :rocking ? 2 : 1
    η = lambdicke(modes[target_idx], ion1, Lb)
    ω_mode = frequency(modes[target_idx])
    T = p.t_gate_us

    detuning!(Lb, ω_mode)
    detuning!(Lr, -ω_mode)

    Nn = length(δn_radus)
    dt = T / Nn

    # Blue phase: ∫(δ_blue + noise)dt + φ_blue
    φ_b = zeros(Float64, Nn + 1)
    @inbounds for i in 1:Nn
        t = (i - 1) * dt
        φ_b[i + 1] = φ_b[i] + (p.delta_blue(t) + δn_radus[i]) * dt
    end
    phi_b_vals = Float64[p.phi_blue(clamp((i - 1) * dt, 0.0, T)) for i in 1:(Nn + 1)]
    phase!(Lb, make_lerp(φ_b .+ phi_b_vals, dt))

    # Red phase: -∫(δ_red + noise)dt + φ_red
    φ_r = zeros(Float64, Nn + 1)
    @inbounds for i in 1:Nn
        t = (i - 1) * dt
        φ_r[i + 1] = φ_r[i] + (p.delta_red(t) + δn_radus[i]) * dt
    end
    phi_r_vals = Float64[p.phi_red(clamp((i - 1) * dt, 0.0, T)) for i in 1:(Nn + 1)]
    phase!(Lr, make_lerp(.-φ_r .+ phi_r_vals, dt))

    # Blue amplitude with noise
    Ω_b = Float64[abs(p.Omega_blue(clamp((i - 1) * dt, 0.0, T))) * (1.0 + εn[i]) for i in 1:Nn]
    Ω_peak_b = max(maximum(abs, Ω_b), 1e-12)
    Ω_car_b = Ω_peak_b / (4π * η * 1e-6)
    intensity_from_rabifrequency!(Lb, Ω_car_b, ion1, ("S", "D"), chamber)
    I_cal_b = Lb.I(0.0)
    Lb.I = make_lerp(Float64[I_cal_b * (Ω_b[i] / Ω_peak_b)^2 for i in 1:Nn], dt)

    # Red amplitude with noise
    Ω_r = Float64[abs(p.Omega_red(clamp((i - 1) * dt, 0.0, T))) * (1.0 + εn[i]) for i in 1:Nn]
    Ω_peak_r = max(maximum(abs, Ω_r), 1e-12)
    Ω_car_r = Ω_peak_r / (4π * η * 1e-6)
    intensity_from_rabifrequency!(Lr, Ω_car_r, ion1, ("S", "D"), chamber)
    I_cal_r = Lr.I(0.0)
    Lr.I = make_lerp(Float64[I_cal_r * (Ω_r[i] / Ω_peak_r)^2 for i in 1:Nn], dt)
end

# ── States ───────────────────────────────────────────────────────────────────

function build_initial_state(chamber)
    chain = iontrap(chamber)
    modes = xmodes(chain)
    return iontensor(ionstate(chamber, ["S", "S"]),
                     fockstate(modes[1], 0), fockstate(modes[2], 0))
end

function build_bell_projector(chamber)
    chain = iontrap(chamber)
    ion_list = ions(chain)
    modes = xmodes(chain)
    # Bell state: (|SS⟩ + i|DD⟩)/√2 ⊗ 𝟙_mode1 ⊗ 𝟙_mode2
    ψ_bell_spin = (iontensor(ion_list[1]["S"], ion_list[2]["S"]) +
                   1im * iontensor(ion_list[1]["D"], ion_list[2]["D"])) / √2
    return iontensor(dm(ψ_bell_spin), one(modes[1]), one(modes[2]))
end

const BELL_PHASES = (plus_i = 1im, minus_i = -1im,
                     plus_1 = 1.0+0im, minus_1 = -1.0+0im)
const BELL_LABELS = Dict(:plus_i => "|SS⟩+i|DD⟩", :minus_i => "|SS⟩-i|DD⟩",
                         :plus_1 => "|SS⟩+|DD⟩",  :minus_1 => "|SS⟩-|DD⟩")

function build_bell_projectors(chamber)
    chain = iontrap(chamber)
    ion_list = ions(chain)
    modes = xmodes(chain)
    projs = Dict{Symbol, Any}()
    for (name, ϕ) in pairs(BELL_PHASES)
        ψ = (iontensor(ion_list[1]["S"], ion_list[2]["S"]) +
             ϕ * iontensor(ion_list[1]["D"], ion_list[2]["D"])) / √2
        projs[name] = iontensor(dm(ψ), one(modes[1]), one(modes[2]))
    end
    return projs
end

# ── Propagate ────────────────────────────────────────────────────────────────

function propagate(chamber, p; n_timepoints::Int=4000)
    h = hamiltonian(chamber; timescale=1e-6, lamb_dicke_order=1,
                    rwa_cutoff=Inf, time_dependent_eta=false)

    ψ0 = build_initial_state(chamber)
    tspan = collect(range(0, p.t_gate_us, length=n_timepoints))

    println("  Propagating $(p.t_gate_us) µs  ($(n_timepoints) steps)...")
    sol = timeevolution.schroedinger_dynamic(tspan, ψ0, h; alg=_Vern7())
    return sol.times, sol.states
end

# ── Analysis ─────────────────────────────────────────────────────────────────

function analyze(chamber, tout, ψ_t)
    chain = iontrap(chamber)
    modes = xmodes(chain)

    SS = ionprojector(chamber, "S", "S")
    DD = ionprojector(chamber, "D", "D")
    SD = ionprojector(chamber, "S", "D")
    DS = ionprojector(chamber, "D", "S")

    P_SS   = real.(expect(SS, ψ_t))
    P_DD   = real.(expect(DD, ψ_t))
    P_SD   = real.(expect(SD, ψ_t))
    P_DS   = real.(expect(DS, ψ_t))

    # Compute fidelity for all 4 canonical Bell states, pick the best
    projs = build_bell_projectors(chamber)
    F_all = Dict(k => real.(expect(v, ψ_t)) for (k, v) in projs)
    F_finals = Dict(k => v[end] for (k, v) in F_all)
    best_bell = argmax(F_finals)
    F_bell = F_all[best_bell]

    b = IonSim.basis(chamber)
    n_com = real(expect(IonSim._embed(b, [3], [IonSim.number(modes[1])]), ψ_t[end]))
    n_str = real(expect(IonSim._embed(b, [4], [IonSim.number(modes[2])]), ψ_t[end]))

    return (; P_SS, P_DD, P_SD, P_DS, F_bell, n_com, n_str, best_bell)
end

# ── evaluate_ionsim.jl specific ──────────────────────────────────────────────

using Distributed
using Statistics: mean, std
using Random
import TOML
using Dates
using LaTeXStrings

# ── Configuration ────────────────────────────────────────────────────────────
const N_WORKERS    = 3        # Distributed workers (capped for thermal throttling)
const MC_RUNS      = 24        # noise realizations
const FOCK_CUTOFF  = 8        # Fock dim (8 ≈ 2× faster than 10). If you wanted to do simulations with higher n^bar you'd need to expand the fock dim
const PTS_IDEAL    = 2000     # output time points for ideal run
const PTS_NOISY    = 1000     # output time points per noisy run

# 1/f noise (Voss-McCartney: inherently 1/f)
const δ_RMS_HZ    = 300.0    # detuning noise RMS (Hz) — typical Ca-40 magnetic noise
const Ω_RMS_FRAC  = 0.007    # amplitude noise RMS (fractional ΔΩ/Ω, 0.3%) — typical AOM

# Multi-mode (from evaluate.jl TWO_ION_CA40 config)
const MOTIONAL_OFFSETS = [0.0, 0.9749153786790529]  # rad/µs: COM, rocking

# Performance: set to e.g. 1e6 (i think) to drop carrier terms for a speed increase. Could be worth doing for MC
const RWA_CUTOFF   = Inf

# ── Colored 1/f noise (Voss-McCartney octave-band method) ───────────────────

"""Generate approximate 1/f noise scaled to `target_rms`. N-point Vector{Float64}."""
function gen_noise(N::Int, target_rms::Float64; n_oct::Int=14)
    x = zeros(N)
    for oct in 0:n_oct-1
        period = 1 << oct
        nv = cld(N, period) + 1
        v = randn(nv)
        @inbounds for i in 1:N
            pos = (i - 1) / period
            j = min(floor(Int, pos) + 1, nv - 1)
            f = pos - (j - 1)
            x[i] += v[j] + f * (v[j + 1] - v[j])
        end
    end
    s = sqrt(sum(abs2, x) / N)
    s > 0 && (x .*= target_rms / s)
    return x
end

# ── Interpolation ────────────────────────────────────────────────────────────

"""Linear-interpolating closure for `vals[1..N]` at times `0, dt, 2dt, ...`."""
function make_lerp(vals::Vector{Float64}, dt::Float64)
    N = length(vals)
    tmax = (N - 1) * dt
    return function(t::Float64)
        tc = clamp(t, 0.0, tmax)
        idx = tc / dt
        i = min(floor(Int, idx) + 1, N - 1)
        f = idx - (i - 1)
        @inbounds vals[i] + f * (vals[i + 1] - vals[i])
    end
end

# ── Noisy laser configuration ───────────────────────────────────────────────

"""Configure lasers with noise-perturbed detuning and amplitude."""
function configure_noisy!(chamber, p::PulseDefinition, Lb, Lr, ion1,
                          δn_radus::Vector{Float64}, εn::Vector{Float64})
    chain = iontrap(chamber)
    modes = xmodes(chain)
    η = lambdicke(modes[1], ion1, Lb)
    ω_com = frequency(modes[1])
    T = p.t_gate_us

    detuning!(Lb, ω_com)
    detuning!(Lr, -ω_com)

    Nn = length(δn_radus)
    dt = T / Nn
    φ = zeros(Float64, Nn + 1)
    @inbounds for i in 1:Nn
        t = (i - 1) * dt
        φ[i + 1] = φ[i] + (p.delta(t) + δn_radus[i]) * dt
    end
    # Precompute phi values and combine with detuning phase
    phi_vals = Float64[p.phi(clamp((i - 1) * dt, 0.0, T)) for i in 1:(Nn + 1)]
    phase!(Lb, make_lerp(φ .+ phi_vals, dt))
    phase!(Lr, make_lerp(.-φ, dt))

    # Precompute amplitude × noise profile into type-stable lerp
    Ω_base = Float64[abs(p.Omega_amp(clamp((i - 1) * dt, 0.0, T))) for i in 1:Nn]
    Ω_noisy = Float64[Ω_base[i] * (1.0 + εn[i]) for i in 1:Nn]
    Ω_peak = maximum(abs, Ω_noisy)
    Ω_peak = max(Ω_peak, 1e-12)

    Ω_car = Ω_peak / (4π * η * 1e-6)
    intensity_from_rabifrequency!(Lb, Ω_car, ion1, ("S", "D"), chamber)
    intensity_from_rabifrequency!(Lr, Ω_car, ion1, ("S", "D"), chamber)
    I_cal = Lb.I(0.0)
    I_vals = Float64[I_cal * (Ω_noisy[i] / Ω_peak)^2 for i in 1:Nn]
    I_fn = make_lerp(I_vals, dt)
    Lb.I = I_fn
    Lr.I = I_fn
end

# ── Semiclassical phase space (matches evaluate.jl) ────────────────────────

"""Compute semiclassical γ(t) trajectories for each motional mode."""
function compute_trajectories(p; N_pts::Int=4000)
    dt = p.t_gate_us / N_pts
    n_modes = length(MOTIONAL_OFFSETS)
    γ_modes = [zeros(ComplexF64, N_pts) for _ in 1:n_modes]
    γ = zeros(ComplexF64, n_modes)
    η = zeros(Float64, n_modes)
    times = zeros(Float64, N_pts)

    @inbounds for i in 1:N_pts
        t = (i - 1) * dt
        times[i] = t
        Ω = p.Omega_amp(t)
        ϕ = p.phi(t)
        δ = p.delta(t)
        for k in 1:n_modes
            γ_modes[k][i] = γ[k]
            γ[k] += 0.5im * Ω * exp(1im * (η[k] + ϕ)) * dt
            η[k] += (δ - MOTIONAL_OFFSETS[k]) * dt
        end
    end
    return times, γ_modes, copy(γ)
end

# ── Quiet propagation (no print) ────────────────────────────────────────────

function propagate_q(chamber, p; n_pts::Int=PTS_NOISY)
    h = hamiltonian(chamber; timescale=1e-6, lamb_dicke_order=1,
                    rwa_cutoff=RWA_CUTOFF, time_dependent_eta=false)
    ψ0 = build_initial_state(chamber)
    tspan = collect(range(0.0, p.t_gate_us, length=n_pts))
    sol = timeevolution.schroedinger_dynamic(tspan, ψ0, h; alg=_Vern7())
    return sol.times, sol.states
end

# ── Distributed MC ──────────────────────────────────────────────────────────

function setup_workers()
    n_add = min(N_WORKERS, Sys.CPU_THREADS - 1) - (nprocs() - 1)
    n_add > 0 && addprocs(n_add; exeflags="--project=$(Base.active_project())")
    # Load code on workers only (main already has everything)
    @everywhere workers() include(joinpath(@__DIR__, "evaluate_ionsim.jl"))
    println("  Workers: $(nprocs()-1) active")
end

"""Run one noisy propagation. Returns fidelity for the best Bell state."""
function run_one_noisy(δn_radus::Vector{Float64}, εn::Vector{Float64};
                       bell_target::Symbol=:plus_i)
    chamber, ion1, _, chain, Lb, Lr = build_system()
    for m in xmodes(chain); modecutoff!(m, FOCK_CUTOFF); end
    p = pulse()
    configure_noisy!(chamber, p, Lb, Lr, ion1, δn_radus, εn)
    tout, ψ_t = propagate_q(chamber, p; n_pts=PTS_NOISY)
    projs = build_bell_projectors(chamber)
    F_t = real.(expect(projs[bell_target], ψ_t))
    return (tout, F_t)
end

function run_mc(F_ideal; n_mc=MC_RUNS, bell_target::Symbol=:plus_i)
    # Generate noise on main process
    # Static offset noise: each MC run samples a single constant offset
    # from a Gaussian distribution (quasi-static noise model)
    noise_pairs = [(fill(randn() * δ_RMS_HZ * (2π * 1e-6), PTS_NOISY),
                     fill(randn() * Ω_RMS_FRAC, PTS_NOISY)) for _ in 1:n_mc]

    bt = bell_target  # capture for closure
    results = pmap(np -> run_one_noisy(np[1], np[2]; bell_target=bt), noise_pairs)

    F_finals = [r[2][end] for r in results]
    for (i, f) in enumerate(F_finals)
        @printf "  MC %d/%d: F=%.6f\n" i n_mc f
    end

    Fm = mean(F_finals)
    σ = n_mc > 1 ? std(F_finals) : 0.0
    mc_trajectories = [(r[1], r[2]) for r in results]  # (tout, F_bell_t) per run
    return (; F_noisy=F_finals, F_mean=Fm, ΔF=F_ideal - Fm, σ, se=σ/sqrt(n_mc),
              trajectories=mc_trajectories)
end

# ── Plots ────────────────────────────────────────────────────────────────────

function plot_results(tout, r, γ_modes, γ_final, mc, F_ideal, p; ideal_only=false)
    dir = joinpath(@__DIR__, "results")

    # 1 — Populations (left) + MC F_bell trajectories (right)
    if ideal_only
        fig, ax = subplots(figsize=(8, 5))
        ax.plot(tout, r.P_SS; lw=1.4, color="#2563eb", label="SS")
        ax.plot(tout, r.P_DD; lw=1.4, color="#dc2626", label="DD")
        ax.plot(tout, r.P_SD; lw=1.1, color="#94a3b8", ls="--", label="SD")
        ax.plot(tout, r.P_DS; lw=1.1, color="#94a3b8", ls=":", label="DS")
        ax.plot(tout, r.F_bell; lw=1.8, color="#16a34a", label=L"F_\mathrm{Bell}")
        ax.set_xlim(tout[1], tout[end]); ax.set_ylim(0, 1)
        ax.legend(loc="right", fontsize=9); ax.grid(true; alpha=0.2)
        ax.set_xlabel(L"t\;(\mu\mathrm{s})"); ax.set_ylabel("Population / Fidelity")
        ax.set_title("IonSim: $(p.description)")
    else
        fig, axes = subplots(1, 2; figsize=(14, 5))
        # Left: ideal populations
        ax = axes[1]
        ax.plot(tout, r.P_SS; lw=1.4, color="#2563eb", label="SS")
        ax.plot(tout, r.P_DD; lw=1.4, color="#dc2626", label="DD")
        ax.plot(tout, r.P_SD; lw=1.1, color="#94a3b8", ls="--", label="SD")
        ax.plot(tout, r.P_DS; lw=1.1, color="#94a3b8", ls=":", label="DS")
        ax.plot(tout, r.F_bell; lw=1.8, color="#16a34a", label=L"F_\mathrm{Bell}")
        ax.set_xlim(tout[1], tout[end]); ax.set_ylim(0, 1)
        ax.legend(loc="right", fontsize=8); ax.grid(true; alpha=0.2)
        ax.set_xlabel(L"t\;(\mu\mathrm{s})"); ax.set_ylabel("Population / Fidelity")
        ax.set_title("Ideal")
        # Right: F_bell(t) for each MC run vs ideal
        ax2 = axes[2]
        mc_colors = plt.cm.tab10(LinRange(0, 1, length(mc.trajectories)))
        for (i, (t_mc, F_mc)) in enumerate(mc.trajectories)
            ax2.plot(t_mc, F_mc; lw=0.7, color=mc_colors[i, :], alpha=0.5)
        end
        ax2.plot(tout, r.F_bell; lw=2.0, color="black", label="Ideal", zorder=10)
        ax2.set_xlim(tout[1], tout[end]); ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8); ax2.grid(true; alpha=0.2)
        ax2.set_xlabel(L"t\;(\mu\mathrm{s})"); ax2.set_ylabel(L"F_\mathrm{Bell}")
        ax2.set_title(@sprintf("MC noise (N=%d, DF=%.2e)", length(mc.F_noisy), mc.ΔF))
    end
    fig.tight_layout()
    savefig(joinpath(dir, "ionsim_populations.png"); dpi=150, bbox_inches="tight")
    close(fig)

    # 2 — Phase space (semiclassical γ(t), same as evaluate.jl)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    fig, ax = subplots(figsize=(5.5, 4.5))
    for k in eachindex(MOTIONAL_OFFSETS)
        c = colors[mod1(k, length(colors))]
        lbl = k == 1 ? "mode 1 (gate)" : "mode $k"
        γ_plot = vcat(γ_modes[k], γ_final[k])
        ax.plot(real.(γ_plot), imag.(γ_plot); lw=1.35, color=c, label=lbl)
    end
    ax.set_aspect("equal"); ax.grid(true; alpha=0.2)
    ax.set_xlabel(L"\mathrm{Re}[\gamma]"); ax.set_ylabel(L"\mathrm{Im}[\gamma]")
    if length(MOTIONAL_OFFSETS) > 1; ax.legend(fontsize=8); end
    fig.tight_layout()
    savefig(joinpath(dir, "ionsim_phase_space.png"); dpi=150, bbox_inches="tight")
    close(fig)

    # 3 — Pulse waveform
    Nw = 500; tw = collect(LinRange(0.0, p.t_gate_us, Nw))
    khz = 1e3 / (2π)
    fig, axes = subplots(3, 1; figsize=(7, 6), sharex=true)
    axes[1].plot(tw, [p.Omega_amp(t) * khz for t in tw]; lw=1.4, color="#1d4ed8")
    axes[1].set_ylabel(L"\Omega\;(\mathrm{kHz})"); axes[1].grid(true; alpha=0.2)
    axes[1].set_title("Pulse: $(p.description)")
    axes[2].plot(tw, [p.phi(t) for t in tw]; lw=1.4, color="#059669")
    axes[2].set_ylabel(L"\phi\;(\mathrm{rad})"); axes[2].set_ylim(-0.05, π + 0.05)
    axes[2].grid(true; alpha=0.2)
    axes[3].plot(tw, [p.delta(t) * khz for t in tw]; lw=1.4, color="#e06666")
    axes[3].set_ylabel(L"\delta\;(\mathrm{kHz})"); axes[3].set_xlabel(L"t\;(\mu\mathrm{s})")
    axes[3].grid(true; alpha=0.2)
    fig.tight_layout()
    savefig(joinpath(dir, "ionsim_pulse.png"); dpi=150, bbox_inches="tight")
    close(fig)

    println("Plots saved to $dir/ionsim_*.png")
end

# ── Logging ──────────────────────────────────────────────────────────────────

function log_results(p, F_ideal, mc, n_com, n_rock; evaluator="ionsim_mc")
    dir = joinpath(@__DIR__, "results"); mkpath(dir)

    ts = Dates.format(now(UTC), dateformat"yyyy-mm-dd\THH:MM:SS") * "Z"
    open(joinpath(dir, "experiment_log.txt"), "a") do io
        println(io, "=" ^ 80)
        println(io, "time_utc: $ts")
        println(io, "evaluator: $evaluator")
        println(io, "description: $(p.description)")
        println(io, "metrics:")
        @printf io "  t_gate_us       = %.1f\n" p.t_gate_us
        @printf io "  F_bell_ideal    = %.8f\n" F_ideal
        if mc !== nothing
            @printf io "  F_bell_noisy    = %.8f +/- %.6f\n" mc.F_mean mc.σ
            @printf io "  Delta_F         = %.6e +/- %.6e\n" mc.ΔF mc.se
            @printf io "  N_mc=%d  d_rms=%.0fHz  O_rms=%.3f%%\n" length(mc.F_noisy) δ_RMS_HZ (Ω_RMS_FRAC*100)
        end
        @printf io "  n_com           = %.4f\n" n_com
        @printf io "  n_rock          = %.4f\n" n_rock
        if !isempty(p.notes)
            println(io, "notes: |")
            for ln in split(p.notes, '\n'); println(io, "  ", ln); end
        end
        println(io)
    end

    if mc !== nothing
        best_path = joinpath(dir, "best_ionsim.toml")
        prev = isfile(best_path) ? TOML.parsefile(best_path) : nothing
        if prev === nothing || mc.ΔF < get(prev, "Delta_F", Inf)
            open(best_path, "w") do io
                println(io, "description = ", repr(p.description))
                @printf io "t_gate_us = %.1f\n" p.t_gate_us
                @printf io "F_bell_ideal = %.8f\n" F_ideal
                @printf io "F_bell_noisy = %.8f\n" mc.F_mean
                @printf io "Delta_F = %.6e\n" mc.ΔF
                @printf io "Delta_F_se = %.6e\n" mc.se
                @printf io "N_mc = %d\n" length(mc.F_noisy)
            end
            println("Updated best_ionsim.toml")
        end
    end
end

# ── Main ─────────────────────────────────────────────────────────────────────

function ionsim_main()
    ideal_only = "--ideal-only" in ARGS || "-i" in ARGS

    println("== IonSim Evaluation ==\n")

    chamber, ion1, _, chain, Lb, Lr = build_system()
    for m in xmodes(chain); modecutoff!(m, FOCK_CUTOFF); end

    p = pulse()
    println("Pulse: $(p.description)")
    @printf "  T=%.1f us, Fock=%d\n" p.t_gate_us FOCK_CUTOFF

    # Ideal propagation
    println("\n-- Ideal --")
    configure_lasers!(chamber, p, Lb, Lr, ion1)
    tout, psi_t = propagate(chamber, p; n_timepoints=PTS_IDEAL)
    r = analyze(chamber, tout, psi_t)
    F_ideal = r.F_bell[end]
    bell_label = get(BELL_LABELS, r.best_bell, string(r.best_bell))
    @printf "  F_bell = %.6f  (target: %s)\n" F_ideal bell_label
    @printf "  n_com = %.4f, n_rock = %.4f\n" r.n_com r.n_str
    if r.n_com > FOCK_CUTOFF - 2
        println("  WARNING: n_com near Fock cutoff!")
    end

    # Semiclassical phase space
    _, γ_modes, γ_final = compute_trajectories(p; N_pts=PTS_IDEAL)

    mc = nothing
    if ideal_only
        println("\n  --ideal-only: skipping MC")
        mkpath(joinpath(@__DIR__, "results"))
        plot_results(tout, r, γ_modes, γ_final, mc, F_ideal, p; ideal_only=true)
        log_results(p, F_ideal, mc, r.n_com, r.n_str; evaluator="ionsim_ideal")
    else
        println("\n-- Setting up workers --")
        setup_workers()

        @printf "\n-- Monte Carlo (N=%d, d=%.0fHz, O=%.1f%%) --\n" MC_RUNS δ_RMS_HZ (Ω_RMS_FRAC*100)
        mc = run_mc(F_ideal; bell_target=r.best_bell)
        @printf "\n  <F> = %.6f +/- %.6f\n" mc.F_mean mc.σ
        @printf "  DF  = %.4e +/- %.4e\n" mc.ΔF mc.se

        rmprocs(workers())
        println("  Workers cleaned up")

        mkpath(joinpath(@__DIR__, "results"))
        plot_results(tout, r, γ_modes, γ_final, mc, F_ideal, p)
        log_results(p, F_ideal, mc, r.n_com, r.n_str)
    end

    if F_ideal > 0.9999;     println("\n  Ideal: PASS")
    else;                   println("\n  Ideal: FAIL")
    end

    #return F_ideal, mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    ionsim_main()
end
