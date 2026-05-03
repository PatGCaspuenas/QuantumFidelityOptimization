# src/calibration.jl

const _SPIN_BASIS = SpinBasis(1 // 2)
const _TWO_QUBIT_BASIS = tensor(_SPIN_BASIS, _SPIN_BASIS)

"""
    bell_fidelity_phi_plus(ρ) -> Float64

Bell-state fidelity estimator for |ϕ⁺⟩ based on the SS/DD populations and the
SS↔DD coherence phase-aligned onto the real axis.

Assumes the computational basis ordering is (SS, SD, DS, DD) so that ρ[1,4]
corresponds to SS↔DD coherence.
"""
@inline function bell_fidelity_phi_plus(ρ::AbstractMatrix{<:Complex})::Float64
    ϕ = angle(ρ[1, 4])
    coh = real(exp(-1im * ϕ) * ρ[1, 4])
    return 0.5 * (real(ρ[1, 1] + ρ[4, 4]) + 2 * coh)
end

# --- QuantumOptics helpers

function _normalized_population_weights(values::NTuple{4,<:Real})
    weights = Float64[max(Float64(v), 0.0) for v in values]
    total = sum(weights)
    total > 0.0 || throw(ArgumentError("Population weights must have positive total."))
    return weights ./ total
end

function _expected_ms_even_populations(numMS::Int)
    numMS ≥ 1 || throw(ArgumentError("numMS must be positive."))
    if isodd(numMS)
        return (SS=0.5, DD=0.5)
    elseif numMS % 4 == 2
        return (SS=0.0, DD=1.0)
    end
    return (SS=1.0, DD=0.0)
end

sigma_binomial(p::Float64, N::Int)::Float64 = sqrt(max(p * (1.0 - p), 0.0) / N)

function sigma_delta(p1::Float64, p2::Float64,
                     r1::Float64, r2::Float64, N::Int)::Float64
    s1 = p1 > r1 ? 1.0 : (p1 < r1 ? -1.0 : 0.0)
    s2 = p2 > r2 ? 1.0 : (p2 < r2 ? -1.0 : 0.0)
    return sqrt(max(p1*(1.0-p1) + p2*(1.0-p2) - 2.0*s1*s2*p1*p2, 0.0) / N)
end

# --- IonSim setup (centralized to avoid repetition)

"""
    build_chamber(; B=6e-4, comfreq=(x=3e6,y=3e6,z=1e6), selected=(;x=[1]))

Create a standard 2-ion Ca40 chamber with two lasers and a single selected x-mode.
Returns a named tuple with (ca, laser1, laser2, chamber, mode).
"""
function build_chamber(; B::Float64=6e-4,
    comfreq=(x=3e6, y=3e6, z=1e6),
    selected=(; x=[1]))
    ca = Ca40([("S1/2", -1 / 2, "S"), ("D5/2", -1 / 2, "D")])
    laser1 = Laser(pointing=[(1, 1.), (2, 1.)])
    laser2 = Laser(pointing=[(1, 1.), (2, 1.)])

    chain, chamber = Logging.with_logger(Logging.NullLogger()) do
        ch = LinearChain(
            ions=[ca, ca],
            comfrequencies=comfreq,
            selectedmodes=selected,
        )
        cb = Chamber(
            iontrap=ch,
            B=B,
            Bhat=(x̂ + ẑ) / √2,
            lasers=[laser1, laser2],
        )
        ch, cb
    end

    mode = xmodes(chamber)[1]
    return (ca=ca, laser1=laser1, laser2=laser2, chamber=chamber, mode=mode)
end

"""
    configure_lasers!(setup, f_cl, f_sb, A; phi_1=0.0, phi_2=0.0)

Configure both lasers at carrier frequency `f_cl`, symmetric sideband detunings ±f_sb,
and intensity `A`. Mutates lasers in `setup`.
"""
function configure_lasers!(setup, f_cl::Float64, f_sb::Float64, A::Float64;
                           phi_1::Float64=0.0, phi_2::Float64=0.0)::Nothing
    laser1, laser2, chamber = setup.laser1, setup.laser2, setup.chamber
    wavelength!(laser1, pc.c / f_cl)
    detuning!(laser1, f_sb)
    polarization!(laser1, ẑ)
    wavevector!(laser1, x̂)
    wavelength!(laser2, pc.c / f_cl)
    detuning!(laser2, -f_sb)
    polarization!(laser2, ẑ)
    wavevector!(laser2, x̂)
    intensity!(laser1, A)
    intensity!(laser2, A)
    phase!(laser1, phi_1)
    phase!(laser2, phi_2)
    return nothing
end

"""
    evolve_reduced_density(setup, t; lamb_dicke_order=1) -> Matrix{Complex}

Time-evolve |SS⟩ ⊗ |0⟩ under IonSim Hamiltonian and return the reduced 2-qubit density matrix
as a plain complex matrix in the (SS, SD, DS, DD) basis.
"""
function evolve_reduced_density(setup, t::Float64; lamb_dicke_order::Int=1)
    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=lamb_dicke_order, rwa_cutoff=Inf)
    tout = Float64[0.0, t]
    _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)
    ρ = ptrace(sol[end] ⊗ dagger(sol[end]), [3]).data
    return ρ
end

# --- Public estimators

"""
    ideal(t) -> NamedTuple

Construct the standard MS-like setup from `t` (in µs-scale consistent with your timescale),
choose symmetric detunings and an intensity inferred from a π-time heuristic, evolve ideally,
and return (fid, f_cl, f_sb, A, delta_phi).

This function is deterministic and produces no I/O.
"""
function ideal(t::Float64)
    setup = build_chamber()
    mode = setup.mode
    ν = frequency(mode)
    ϵ = 1 / (t * 1e-6)

    # Transition setup on both lasers
    wavelength_from_transition!(setup.laser1, setup.ca, ("S", "D"), setup.chamber)
    detuning!(setup.laser1, ν + ϵ)
    polarization!(setup.laser1, ẑ)
    wavevector!(setup.laser1, x̂)
    phase!(setup.laser1, 0.0)

    wavelength_from_transition!(setup.laser2, setup.ca, ("S", "D"), setup.chamber)
    detuning!(setup.laser2, -ν - ϵ)
    polarization!(setup.laser2, ẑ)
    wavevector!(setup.laser2, x̂)
    phase!(setup.laser2, 0.0)

    c = pc.c
    f_b = c / setup.laser1.λ + setup.laser1.Δ
    f_r = c / setup.laser2.λ + setup.laser2.Δ
    f_cl = (f_b + f_r) / 2
    f_sb = (f_b - f_r) / 2

    η = abs(lambdicke(mode, setup.ca, setup.laser1))
    pi_time = η / ϵ
    A = intensity_from_pitime!(1, pi_time, 1, ("S", "D"), setup.chamber)
    intensity_from_pitime!(2, pi_time, 1, ("S", "D"), setup.chamber)

    ρ = evolve_reduced_density(setup, t)
    fid = bell_fidelity_phi_plus(ρ)

    return (fid=fid, f_cl=f_cl, f_sb=f_sb, A=A, delta_phi=0.0)
end

"""
    Q_det(t, f_cl, f_sb, A; phi_1=0.0, phi_2=0.0) -> Real

Deterministic fidelity estimator: evolve ideally with specified `(f_cl, f_sb, A, phi_1, phi_2)`
and compute `bell_fidelity_phi_plus` on the reduced state.
"""
function Q_det(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64;
               phi_1::Float64=0.0, phi_2::Float64=0.0)::Float64
    setup = build_chamber()
    configure_lasers!(setup, f_cl, f_sb, A, phi_1=phi_1, phi_2=phi_2)
    ρ = evolve_reduced_density(setup, t)
    return bell_fidelity_phi_plus(ρ)
end


function _varMS_sample_pops(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                             N::Int, numMS::Int,
                             relative_phase::Float64, phase_drift::Float64)
    N > 0 || throw(ArgumentError("N must be positive."))
    setup = build_chamber()
    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    tout = Float64[0.0, t]
    state = ca["S"] ⊗ ca["S"] ⊗ mode[0]
    for gate_idx in 1:numMS
        net_phase = (gate_idx - 1) * (relative_phase - phase_drift)
        configure_lasers!(setup, f_cl, Δ, I, phi_1=net_phase, phi_2=0.0)
        h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
        _, sol = timeevolution.schroedinger_dynamic(tout, state, h)
        state = sol[end]
    end
    SS = real(expect(ionprojector(chamber, "S", "S"), state))
    DD = real(expect(ionprojector(chamber, "D", "D"), state))
    SD = real(expect(ionprojector(chamber, "S", "D"), state))
    DS = real(expect(ionprojector(chamber, "D", "S"), state))
    weights = _normalized_population_weights((SS, DD, SD, DS))
    samples = StatsBase.sample(1:4, StatsBase.Weights(weights), N)
    return count(==(1), samples) / N, count(==(2), samples) / N
end

"""
    Q_varMS(t, f_cl, Δ, I; N=1000, numMS=2, relative_phase=0.0, phase_drift=0.0) -> Float64

Sampled population-score estimator for repeated MS pulses. The expected even
populations are inferred from `numMS`: odd counts target a balanced SS/DD readout,
`numMS % 4 == 2` targets DD, and `numMS % 4 == 0` targets SS.
"""
function Q_varMS(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                 N::Int=1000, numMS::Int=2,
                 relative_phase::Float64=0.0, phase_drift::Float64=0.0)::Float64
    P_SS, P_DD = _varMS_sample_pops(t, f_cl, Δ, I; N=N, numMS=numMS,
                                     relative_phase=relative_phase, phase_drift=phase_drift)
    expected = _expected_ms_even_populations(numMS)
    return clamp(1.0 - (abs(expected.SS - P_SS) + abs(expected.DD - P_DD)), 0.0, 1.0)
end

function Q_varMS_σ(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                   N::Int=1000, numMS::Int=2,
                   relative_phase::Float64=0.0, phase_drift::Float64=0.0)
    P_SS, P_DD = _varMS_sample_pops(t, f_cl, Δ, I; N=N, numMS=numMS,
                                     relative_phase=relative_phase, phase_drift=phase_drift)
    expected = _expected_ms_even_populations(numMS)
    Q = clamp(1.0 - (abs(expected.SS - P_SS) + abs(expected.DD - P_DD)), 0.0, 1.0)
    return Q, sigma_binomial(Q, N)
end

function Q_varMS_balance_σ(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                            N::Int=1000, numMS::Int=3,
                            relative_phase::Float64=0.0, phase_drift::Float64=0.0)
    P_SS, P_DD = _varMS_sample_pops(t, f_cl, Δ, I; N=N, numMS=numMS,
                                     relative_phase=relative_phase, phase_drift=phase_drift)
    Q = clamp(1.0 - (abs(0.5 - P_SS) + abs(0.5 - P_DD)), 0.0, 1.0)
    return Q, sigma_delta(P_SS, P_DD, 0.5, 0.5, N)
end

