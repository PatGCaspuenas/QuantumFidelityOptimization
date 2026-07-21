# src/calibration.jl

# ── Module-level constants (immutable, safe to share) ────────────────────────
# Parity lookup: index 1=SS(+1), 2=SD(-1), 3=DS(-1), 4=DD(+1)
const PARITY_VALUES = (1, -1, -1, 1)
const _TARGET_PARITY_ANALYSIS_PHASE = π / 4
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

# --- QuantumOptics helpers (kept small and pure)

# Global two-qubit rotation used in parity-scan estimators.
function global_rotation(θ::Real, φ::Real=0.0)
    s, c = sincos(θ / 2)
    eφm = cis(-φ)
    eφp = cis(φ)
    U = ComplexF64[c        -1im*s*eφm;
                   -1im*s*eφp  c]
    return Operator(_TWO_QUBIT_BASIS, kron(U, U))
end

proj(state, ρ) = real(expect(state ⊗ dagger(state), ρ))

function _normalized_population_weights(values::NTuple{4,<:Real})
    weights = Float64[max(Float64(v), 0.0) for v in values]
    total = sum(weights)
    total > 0.0 || throw(ArgumentError("Population weights must have positive total."))
    return weights ./ total
end

function _shot_count_or_inf(N::Real)
    n = Float64(N)
    if isinf(n) && n > 0.0
        return Inf
    end
    isfinite(n) || throw(ArgumentError("N must be a positive integer or Inf."))
    isinteger(n) || throw(ArgumentError("N must be a positive integer or Inf, got $N."))
    n_int = Int(n)
    n_int > 0 || throw(ArgumentError("N must be positive."))
    return n_int
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

"""
    Q_noisy(t, f_cl, f_sb, A; phi_1=0.0, phi_2=0.0, N=100, phase_grid=0:0.1:π) -> Real

Noisy Bell-state fidelity estimator based on sampled populations and a parity scan.
The parity scan extracts the signed coherence at the target analysis phase, so
accumulated Bell-phase errors reduce the returned fidelity instead of being fit away.
Use `N=Inf` to return the Born-rule expectation value without projection sampling.
"""
function Q_noisy(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64;
                 phi_1::Float64=0.0, phi_2::Float64=0.0,
                 N::Real=100, phase_grid::AbstractRange{Float64}=0.0:0.1:π)::Float64
    n_eval = _shot_count_or_inf(N)
    scan_phases = collect(Float64, phase_grid)
    length(scan_phases) ≥ 3 || throw(ArgumentError("phase_grid must contain at least 3 points."))

    setup = build_chamber()
    configure_lasers!(setup, f_cl, f_sb, A, phi_1=phi_1, phi_2=phi_2)

    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
    tout = Float64[0.0, t]
    _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)

    # Outcome probabilities in computational basis from projectors
    SS = real(expect(ionprojector(chamber, "S", "S"), sol[end]))
    SD = real(expect(ionprojector(chamber, "S", "D"), sol[end]))
    DS = real(expect(ionprojector(chamber, "D", "S"), sol[end]))
    DD = real(expect(ionprojector(chamber, "D", "D"), sol[end]))

    weights = _normalized_population_weights((SS, SD, DS, DD))
    if isinf(Float64(n_eval))
        P_odd = weights[2] + weights[3]
    else
        samples = StatsBase.sample(1:4, StatsBase.Weights(weights), n_eval)
        P_odd = count(s -> s == 2 || s == 3, samples) / n_eval
    end

    ρ_red = ptrace(sol[end] ⊗ dagger(sol[end]), [3])
    ρ = Operator(_TWO_QUBIT_BASIS, ρ_red.data)

    SSs = tensor(spinup(_SPIN_BASIS), spinup(_SPIN_BASIS))
    SDs = tensor(spinup(_SPIN_BASIS), spindown(_SPIN_BASIS))
    DSs = tensor(spindown(_SPIN_BASIS), spinup(_SPIN_BASIS))
    DDs = tensor(spindown(_SPIN_BASIS), spindown(_SPIN_BASIS))

    meas = zeros(Float64, length(phase_grid))
    p = Vector{Float64}(undef, 4)
    for (i, φ) in enumerate(phase_grid)
        Rφ = global_rotation(π / 2, φ)
        ρφ = Rφ * ρ * dagger(Rφ)
        p[1] = proj(SSs, ρφ)
        p[2] = proj(SDs, ρφ)
        p[3] = proj(DSs, ρφ)
        p[4] = proj(DDs, ρφ)
        p .= _normalized_population_weights((p[1], p[2], p[3], p[4]))

        if isinf(Float64(n_eval))
            meas[i] = sum(PARITY_VALUES[j] * p[j] for j in 1:4)
        else
            s = StatsBase.sample(1:4, StatsBase.Weights(p), n_eval)
            meas[i] = sum(PARITY_VALUES[x] for x in s) / n_eval
        end
    end

    X = hcat(cos.(2.0 .* scan_phases), sin.(2.0 .* scan_phases), ones(length(scan_phases)))
    coeff = X \ meas

    target_phase = _TARGET_PARITY_ANALYSIS_PHASE
    parity_at_target = coeff[1] * cos(2.0 * target_phase) + coeff[2] * sin(2.0 * target_phase)
    C = -parity_at_target

    return clamp((1 - P_odd + C) / 2, 0.0, 1.0)
end

# Normalized (p_ss, p_sd, p_ds, p_dd) populations after `numMS` closed-loop MS(π/2) gates.
function varms_weights(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                       numMS::Int=2, relative_phase::Float64=0.0, phase_drift::Float64=0.0)
    subgates = [ms_subgate(π / 2, 0.0) for _ in 1:numMS]
    pulses = build_closed_loop_ms_sequence(t, f_cl, Δ, I, subgates;
                                           relative_phase=relative_phase, phase_drift=phase_drift)
    pops = populations_ms_sequence(pulses)
    return _normalized_population_weights((pops.gg, pops.eg, pops.ge, pops.ee))
end

"""
    Q_varMS(t, f_cl, Δ, I; N=1000, numMS=2, relative_phase=0.0, phase_drift=0.0) -> (y, σy)

Fidelity observation for a closed-loop sequence of `numMS` MS(π/2) gates,
targeting all population in |DD⟩. Returns the score `y = p_dd` (population
observed in |DD⟩) and the binomial projection-noise std `σy`. With finite `N`
`p_dd` is estimated from multinomially sampled shots; `N=Inf` returns the
exact Born-rule population with `σy=0`.
"""
function Q_varMS(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                 N::Real=1000, numMS::Int=2,
                 relative_phase::Float64=0.0,
                 phase_drift::Float64=0.0,
                 rng::Random.AbstractRNG=Random.default_rng())
    n_eval = _shot_count_or_inf(N)
    w = varms_weights(t, f_cl, Δ, I; numMS=numMS,
                      relative_phase=relative_phase, phase_drift=phase_drift)
    isinf(Float64(n_eval)) && return clamp(w[4], 0.0, 1.0), 0.0
    p_dd = w[4]
    σy = sqrt(max(p_dd * (1.0 - p_dd), 0.0) / Float64(n_eval))
    counts = rand(rng, Distributions.Multinomial(Int(n_eval), collect(Float64, w)))
    return clamp(counts[4] / Float64(n_eval), 0.0, 1.0), σy
end
