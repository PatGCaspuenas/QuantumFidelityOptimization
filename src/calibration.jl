# src/calibration.jl

# ── Module-level constants (immutable, safe to share) ────────────────────────
# Parity lookup: index 1=SS(+1), 2=SD(-1), 3=DS(-1), 4=DD(+1)
const PARITY_VALUES = (1, -1, -1, 1)
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

# Convert IonSim reduced density operator to a QuantumOptics.Operator on 2 qubits
function ion_sim_to_qo_operator(ρ)
    return Operator(_TWO_QUBIT_BASIS, ρ.data)
end

proj(state, ρ) = real(expect(state ⊗ dagger(state), ρ))

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

Noisy estimator based on sampling + parity scan + cosine fit.

Requires `StatsBase` and `LsqFit`. This method attempts to load them at call-time
and throws an informative error if unavailable.
"""
function Q_noisy(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64;
                 phi_1::Float64=0.0, phi_2::Float64=0.0,
                 N::Int=100, phase_grid::AbstractRange{Float64}=0.0:0.1:π)::Float64

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

    weights = Float64[SS, SD, DS, DD]

    samples = StatsBase.sample(1:4, StatsBase.Weights(weights), N)
    P_odd = count(s -> s == 2 || s == 3, samples) / N

    ρ_red = ptrace(sol[end] ⊗ dagger(sol[end]), [3])
    ρ = ion_sim_to_qo_operator(ρ_red)

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
        @. p = max(p, 0.0)
        p ./= sum(p)

        s = StatsBase.sample(1:4, StatsBase.Weights(p), N)
        meas[i] = sum(PARITY_VALUES[x] for x in s) / N
    end

    model(φ, par) = @. par[1] * cos(par[2] * φ + par[3]) + par[4]
    p0 = Float64[0.8, 1.0, 0.0, 0.0]
    fit = LsqFit.curve_fit(model, collect(Float64, phase_grid), meas, p0,
        lower=Float64[-1.0, -2.0, -Inf, -0.1],
        upper=Float64[1.0, 2.0, Inf, 0.1])
    C = abs(fit.param[1])

    return (1 - P_odd + C) / 2
end

function Q_varMS(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                 N::Int=1000, numMS::Int=2,
                 phi_1::Float64=0.0, phi_2::Float64=0.0)::Float64

    setup = build_chamber()
    configure_lasers!(setup, f_cl, Δ, I, phi_1=phi_1, phi_2=phi_2)

    ca, chamber, mode = setup.ca, setup.chamber, setup.mode

    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
    tout = Float64[0.0, t]
    _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)

    for _ in 2:numMS  # no-op when numMS == 1
        _, sol = timeevolution.schroedinger_dynamic(tout, sol[end], h)
    end

    SS = real(expect(ionprojector(chamber, "S", "S"), sol[end]))
    DD = real(expect(ionprojector(chamber, "D", "D"), sol[end]))
    SD = real(expect(ionprojector(chamber, "S", "D"), sol[end]))
    DS = real(expect(ionprojector(chamber, "D", "S"), sol[end]))

    # Define success predicate based on the number of gates (zero-alloc)
    success_pred = if isodd(numMS)
        s -> s == 1 || s == 2   # SS or DD
    elseif numMS % 4 == 2
        s -> s == 2             # DD only
    else
        s -> s == 1             # SS only
    end

    weights = Float64[max(SS, 0.0), max(DD, 0.0), max(SD, 0.0), max(DS, 0.0)]
    samples = StatsBase.sample(1:4, StatsBase.Weights(weights), N)

    parity = count(success_pred, samples)

    return parity / N
end

"""
    Q_mc_varMS(t, f_cl, Δ, I; N=50, numMS=2, ..., δ_rms_hz=300.0, Ω_rms_frac=0.007) -> Float64

Monte-Carlo fidelity estimator with shot-to-shot parameter noise AND quantum
projection noise.  For each of `N` shots:
  1. Draw quasi-static detuning offset  δ ~ N(0, δ_rms_hz·2π)
  2. Draw quasi-static amplitude error   ε ~ N(0, Ω_rms_frac), intensity scales as (1+ε)²
  3. Run full Hamiltonian propagation with perturbed parameters
  4. Sample a single measurement outcome from the population probabilities

Returns fraction of successful outcomes / N.
"""
function Q_mc_varMS(t::Float64, f_cl::Float64, Δ::Float64, I::Float64;
                    N::Int=50, numMS::Int=2,
                    phi_1::Float64=0.0, phi_2::Float64=0.0,
                    δ_rms_hz::Float64=300.0, Ω_rms_frac::Float64=0.007,
                    rng::Random.AbstractRNG=Random.default_rng())::Float64

    setup = build_chamber()
    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    tout = Float64[0.0, t]
    init_state = ca["S"] ⊗ ca["S"] ⊗ mode[0]

    success_pred = if isodd(numMS)
        s -> s == 1 || s == 2   # SS or DD
    elseif numMS % 4 == 2
        s -> s == 2             # DD only
    else
        s -> s == 1             # SS only
    end

    successes = 0
    for _ in 1:N
        # Shot-to-shot parameter perturbation
        δ  = randn(rng) * δ_rms_hz * 2π
        ε  = randn(rng) * Ω_rms_frac
        Δ_noisy = Δ + δ
        I_noisy = I * (1.0 + ε)^2

        configure_lasers!(setup, f_cl, Δ_noisy, I_noisy;
                          phi_1=phi_1, phi_2=phi_2)

        h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
        _, sol = timeevolution.schroedinger_dynamic(tout, init_state, h)

        for _ in 2:numMS
            _, sol = timeevolution.schroedinger_dynamic(tout, sol[end], h)
        end

        SS = real(expect(ionprojector(chamber, "S", "S"), sol[end]))
        DD = real(expect(ionprojector(chamber, "D", "D"), sol[end]))
        SD = real(expect(ionprojector(chamber, "S", "D"), sol[end]))
        DS = real(expect(ionprojector(chamber, "D", "S"), sol[end]))

        weights = Float64[max(SS, 0.0), max(DD, 0.0), max(SD, 0.0), max(DS, 0.0)]
        outcome = StatsBase.sample(1:4, StatsBase.Weights(weights))

        if success_pred(outcome)
            successes += 1
        end
    end

    return successes / N
end
