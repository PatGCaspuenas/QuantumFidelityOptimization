# src/ms_sequences.jl
#
# Shared helpers for closed-loop MS subgate concatenations used by search and
# plotting scripts. These live in `src/` so all callers exercise the same
# pulse-construction and IonSim evolution path.

const _ODEMod = Base.loaded_modules[Base.PkgId(
    Base.UUID("1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"), "OrdinaryDiffEq")]
const Vern7 = _ODEMod.Vern7

Base.@kwdef struct MSSubgate
    theta::Float64
    phi::Float64 = 0.0
end

Base.@kwdef struct MSSequenceSpec
    name::Symbol
    label::String
    subgates::Vector{MSSubgate}
    linestyle::Symbol = :solid
end

ms_subgate(theta::Real, phi::Real=0.0) = MSSubgate(Float64(theta), Float64(phi))

make_ms_pulse(t, f_cl, Δ, I, phi_1, phi_2) = (
    t=Float64(t),
    f_cl=Float64(f_cl),
    Δ=Float64(Δ),
    I=Float64(I),
    phi_1=Float64(phi_1),
    phi_2=Float64(phi_2),
)

function build_closed_loop_ms_sequence(t, f_cl, Δ, I_pi2,
                                       subgates::AbstractVector{<:MSSubgate};
                                       omega_ratio::Float64=1.0,
                                       relative_phase::Float64=0.0,
                                       phase_drift::Float64=0.0)
    pulses = Vector{NamedTuple}(undef, length(subgates))
    for (pulse_idx, subgate) in enumerate(subgates)
        accumulated = (pulse_idx - 1)
        intensity = I_pi2 * (2.0 * subgate.theta / π) * omega_ratio^2
        net_phase = accumulated * (relative_phase - phase_drift)
        pulses[pulse_idx] = make_ms_pulse(
            t,
            f_cl,
            Δ,
            intensity,
            subgate.phi + net_phase,
            subgate.phi,
        )
    end
    return pulses
end

function populations_ms_sequence(pulses)
    setup = build_chamber()
    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    state = ca["S"] ⊗ ca["S"] ⊗ mode[0]
    for pulse in pulses
        configure_lasers!(setup, pulse.f_cl, pulse.Δ, pulse.I;
                          phi_1=pulse.phi_1, phi_2=pulse.phi_2)
        h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
        tout = Float64[0.0, pulse.t]
        _, sol = timeevolution.schroedinger_dynamic(tout, state, h; alg=Vern7())
        state = sol[end]
    end
    SS = real(expect(ionprojector(chamber, "S", "S"), state))
    SD = real(expect(ionprojector(chamber, "S", "D"), state))
    DS = real(expect(ionprojector(chamber, "D", "S"), state))
    DD = real(expect(ionprojector(chamber, "D", "D"), state))
    return (gg=SS, eg=SD, ge=DS, ee=DD)
end

function reduced_density_ms_sequence(pulses)
    setup = build_chamber()
    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    state = ca["S"] ⊗ ca["S"] ⊗ mode[0]
    for pulse in pulses
        configure_lasers!(setup, pulse.f_cl, pulse.Δ, pulse.I;
                          phi_1=pulse.phi_1, phi_2=pulse.phi_2)
        h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
        tout = Float64[0.0, pulse.t]
        _, sol = timeevolution.schroedinger_dynamic(tout, state, h; alg=Vern7())
        state = sol[end]
    end
    return ptrace(state ⊗ dagger(state), [3]).data
end

function trinary_ms_probabilities(pops)
    probs = Float64[
        max(pops.gg, 0.0),
        max(pops.ee, 0.0),
        max(pops.eg + pops.ge, 0.0),
    ]
    total = sum(probs)
    total > 0.0 || return (gg=0.5, ee=0.5, odd=0.0)
    probs ./= total
    return (gg=probs[1], ee=probs[2], odd=probs[3])
end

ms_balance_and_odd(pops) = begin
    probs = trinary_ms_probabilities(pops)
    Float64[probs.gg - probs.ee, probs.odd]
end

function ms_observable_covariance(pops)
    probs = trinary_ms_probabilities(pops)
    p = Float64[probs.gg, probs.ee, probs.odd]
    Σp = Diagonal(p) - p * transpose(p)
    A = Float64[1.0 -1.0 0.0;
                0.0  0.0 1.0]
    return A * Σp * transpose(A)
end

function same_ms_subgates(a::AbstractVector{<:MSSubgate},
                          b::AbstractVector{<:MSSubgate};
                          atol_theta::Float64=1e-8,
                          atol_phi::Float64=1e-8)
    length(a) == length(b) || return false
    return all(
        isapprox(ga.theta, gb.theta; atol=atol_theta, rtol=0.0) &&
        isapprox(ga.phi, gb.phi; atol=atol_phi, rtol=0.0)
        for (ga, gb) in zip(a, b)
    )
end

sequence_A_subgates() = [
    ms_subgate(π / 2, 0.0),
    ms_subgate(π / 2, 0.0),
    ms_subgate(π / 2, 0.0),
]

sequence_B_subgates() = [
    ms_subgate(π / 2, 0.0),
    ms_subgate(π / 2, π / 4),
]

default_ms_sequence_specs() = [
    MSSequenceSpec(
        name=:seq_A,
        label="3 × MS₀(π/2)",
        subgates=sequence_A_subgates(),
        linestyle=:dot,
    ),
    MSSequenceSpec(
        name=:seq_B,
        label="MS₀(π/2) then MS_{π/4}(π/2)",
        subgates=sequence_B_subgates(),
        linestyle=:dash,
    ),
]

function refine_bell_ms_sequence_omega_ratio(t, f_cl, Δ, I_pi2,
                                             subgates::AbstractVector{<:MSSubgate};
                                             ratio_grid::AbstractVector{<:Real})
    best_ratio = Float64(first(ratio_grid))
    best_fid = -Inf
    for ratio in ratio_grid
        pulses = build_closed_loop_ms_sequence(
            t, f_cl, Δ, I_pi2, subgates; omega_ratio=Float64(ratio))
        fid = bell_fidelity_phi_plus(reduced_density_ms_sequence(pulses))
        if fid > best_fid
            best_ratio = Float64(ratio)
            best_fid = fid
        end
    end
    return (ratio=best_ratio, fid=best_fid)
end

# Bias correction for |d + ε| where ε ~ N(0, σ²), d = |measured - reference| ≥ 0.
# Returns E[|d+ε|] - d, the systematic upward bias in the absolute-deviation estimator.
# Largest (= σ√(2/π)) at d=0 (true optimum) and decays to 0 as d ≫ σ.
@inline function _folded_bias(d::Float64, σ::Float64)::Float64
    σ < 1e-15 && return 0.0
    z = d / σ
    return σ * sqrt(2.0 / π) * exp(-0.5 * z^2) - d * (1.0 - erf(z / sqrt(2.0)))
end

# Variance of |d + ε| where ε ~ N(0, σ²), d ≥ 0.
# Var[|X|] = E[X²] - E[|X|]² = σ² + d² - E[|X|]²
# Used as the consistent noise model after debiasing: replaces binomial sigma_delta.
@inline function _folded_var(d::Float64, σ::Float64)::Float64
    σ < 1e-15 && return 0.0
    E_abs = d + _folded_bias(d, σ)
    return max(σ^2 + d^2 - E_abs^2, 0.0)
end

function Q_ms_sequence_det(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64,
                           subgates::AbstractVector{<:MSSubgate};
                           relative_phase::Float64=0.0,
                           phase_drift::Float64=0.0)::Float64
    pulses = build_closed_loop_ms_sequence(t, f_cl, f_sb, A, subgates;
                                           relative_phase=relative_phase,
                                           phase_drift=phase_drift)
    rho = reduced_density_ms_sequence(pulses)
    return bell_fidelity_phi_plus(rho)
end

function Q_ms_sequence(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64,
                       subgates::AbstractVector{<:MSSubgate};
                       N::Int=400,
                       expected_gg::Float64=NaN,
                       expected_ee::Float64=NaN,
                       relative_phase::Float64=0.0,
                       phase_drift::Float64=0.0,
                       debias::Bool=false)::Float64
    pulses = build_closed_loop_ms_sequence(t, f_cl, f_sb, A, subgates;
                                           relative_phase=relative_phase,
                                           phase_drift=phase_drift)
    pops = populations_ms_sequence(pulses)
    weights = Float64[max(pops.gg, 0.0), max(pops.ee, 0.0),
                      max(pops.eg, 0.0), max(pops.ge, 0.0)]
    samples = StatsBase.sample(1:4, StatsBase.Weights(weights), N)
    P_SS = count(==(1), samples) / N
    P_DD = count(==(2), samples) / N
    if !isnan(expected_gg)
        Q = 1.0 - (abs(expected_gg - P_SS) + abs(expected_ee - P_DD))
        if debias
            σ_gg = sqrt(max(P_SS * (1.0 - P_SS), 0.0) / N)
            σ_ee = sqrt(max(P_DD * (1.0 - P_DD), 0.0) / N)
            Q += _folded_bias(abs(expected_gg - P_SS), σ_gg) +
                 _folded_bias(abs(expected_ee - P_DD), σ_ee)
        end
        return Q
    end
    return P_SS + P_DD
end

function Q_ms_sequence_σ(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64,
                         subgates::AbstractVector{<:MSSubgate};
                         N::Int=400,
                         expected_gg::Float64=NaN,
                         expected_ee::Float64=NaN,
                         relative_phase::Float64=0.0,
                         phase_drift::Float64=0.0,
                         debias::Bool=false)
    pulses = build_closed_loop_ms_sequence(t, f_cl, f_sb, A, subgates;
                                           relative_phase=relative_phase,
                                           phase_drift=phase_drift)
    pops = populations_ms_sequence(pulses)
    weights = Float64[max(pops.gg, 0.0), max(pops.ee, 0.0),
                      max(pops.eg, 0.0), max(pops.ge, 0.0)]
    samples = StatsBase.sample(1:4, StatsBase.Weights(weights), N)
    P_SS = count(==(1), samples) / N
    P_DD = count(==(2), samples) / N
    if !isnan(expected_gg)
        d_gg = abs(expected_gg - P_SS)
        d_ee = abs(expected_ee - P_DD)
        Q = clamp(1.0 - d_gg - d_ee, 0.0, 1.0)
        σ_gg = sqrt(max(P_SS * (1.0 - P_SS), 0.0) / N)
        σ_ee = sqrt(max(P_DD * (1.0 - P_DD), 0.0) / N)
        if debias
            Q = clamp(Q + _folded_bias(d_gg, σ_gg) + _folded_bias(d_ee, σ_ee), 0.0, 1.0)
            return Q, sqrt(max(_folded_var(d_gg, σ_gg) + _folded_var(d_ee, σ_ee), 0.0))
        end
        return Q, sigma_delta(P_SS, P_DD, expected_gg, expected_ee, N)
    end
    Q = clamp(P_SS + P_DD, 0.0, 1.0)
    return Q, sigma_binomial(Q, N)
end

function Q_ms_sequence_probs(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64,
                              subgates::AbstractVector{<:MSSubgate};
                              relative_phase::Float64=0.0,
                              phase_drift::Float64=0.0)::NTuple{4,Float64}
    pulses = build_closed_loop_ms_sequence(t, f_cl, f_sb, A, subgates;
                                           relative_phase=relative_phase,
                                           phase_drift=phase_drift)
    pops = populations_ms_sequence(pulses)
    return (max(pops.gg, 0.0), max(pops.ee, 0.0), max(pops.eg, 0.0), max(pops.ge, 0.0))
end

function sequence_C_subgates()
    return [
        ms_subgate(3π / 16, 0.0),
        ms_subgate(π / 4, 0.0),
        ms_subgate(5π / 16, 0.0),
    ]
end
