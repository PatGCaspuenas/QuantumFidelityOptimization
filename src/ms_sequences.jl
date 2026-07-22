# src/ms_sequences.jl
#
# Closed-loop MS subgate pulse construction + IonSim evolution. Used by the
# Q_varMS estimator in calibration.jl.

using OrdinaryDiffEqVerner: Vern7

Base.@kwdef struct MSSubgate
    theta::Float64
    phi::Float64 = 0.0
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
