# scripts/response_surface_all_q_slices.jl
#
# Samples every Q function along 4 independent 1D slices of the input space
# (one slice per parameter: f_cl, f_sb, A, phi), fixing the other three at
# their ideal centre values (u=0).
#
# Functions evaluated:
#   Q_varMS         numMS = 2, 3          (shot noise, multi-gate)
#   Q_varMS_balance numMS = 3             (balance metric, shot noise)
#   Q_jacobian      (Q_ms_sequence with searched subgates, if available)
#   Q_det           (noiseless, 3-param only — phi slice will be flat)
#   Q_noisy         (single-gate parity scan)
#
# ENV variables (all optional):
#   SLICE_NPTS        — grid points per slice          (default 80)
#   SLICE_N           — shots per Q call               (default 1000)
#   SLICE_OUTPUT      — output CSV path                (default data/response_surface_all_q_slices.csv)
#   SLICE_PHASE_DRIFT — phase drift per gate (rad)     (default 0.0)

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Statistics
using Printf
using IonSim, QuantumOptics, StatsBase

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

# ── Helpers ─────────────────────────────────────────────────────────────────

function env_int(key, default)
    v = get(ENV, key, "")
    isempty(v) ? default : parse(Int, v)
end
function env_float(key, default)
    v = get(ENV, key, "")
    isempty(v) ? default : parse(Float64, v)
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    npts        = env_int("SLICE_NPTS", 80)
    N           = env_int("SLICE_N", 1000)
    phase_drift = env_float("SLICE_PHASE_DRIFT", 0.0)
    output      = get(ENV, "SLICE_OUTPUT",
                      joinpath(@__DIR__, "..", "data",
                               "response_surface_all_q_slices.csv"))

    println("=== Response surface 1D slices — all Q functions ===")
    @printf("npts=%d  N=%d  threads=%d  phase_drift=%.4f\n",
            npts, N, Threads.nthreads(), phase_drift)

    t = 100.0
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    span_kHz  = 2.0
    span_fcl  = span_kHz * 1e3 * 2π
    span_fsb  = span_kHz * 1e3 * 2π
    span_A    = 1.2 * A0 - A0
    span_phi  = π / 10

    # Normalised → physical
    function u_to_phys(u1, u2, u3, u4)
        fcl = f_cl0 + span_fcl * u1
        fsb = f_sb0 + span_fsb * u2
        A   = A0    + span_A   * u3
        phi =         span_phi * u4
        return fcl, fsb, A, phi
    end

    # Load Jacobian subgates if available
    _jac_path = joinpath(@__DIR__, "..", "data", "ms_sequence_search_result.jl")
    jac_available = isfile(_jac_path)
    jac_subgates  = CalibrationCode.MSSubgate[]
    jac_exp_gg    = NaN
    jac_exp_ee    = NaN
    if jac_available
        try
            raw = include(_jac_path)
            jac_subgates = [CalibrationCode.MSSubgate(sg.theta, sg.phi)
                            for sg in raw.best_overall.subgates]
            I_center = Float64(raw.I_center)
            pulses   = CalibrationCode.build_closed_loop_ms_sequence(
                           t, f_cl0, f_sb0, I_center, jac_subgates)
            pops     = CalibrationCode.populations_ms_sequence(pulses)
            jac_exp_gg = Float64(pops.gg)
            jac_exp_ee = Float64(pops.ee)
            println("Jacobian subgates loaded. expected_gg=$(round(jac_exp_gg,digits=5)), " *
                    "expected_ee=$(round(jac_exp_ee,digits=5))")
        catch e
            println("Warning: failed to load Jacobian subgates — will write NaN: $e")
            jac_available = false
        end
    else
        println("Jacobian search result not found — Q_jacobian column will be NaN")
    end

    phase_grid = 0.0:0.1:π

    # Each slice: (name, which u index varies, the other three fixed at 0)
    slices = [("fcl", 1), ("fsb", 2), ("A", 3), ("phi", 4)]
    us     = collect(range(-1.0, 1.0; length=npts))

    # All (slice, point) jobs as a flat list so Threads.@threads can distribute them
    jobs = [(sname, sidx, u_val)
            for (sname, sidx) in slices
            for u_val in us]

    total_pts = length(jobs)
    rows      = Vector{String}(undef, total_pts)
    done_cnt  = Threads.Atomic{Int}(0)

    # Safe wrapper — returns NaN on any error without crashing the thread
    safe(f) = try Float64(f()) catch; NaN end

    println("Dispatching $total_pts points across $(Threads.nthreads()) thread(s)...")
    flush(stdout)

    Threads.@threads for idx in 1:total_pts
        sname, sidx, u_val = jobs[idx]

        u1 = sidx == 1 ? u_val : 0.0
        u2 = sidx == 2 ? u_val : 0.0
        u3 = sidx == 3 ? u_val : 0.0
        u4 = sidx == 4 ? u_val : 0.0
        fcl, fsb, A, phi = u_to_phys(u1, u2, u3, u4)

        # Q_varMS (numMS 2 and 3)
        qv2 = safe(() -> CalibrationCode.Q_varMS(t, fcl, fsb, A;
                         N=N, numMS=2,
                         relative_phase=phi, phase_drift=phase_drift))
        qv3 = safe(() -> CalibrationCode.Q_varMS(t, fcl, fsb, A;
                         N=N, numMS=3,
                         relative_phase=phi, phase_drift=phase_drift))

        # Q_varMS_balance (numMS 3)
        qb3 = safe(() -> CalibrationCode.Q_varMS_balance(t, fcl, fsb, A;
                         N=N, numMS=3,
                         relative_phase=phi, phase_drift=phase_drift))

        # Q_jacobian
        qjac = if jac_available
            safe(() -> clamp(CalibrationCode.Q_ms_sequence(
                           t, fcl, fsb, A, jac_subgates;
                           N=N,
                           expected_gg=jac_exp_gg, expected_ee=jac_exp_ee,
                           relative_phase=phi, phase_drift=phase_drift), 0.0, 1.0))
        else
            NaN
        end

        # Q_det — phi-insensitive; constant on the phi slice
        qdet = safe(() -> clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0))

        # Q_noisy — single-gate parity scan from calibration.jl
        qn1 = safe(() -> CalibrationCode.Q_noisy(t, fcl, fsb, A;
                        phi_1=phi, phi_2=0.0,
                        N=N, phase_grid=phase_grid))

        rows[idx] = @sprintf(
            "%s,%.6f,%.15e,%.15e,%.15e,%.10f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f",
            sname, u_val, fcl, fsb, A, phi,
            qv2, qv3, qb3, qjac, qdet, qn1)

        n = Threads.atomic_add!(done_cnt, 1) + 1
        if n % max(1, total_pts ÷ 20) == 0 || n == total_pts
            @printf("  [%d/%d] slice=%s u=%.3f  Q_varMS_2=%.4f  Q_det=%.4f\n",
                    n, total_pts, sname, u_val, qv2, qdet)
            flush(stdout)
        end
    end

    # CSV header
    header = join([
        "slice", "u",
        "f_cl", "f_sb", "A", "phi",
        "Q_varMS_2ms", "Q_varMS_3ms",
        "Q_varMS_balance_3ms",
        "Q_jacobian",
        "Q_det",
        "Q_noisy_1ms",
    ], ",")

    mkpath(dirname(output))
    open(output, "w") do io
        println(io, header)
        for r in rows
            println(io, r)
        end
    end
    println("\nWrote $(length(rows)) rows to: $output")
    return nothing
end

main()
