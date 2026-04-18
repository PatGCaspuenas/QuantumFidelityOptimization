# scripts/plot_3ms_balance_fsb_slice.jl
#
# Standalone: fixes f_cl and A from benchmark_3ms_balance_N10000.txt, scans f_sb,
# plots Born-rule 1 - (|0.5-SS| + |0.5-DD|) after the same 3×MS stack as Q_varMS_balance.
#
# Optional ENV:
#   SLICE_FSB_HALF_WIDTH_KHZ — half-width of f_sb scan in kHz (default 2.0, matches BO box)
#   SLICE_NPTS               — number of grid points (default 400)
#   SLICE_OUT                — output PNG path (default figures/3ms_balance_born_fsb_slice.png)
#   SLICE_F_CL, SLICE_F_SB0, SLICE_A — override physical defaults (all Float64 parseable)

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using IonSim
using QuantumOptics
using Plots

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode

"""Same evolution as `Q_varMS_balance`; returns Born 1 - |½-SS| - |½-DD| (no shot noise)."""
function balance_score_born(t::Float64, f_cl::Float64, f_sb::Float64, A::Float64;
                            numMS::Int=3, phi::Float64=0.0)::Float64
    setup = CalibrationCode.build_chamber()
    CalibrationCode.configure_lasers!(setup, f_cl, f_sb, A; phi_1=phi, phi_2=phi)
    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    h = hamiltonian(chamber; timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
    tout = Float64[0.0, t]
    _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)
    for _ in 2:numMS
        _, sol = timeevolution.schroedinger_dynamic(tout, sol[end], h)
    end
    SS = max(0.0, real(expect(ionprojector(chamber, "S", "S"), sol[end])))
    DD = max(0.0, real(expect(ionprojector(chamber, "D", "D"), sol[end])))
    return 1.0 - (abs(0.5 - SS) + abs(0.5 - DD))
end

function env_float(key::String, default::Float64)::Float64
    v = get(ENV, key, "")
    isempty(v) && return default
    return parse(Float64, v)
end

function env_int(key::String, default::Int)::Int
    v = get(ENV, key, "")
    isempty(v) && return default
    return parse(Int, v)
end

# Recommended + baseline f_sb from scripts/data/benchmark_3ms_balance_N10000.txt
const _DEF_F_CL = 4.11155035201374e14
const _DEF_F_SB_CENTER = 3.0171965020022583e6
const _DEF_A = 1.2530856303064195e6
const _BASELINE_F_SB = 3.01e6

function main()
    t = 100.0
    numMS = 3
    phi = 0.0

    f_cl = env_float("SLICE_F_CL", _DEF_F_CL)
    f_sb_center = env_float("SLICE_F_SB0", _DEF_F_SB_CENTER)
    A = env_float("SLICE_A", _DEF_A)

    half_width_khz = env_float("SLICE_FSB_HALF_WIDTH_KHZ", 2.0)
    span_fsb = half_width_khz * 1e3 * 2π
    npts = max(2, env_int("SLICE_NPTS", 400))

    out = get(ENV, "SLICE_OUT", joinpath(@__DIR__, "..", "figures", "3ms_balance_born_fsb_slice.png"))

    fsb_grid = range(f_sb_center - span_fsb, f_sb_center + span_fsb; length=npts)
    ys_balance = Vector{Float64}(undef, length(fsb_grid))
    ys_qdet = Vector{Float64}(undef, length(fsb_grid))
    for (i, fsb) in enumerate(fsb_grid)
        ys_balance[i] = try
            balance_score_born(t, f_cl, fsb, A; numMS=numMS, phi=phi)
        catch
            NaN
        end
        ys_qdet[i] = try
            CalibrationCode.Q_det(t, f_cl, fsb, A)
        catch
            NaN
        end
    end

    x_khz = collect(fsb_grid .- f_sb_center) ./ (2π * 1e3)
    baseline_offset_khz = (_BASELINE_F_SB - f_sb_center) / (2π * 1e3)

    p = plot(x_khz, ys_balance;
             lw=2, label="3MS balance (Born)",
             xlabel="Δf_sb from scan center (kHz)",
             ylabel="Score",
             title="3MS balance & Q_det Born slice (fixed f_cl, A)\n" *
                   "f_cl=$(round(f_cl, sigdigits=6)), A=$(round(A, sigdigits=6)), t=$(t) µs",
             legend=:topright)
    plot!(p, x_khz, ys_qdet; lw=2, ls=:dash, label="Q_det (Bell fidelity)")
    vline!(p, [0.0]; ls=:dash, lw=1.5, color=:red, label="scan center (rec. f_sb)")
    vline!(p, [baseline_offset_khz]; ls=:dot, lw=1.5, color=:green, label="baseline f_sb")

    mkpath(dirname(out))
    savefig(p, out)
    println("Wrote $out")
    return nothing
end

main()
