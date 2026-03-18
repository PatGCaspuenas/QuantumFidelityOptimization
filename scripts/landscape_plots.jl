using Distributed
if nprocs() == 1
    addprocs()
end

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

@everywhere begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
end

@everywhere include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
@everywhere using .CalibrationCode
@everywhere using IonSim

using Plots
using Measures

@everywhere using QuantumOptics

# -----------------------------------------------------------------------
# Deterministic P(|DD>) after numMS sequential MS gate evolutions
# -----------------------------------------------------------------------
@everywhere function P_DD_det(t, f_cl, f_sb, A; numMS::Int=2, dt::Float64=0.1)
    setup = CalibrationCode.build_chamber()
    CalibrationCode.configure_lasers!(setup, f_cl, f_sb, A)

    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    h = IonSim.hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)

    psi = ca["S"] ⊗ ca["S"] ⊗ mode[0]
    tout = 0:dt:t
    for _ in 1:numMS
        _, sol = QuantumOptics.timeevolution.schroedinger_dynamic(tout, psi, h)
        psi = sol[end]
    end

    return real(IonSim.expect(IonSim.ionprojector(chamber, "D", "D"), psi))
end

# -----------------------------------------------------------------------
# Generate one landscape heatmap
# -----------------------------------------------------------------------
function plot_landscape(; t::Float64=100.0,
    n_sb::Int=40,
    n_A::Int=40,
    sb_span_kHz::Float64=2.0,         # ± 2×2π kHz
    A_ratio_range::Tuple{Float64,Float64}=(0.75, 1.25),
    numMS::Int=2,
    outfile::String="landscape.png")

    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    # Sideband detuning offset in units of 2π kHz → convert to Hz
    sb_span_Hz = sb_span_kHz * 1e3 * 2π
    Δfsb_range = range(-sb_span_Hz, sb_span_Hz; length=n_sb)       # offset from f_sb0
    Δfsb_kHz = collect(Δfsb_range ./ (2π * 1e3))                  # for axis label

    A_range = range(A_ratio_range[1] * A0, A_ratio_range[2] * A0; length=n_A)
    A_ratio = collect(A_range ./ A0)

    params = [(j, i, a_val, Δf) for (j, a_val) in enumerate(A_range) for (i, Δf) in enumerate(Δfsb_range)]
    results = pmap(params; batch_size=max(1, div(n_sb * n_A, 100))) do p
        j_idx, i_idx, a_val, Δf = p
        val = P_DD_det(t, f_cl0, f_sb0 + Δf, a_val; numMS=numMS)
        (j_idx, i_idx, val)
    end

    Z = Matrix{Float64}(undef, n_A, n_sb)
    for (j_idx, i_idx, val) in results
        Z[j_idx, i_idx] = val
    end

    p = heatmap(Δfsb_kHz, A_ratio, Z;
        xlabel="Δf_sb  (2π kHz)",
        ylabel="A / A₀",
        title="P(|DD⟩)  –  $(numMS) MS gates",
        color=:plasma,
        clims=(0.0, 1.0),
        xtickfont=font(8), ytickfont=font(8),
        guidefont=font(10), titlefont=font(11),
        left_margin=7mm, bottom_margin=7mm,
        colorbar_title=" P",
        dpi=200)

    savefig(p, outfile)
    println("Saved -> $outfile")
    return p
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
function main()
    outdir = joinpath(@__DIR__, "plots")
    isdir(outdir) || mkdir(outdir)

    t = 100.0
    n = 30       # grid points per axis (increase for higher resolution)

    println("Generating landscape plots (3 MS gate counts)...")
    for numMS in [6, 10]
        println("  numMS = $numMS ...")
        plot_landscape(;
            t=t, n_sb=n, n_A=n,
            sb_span_kHz=2.0,
            A_ratio_range=(0.75, 1.25),
            numMS=numMS,
            outfile=joinpath(outdir, "landscape_$(numMS)MS.png"))
    end

    println("Done!")
end

main()
