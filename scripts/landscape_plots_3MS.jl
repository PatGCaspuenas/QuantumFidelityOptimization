# scripts/landscape_plots_3MS.jl
#
# Generates landscape contourf plots scanning 
#   Δf_sb ∈ [-2, +2] × 2π kHz  vs.  Ω/Ω₀ ∈ [0.75, 1.25]
#   Δf_cl ∈ [-2, +2] × 2π kHz  vs.  Ω/Ω₀ ∈ [0.75, 1.25]
#
# for 2 and 3 sequential MS gates.
# - 2 MS gates: measures P(|DD⟩)
# - 3 MS gates: measures closeness to 50/50 |00⟩/|11⟩ state: 1 - |P(SS) - 0.5| - |P(DD) - 0.5| - P(SD) - P(DS)
#
# Run with: julia scripts/landscape_plots_3MS.jl

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
@everywhere using QuantumOptics

using Plots
using Measures

# -----------------------------------------------------------------------
# Deterministic Eval for Multi-MS gate Sequences
# -----------------------------------------------------------------------
@everywhere function eval_multi_MS(t, f_cl, f_sb, A; numMS::Int=2)
    setup = CalibrationCode.build_chamber()
    CalibrationCode.configure_lasers!(setup, f_cl, f_sb, A)

    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    h = IonSim.hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)

    psi = ca["S"] ⊗ ca["S"] ⊗ mode[0]
    tout = [0.0, t]
    for _ in 1:numMS
        _, sol = QuantumOptics.timeevolution.schroedinger_dynamic(tout, psi, h)
        psi = sol[end]
    end
    
    # Measurements
    P_SS = real(IonSim.expect(IonSim.ionprojector(chamber, "S", "S"), psi))
    P_DD = real(IonSim.expect(IonSim.ionprojector(chamber, "D", "D"), psi))
    P_SD = real(IonSim.expect(IonSim.ionprojector(chamber, "S", "D"), psi))
    P_DS = real(IonSim.expect(IonSim.ionprojector(chamber, "D", "S"), psi))

    if numMS == 2
        # Target is perfect |DD> population
        return P_DD
    else
        # Target is perfect 50/50 |SS> / |DD> superposition.
        # So we penalize distance from 0.5 for both, and heavily penalize SD/DS populations.
        # Maximum score is 1.0 when SS=0.5, DD=0.5, SD=0.0, DS=0.0.
        score = 1.0 - abs(P_SS - 0.5) - abs(P_DD - 0.5) - P_SD - P_DS
        return max(0.0, score) # Clamp bottom at 0
    end
end

# -----------------------------------------------------------------------
# Generate landscape (Heatmap over 2 params)
# -----------------------------------------------------------------------
function plot_landscape_fsb(; t::Float64=100.0, n_grid::Int=50, numMS::Int=2, outfile::String="landscape_fsb.png")
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    # ranges
    span_kHz = 2.0
    span_Hz = span_kHz * 1e3 * 2π
    Δfsb_range = range(-span_Hz, span_Hz; length=n_grid)
    Δfsb_kHz = collect(Δfsb_range ./ (2π * 1e3))
    
    A_range = range(0.75 * A0, 1.25 * A0; length=n_grid)
    A_ratio = collect(A_range ./ A0)

    params = [(j, i, a_val, Δf) for (j, a_val) in enumerate(A_range) for (i, Δf) in enumerate(Δfsb_range)]
    results = pmap(params; batch_size=max(1, div(n_grid * n_grid, 100))) do p
        j_idx, i_idx, a_val, Δf = p
        val = eval_multi_MS(t, f_cl0, f_sb0 + Δf, a_val; numMS=numMS)
        (j_idx, i_idx, val)
    end

    Z = Matrix{Float64}(undef, n_grid, n_grid)
    for (j_idx, i_idx, val) in results
        Z[j_idx, i_idx] = val
    end

    title_str = numMS == 2 ? "2 MS Gates (P|DD⟩)" : "3 MS Gates (50/50 |00⟩/|11⟩)"

    p = contourf(Δfsb_kHz, A_ratio, Z;
                 xlabel="Δf_sb (2π kHz)",
                 ylabel="Ω / Ω₀",
                 title=title_str,
                 color=:plasma,
                 clims=(0.0, 1.0),
                 xtickfont=font(8), ytickfont=font(8),
                 guidefont=font(10), titlefont=font(11),
                 left_margin=7mm, bottom_margin=7mm)
    
    savefig(p, outfile)
    println("Saved -> $outfile")
end

function plot_landscape_fcl(; t::Float64=100.0, n_grid::Int=50, numMS::Int=2, outfile::String="landscape_fcl.png")
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    # ranges
    span_kHz = 2.0
    span_Hz = span_kHz * 1e3 * 2π
    Δfcl_range = range(-span_Hz, span_Hz; length=n_grid)
    Δfcl_kHz = collect(Δfcl_range ./ (2π * 1e3))
    
    A_range = range(0.75 * A0, 1.25 * A0; length=n_grid)
    A_ratio = collect(A_range ./ A0)

    params = [(j, i, a_val, Δf) for (j, a_val) in enumerate(A_range) for (i, Δf) in enumerate(Δfcl_range)]
    results = pmap(params; batch_size=max(1, div(n_grid * n_grid, 100))) do p
        j_idx, i_idx, a_val, Δf = p
        val = eval_multi_MS(t, f_cl0 + Δf, f_sb0, a_val; numMS=numMS)
        (j_idx, i_idx, val)
    end

    Z = Matrix{Float64}(undef, n_grid, n_grid)
    for (j_idx, i_idx, val) in results
        Z[j_idx, i_idx] = val
    end

    title_str = numMS == 2 ? "2 MS Gates (P|DD⟩)" : "3 MS Gates (50/50 |00⟩/|11⟩)"

    p = contourf(Δfcl_kHz, A_ratio, Z;
                 xlabel="Δf_cl (2π kHz)",
                 ylabel="Ω / Ω₀",
                 title=title_str,
                 color=:plasma,
                 clims=(0.0, 1.0),
                 xtickfont=font(8), ytickfont=font(8),
                 guidefont=font(10), titlefont=font(11),
                 left_margin=7mm, bottom_margin=7mm)
    
    savefig(p, outfile)
    println("Saved -> $outfile")
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    outdir = joinpath(@__DIR__, "plots")
    isdir(outdir) || mkdir(outdir)

    t = 100.0
    n = 50

    println("Generating 2 and 3 MS Gate Landscapes...")
    
    plot_landscape_fsb(; t=t, n_grid=n, numMS=2, outfile=joinpath(outdir, "landscape_fsb_2MS.png"))
    plot_landscape_fcl(; t=t, n_grid=n, numMS=2, outfile=joinpath(outdir, "landscape_fcl_2MS.png"))
    
    plot_landscape_fsb(; t=t, n_grid=n, numMS=3, outfile=joinpath(outdir, "landscape_fsb_3MS.png"))
    plot_landscape_fcl(; t=t, n_grid=n, numMS=3, outfile=joinpath(outdir, "landscape_fcl_3MS.png"))

    println("Done!")
end
