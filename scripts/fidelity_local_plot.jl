# scripts/fidelity_local_scan.jl

using Plots
using Measures
using CalibrationCode

"""
    local_scan_fcl_fsb(; t=100.0, span=2e4, n=100, outfile="fidelity_local_scan_offsets.png",
                        method=:det, N=200, verbose=false)

Creates a 2D heatmap of fidelity around the ideal baseline, scanning (f_cl, f_sb)
while keeping A fixed at its baseline value.

- `span`: half-width of scan interval (±span around baseline)
- `n`: grid points per axis
- `method`: :det uses Q_det; :noisy uses Q_noisy (slower)
- `N`: number of shots/samples (used only if method=:noisy)
"""
function local_scan_fcl_fsb(; t::Float64=100.0,
                             span::Float64=2e4,
                             n::Int=100,
                             outfile::String="fidelity_local_scan_offsets.png",
                             method::Symbol=:det,
                             N::Int=200,
                             verbose::Bool=false)

    N ≥ 1 || throw(ArgumentError("N must be ≥ 1"))

    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    println("Baseline (t=$(t) μs):")
    println(" f_cl0 = $(round(f_cl0, digits=3))")
    println(" f_sb0 = $(round(f_sb0, digits=3))")
    println(" A0    = $(round(A0,    digits=3))")
    println(" fid0  = $(round(base.fid, digits=6))")

    f_cl_range = range(f_cl0 - span, f_cl0 + span; length=n)
    f_sb_range = range(f_sb0 - span, f_sb0 + span; length=n)

    f_cl_ticks = collect(f_cl_range .- f_cl0)
    f_sb_ticks = collect(f_sb_range .- f_sb0)

    qfun = method === :noisy ?
        ((fcl, fsb) -> CalibrationCode.Q_noisy(t, fcl, fsb, A0; N=N)) :
        ((fcl, fsb) -> CalibrationCode.Q_det(t, fcl, fsb, A0))

    function safe_eval(fcl, fsb)
        try
            return qfun(fcl, fsb)
        catch err
            if verbose
                @warn "Evaluation failed" fcl fsb err
            end
            return NaN
        end
    end

    # Z shape: (n_fsb, n_fcl) for heatmap(x, y, Z)
    Z = Matrix{Float64}(undef, length(f_sb_range), length(f_cl_range))
    for (j, fsb) in enumerate(f_sb_range)
        verbose && println("row $j / $(length(f_sb_range))")
        for (i, fcl) in enumerate(f_cl_range)
            Z[j, i] = safe_eval(fcl, fsb)
        end
    end

    finite_vals = filter(isfinite, vec(Z))
    isempty(finite_vals) && error("All evaluations failed (all NaN/Inf).")

    cmin, cmax = minimum(finite_vals), maximum(finite_vals)

    note = "Axes centered on baseline:\n" *
           "f_cl0=$(round(f_cl0,digits=3)), f_sb0=$(round(f_sb0,digits=3)), A0=$(round(A0,digits=3))\n" *
           "method=$(method)" * (method === :noisy ? ", N=$N" : "")

    p = heatmap(f_cl_ticks, f_sb_ticks, Z;
                xlabel="Δf_cl", ylabel="Δf_sb",
                title="Fidelity local scan (A fixed)",
                xtickfont=font(7), ytickfont=font(7),
                guidefont=font(9), titlefont=font(10),
                left_margin=6mm, bottom_margin=6mm,
                clims=(cmin, cmax), color=:plasma)

    annotate!(p, (minimum(f_cl_ticks), maximum(f_sb_ticks), text(note, 8, :left)))

    savefig(p, outfile)
    println("Saved -> $outfile")

    return p
end

if abspath(PROGRAM_FILE) == @__FILE__
    local_scan_fcl_fsb(; t=100.0, span=2e4, n=100,
                        outfile="fidelity_local_scan_offsets.png",
                        method=:det, N=200, verbose=false)
end
